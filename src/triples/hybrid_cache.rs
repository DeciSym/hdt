//! Cache file format for HybridTripleAccess
//!
//! This module provides functionality to serialize/deserialize the in-memory
//! structures used by HybridTripleAccess, allowing them to be prebuilt from
//! TriplesBitmap and reused.
//!
//! Cache file format (.hdt.index.v5-rust-cache):
//! ```text
//! [ControlInfo]                     (HDT ControlInfo structure with type=Index)
//!   - format: "<http://purl.org/HDT/hdt#cacheV5>"
//!   - properties["order"]           (SPO=1, SOP=2, PSO=3, etc.)
//!   - properties["numTriples"]      (total number of triples)
//!   - properties["headerSize"]      (size of HDT header section in bytes)
//! [Wavelet Y]                       (variable - bincode serialized QWaveletTree)
//! [Bitmap Y Offset: u64]            (8 bytes - offset in HDT file where bitmap_y begins)
//! [Bitmap Z Offset: u64]            (8 bytes - offset in HDT file where bitmap_z begins)
//! [Sequence Z Offset: u64]          (8 bytes - offset in HDT file where sequence_z begins)
//! [Dictionary Offset: u64]          (8 bytes - offset in HDT file where Dictionary section begins)
//! [Dict Shared Offset: u64]         (8 bytes - offset where shared dictionary section begins)
//! [Dict Subjects Offset: u64]       (8 bytes - offset where subjects dictionary section begins)
//! [Dict Predicates Offset: u64]     (8 bytes - offset where predicates dictionary section begins)
//! [Dict Objects Offset: u64]        (8 bytes - offset where objects dictionary section begins)
//! [Triples Offset: u64]             (8 bytes - offset in HDT file where Triples section begins)
//! [Op Index Bitmap]                 (variable - serialized Bitmap, offset returned by read_from_file())
//! [Op Index Sequence]               (variable - serialized Sequence)
//! ```
//!
//! ## Design Rationale
//! - **Stored in cache (in memory)**: wavelet_y - computed structure, expensive to rebuild, always loaded
//! - **Stored in cache (on disk)**: op_index.bitmap, op_index.sequence - can be accessed on-demand or mmapped
//! - **File offsets only**: bitmap_y, bitmap_z - read directly from HDT file on-demand
//! - **File offsets only**: sequence_z - metadata parsed during MmapSequence::from_mmap()
//! - **Version 5 changes**: Use ControlInfo structure, moved order/numTriples/headerSize to properties

use crate::containers::AdjList;
use crate::containers::Bitmap;
use crate::containers::ControlInfo;
use crate::containers::Sequence;
use crate::containers::bitmap_access::build_rank_index;
use crate::header::Header;
use crate::triples::Order;
use crate::triples::TriplesBitmap;
use crate::triples::WT;
use bytesize::ByteSize;
use fs2::FileExt;
use log::debug;
use log::warn;
use mem_dbg::{MemSize, SizeFlags};
use std::collections::hash_map::DefaultHasher;
use std::fmt;
use std::fs::{File, OpenOptions};
use std::hash::{Hash, Hasher};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

pub const CACHE_EXT: &str = "index.v5-rust-cache";
const CACHE_FORMAT: &str = "<http://purl.org/HDT/hdt#cacheV5>";

/// Typed error for everything that can go wrong when loading, validating,
/// or generating a `HybridCache`. Replaces the previous `Box<dyn Error>`
/// surface so callers (notably `Hdt::new_hybrid_cache`) get structured
/// errors with intact source chains.
#[derive(thiserror::Error, Debug)]
pub enum HybridCacheError {
    #[error("IO error")]
    Io(#[from] std::io::Error),
    #[error("failed to read HDT control info")]
    ControlInfo(#[from] crate::containers::control_info::Error),
    #[error("failed to read HDT header")]
    Header(#[from] crate::header::Error),
    #[error("failed to read HDT bitmap")]
    Bitmap(#[from] crate::containers::bitmap::Error),
    #[error("failed to read HDT sequence")]
    Sequence(#[from] crate::containers::sequence::Error),
    #[error("failed to read HDT dictionary section")]
    DictSect(#[from] crate::dict_sect_pfc::Error),
    #[error("failed to construct HDT triples section")]
    Triples(#[from] crate::triples::Error),
    #[error("bincode decode error")]
    BincodeDecode(#[from] bincode::error::DecodeError),
    #[error("bincode encode error")]
    BincodeEncode(#[from] bincode::error::EncodeError),
    #[error("invalid cache control type: expected Index, found {found:?}")]
    InvalidControlType { found: crate::containers::ControlType },
    #[error("unsupported cache format: expected {expected}, found {found}")]
    UnsupportedFormat { expected: String, found: String },
    #[error("missing required cache property: {0}")]
    MissingProperty(&'static str),
    #[error("invalid value for cache property {key}: {value}")]
    InvalidProperty { key: &'static str, value: String },
    #[error("HDT file has no triples; hybrid cache cannot be created")]
    EmptyHdt,
    #[error("DictSectPFC read worker thread panicked")]
    DictSectThreadPanic,
    #[error("failed to canonicalize HDT path {path}")]
    Canonicalize {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to {op} cache lock file {path}")]
    Lock {
        op: &'static str,
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to atomically rename cache temp {tmp} to {final_path}")]
    Rename {
        tmp: String,
        final_path: String,
        #[source]
        source: std::io::Error,
    },
}

pub type HybridCacheResult<T> = core::result::Result<T, HybridCacheError>;

/// Canonicalize the HDT path so that two callers that reach the same file
/// via different relative paths still hash to the same lock file.
fn canonical_hdt_path(path: &Path) -> HybridCacheResult<PathBuf> {
    path.canonicalize()
        .map_err(|source| HybridCacheError::Canonicalize { path: path.display().to_string(), source })
}

/// Build the path of the lock file used to serialize cache generation for
/// the given canonical HDT path. Lives in the system temp directory under
/// `hdt-hybrid-cache-locks/`, keyed by the FNV-style hash of the path.
fn cache_lock_file_path(canonical_hdt_path: &Path) -> HybridCacheResult<PathBuf> {
    let mut hasher = DefaultHasher::new();
    canonical_hdt_path.as_os_str().hash(&mut hasher);
    let lock_name = format!("hdt-hybrid-cache-{:016x}.lock", hasher.finish());
    let lock_root = std::env::temp_dir().join("hdt-hybrid-cache-locks");
    std::fs::create_dir_all(&lock_root).map_err(|source| HybridCacheError::Lock {
        op: "create directory for",
        path: lock_root.display().to_string(),
        source,
    })?;
    Ok(lock_root.join(lock_name))
}

fn open_cache_lock_file(canonical_hdt_path: &Path) -> HybridCacheResult<(File, PathBuf)> {
    let lock_path = cache_lock_file_path(canonical_hdt_path)?;
    let lock_file =
        OpenOptions::new().read(true).write(true).create(true).truncate(false).open(&lock_path).map_err(
            |source| HybridCacheError::Lock { op: "open", path: lock_path.display().to_string(), source },
        )?;
    Ok((lock_file, lock_path))
}

fn unlock_cache_lock(lock_file: &File, lock_path: &Path, mode: &'static str) -> HybridCacheResult<()> {
    FileExt::unlock(lock_file).map_err(|source| HybridCacheError::Lock {
        op: match mode {
            "shared" => "release shared",
            "exclusive" => "release exclusive",
            _ => "release",
        },
        path: lock_path.display().to_string(),
        source,
    })
}

/// Advance the reader past a DictSectPFC section without allocating its
/// packed data or computing its CRC32. Only the header bytes (preamble +
/// 3 vbytes + CRC8) and the embedded Sequence header are parsed; the bulk
/// data (packed strings + CRC32) is seeked past.
///
/// This is used during cache generation: the cache writer needs the stream
/// position of each section but never inspects the dictionary data itself.
/// Skipping avoids allocating ~1.5 GB for the largest section and saves
/// the CRC32 computation over that data.
fn skip_dict_section<R: std::io::BufRead + std::io::Seek>(reader: &mut R) -> HybridCacheResult<()> {
    use crate::containers::vbyte::read_vbyte;
    use std::io::SeekFrom;

    // [preamble: u8] [num_strings: vbyte] [packed_length: vbyte] [block_size: vbyte] [CRC8: u8]
    let mut buf = [0u8];
    reader.read_exact(&mut buf)?; // preamble
    let _ = read_vbyte(reader)?; // num_strings
    let (packed_length, _) = read_vbyte(reader)?; // packed_length — needed for seek
    let _ = read_vbyte(reader)?; // block_size
    reader.read_exact(&mut buf)?; // CRC8

    // [Sequence section: header + data + CRC32]
    skip_sequence(reader)?;

    // [packed_data: packed_length bytes] [CRC32: 4 bytes]
    reader.seek(SeekFrom::Current(packed_length as i64 + 4))?;
    Ok(())
}

/// Advance the reader past a Sequence section without reading its data.
/// Parses the header to determine the data length, then seeks past data +
/// CRC32.
fn skip_sequence<R: std::io::BufRead + std::io::Seek>(reader: &mut R) -> HybridCacheResult<()> {
    use crate::containers::vbyte::read_vbyte;
    use std::io::SeekFrom;

    // [type: u8] [bits_per_entry: u8] [entries: vbyte] [CRC8: u8]
    let mut buf = [0u8];
    reader.read_exact(&mut buf)?; // type
    reader.read_exact(&mut buf)?; // bits_per_entry
    let bits_per_entry = buf[0] as u64;
    let (entries, _) = read_vbyte(reader)?; // entries
    reader.read_exact(&mut buf)?; // CRC8

    // Data is byte-aligned: ceil(entries * bits_per_entry / 8) bytes,
    // followed by a 4-byte CRC32 trailer.
    let data_bytes = (entries as u64 * bits_per_entry).div_ceil(8);
    reader.seek(SeekFrom::Current(data_bytes as i64 + 4))?;
    Ok(())
}

/// Cached structures for HybridTripleAccess
///
/// ## Storage Strategy:
/// - **In cache (in memory)**: wavelet_y — computed/built structures loaded into memory
/// - **In cache (on disk)**: rank indices for bitmap_y, bitmap_z, and
///   op_index.bitmap; op_index.bitmap itself; op_index.sequence. All mmaped
///   at load time and accessed via slice indexing.
/// - **HDT file offsets**: bitmap_y, bitmap_z, sequence_z, dictionary sections —
///   read from HDT file on-demand via shared mmap.
/// - **Metadata in ControlInfo**: order, numTriples, headerSize stored in properties.
pub struct HybridCache {
    /// Control information containing metadata (order, numTriples, headerSize)
    pub control_info: ControlInfo,
    /// Wavelet tree (stored in cache file, always loaded into memory)
    pub wavelet_y: WT,
    /// File offset where bitmap_y begins in HDT file
    pub bitmap_y_offset: u64,
    /// File offset where bitmap_z (adjlist_z.bitmap) begins in HDT file
    pub bitmap_z_offset: u64,
    /// File offset where sequence_z (adjlist_z.sequence) begins in HDT file
    pub sequence_z_offset: u64,
    /// File offset where Dictionary section begins in HDT file
    pub dictionary_offset: u64,
    /// File offset where shared dictionary section begins
    pub dict_shared_offset: u64,
    /// File offset where subjects dictionary section begins
    pub dict_subjects_offset: u64,
    /// File offset where predicates dictionary section begins
    pub dict_predicates_offset: u64,
    /// File offset where objects dictionary section begins
    pub dict_objects_offset: u64,
    /// File offset where Triples section begins in HDT file
    pub triples_offset: u64,
    /// Offset inside the cache file where the bitmap_y rank index begins.
    /// A dense `[u64]` array; `MmapBitmap::from_mmap_with_rank_index` wraps it.
    pub bitmap_y_rank_index_offset: u64,
    /// Offset inside the cache file where the bitmap_z (= adjlist_z.bitmap)
    /// rank index begins.
    pub bitmap_z_rank_index_offset: u64,
    /// Offset inside the cache file where the op_index.bitmap rank index begins.
    pub op_index_bitmap_rank_index_offset: u64,
    /// Offset inside the cache file where the op_index bitmap section begins.
    /// Stored explicitly because its position now depends on the combined
    /// size of the three rank indices that precede it, which can't be
    /// reconstructed from the fixed-size header alone.
    pub op_index_bitmap_offset: u64,
}

impl fmt::Debug for HybridCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "in-memory size {}: {{ {} wavelet_y }}",
            ByteSize(self.wavelet_y.mem_size(SizeFlags::default()) as u64),
            ByteSize(self.wavelet_y.mem_size(SizeFlags::default()) as u64),
        )
    }
}

impl HybridCache {
    /// Get the triple ordering from cache metadata
    pub fn order(&self) -> HybridCacheResult<Order> {
        let value = self.control_info.get("order").ok_or(HybridCacheError::MissingProperty("order"))?;
        let parsed = value
            .parse::<u8>()
            .map_err(|_| HybridCacheError::InvalidProperty { key: "order", value: value.clone() })?;
        Order::try_from(parsed as u32).map_err(|_| HybridCacheError::InvalidProperty { key: "order", value })
    }

    /// Get the number of triples from cache metadata
    pub fn num_triples(&self) -> HybridCacheResult<usize> {
        let value = self.control_info.get("numTriples").ok_or(HybridCacheError::MissingProperty("numTriples"))?;
        value.parse::<usize>().map_err(|_| HybridCacheError::InvalidProperty { key: "numTriples", value })
    }

    /// Get the header size from cache metadata
    pub fn header_size(&self) -> HybridCacheResult<u64> {
        let value = self.control_info.get("headerSize").ok_or(HybridCacheError::MissingProperty("headerSize"))?;
        value.parse::<u64>().map_err(|_| HybridCacheError::InvalidProperty { key: "headerSize", value })
    }
}

impl HybridCache {
    /// Smart constructor: Load cache if exists, otherwise create it.
    ///
    /// Uses a cross-process advisory file lock keyed on the canonical HDT
    /// path so that concurrent loaders never duplicate cache generation work
    /// and never observe a half-written cache file. The lock pattern is:
    ///
    /// 1. Acquire a shared lock and try to read an existing cache.
    /// 2. If the cache exists and parses, return it (shared lock allows
    ///    multiple concurrent readers).
    /// 3. Otherwise upgrade to an exclusive lock, re-check the cache, and
    ///    only generate if it is still missing or unreadable.
    /// 4. The writer always emits to a temporary file first and atomically
    ///    renames it into place, so readers either see the previous cache
    ///    or the next one — never a partial file.
    ///
    /// # Arguments
    /// * `hdt_path` - Path to the HDT file
    ///
    /// # Cache File Location
    /// `<hdt_filename>.index.v5-rust-cache`, alongside the HDT file.
    /// The lock file lives in `${TMPDIR}/hdt-hybrid-cache-locks/`.
    ///
    /// # Returns
    /// Returns a tuple `(HybridCache, u64)` where:
    /// - `HybridCache`: The loaded/created cache
    /// - `u64`: File offset in the cache file where the OpIndex bitmap begins
    pub fn from_hdt_path(hdt_path: impl AsRef<Path>) -> HybridCacheResult<(Self, u64)> {
        let hdt_path = hdt_path.as_ref();
        let canonical = canonical_hdt_path(hdt_path)?;
        let (lock_file, lock_path) = open_cache_lock_file(&canonical)?;

        let cache_path = Self::get_cache_path(hdt_path);

        // Reader path: shared lock allows many concurrent loaders to share an
        // already-built cache without blocking each other.
        FileExt::lock_shared(&lock_file).map_err(|source| HybridCacheError::Lock {
            op: "acquire shared",
            path: lock_path.display().to_string(),
            source,
        })?;

        if cache_path.exists() {
            debug!("Found existing cache: {}", cache_path.display());
            match Self::read_from_file(&cache_path) {
                Ok((cache, op_index_bitmap_offset)) => {
                    debug!("Loaded cache successfully");
                    debug!("{cache:#?}");
                    unlock_cache_lock(&lock_file, &lock_path, "shared")?;
                    return Ok((cache, op_index_bitmap_offset));
                }
                Err(e) => {
                    warn!("Cache file exists but couldn't be read: {e}");
                    warn!("Regenerating cache...");
                }
            }
        } else {
            debug!("Cache not found, generating from HDT file...");
        }

        unlock_cache_lock(&lock_file, &lock_path, "shared")?;

        // Writer path: serialize cache regeneration. We must release the
        // shared lock first because flock() upgrade is not portable.
        FileExt::lock_exclusive(&lock_file).map_err(|source| HybridCacheError::Lock {
            op: "acquire exclusive",
            path: lock_path.display().to_string(),
            source,
        })?;

        // Re-check after acquiring the exclusive lock — another process may
        // have generated the cache while we were waiting.
        if cache_path.exists() {
            debug!("Re-checking cache after acquiring exclusive lock");
            if let Ok((cache, op_index_bitmap_offset)) = Self::read_from_file(&cache_path) {
                unlock_cache_lock(&lock_file, &lock_path, "exclusive")?;
                return Ok((cache, op_index_bitmap_offset));
            }
            warn!("Cache remained unreadable under exclusive lock; regenerating...");
        }

        let generated = Self::write_cache_from_hdt_file(hdt_path);
        unlock_cache_lock(&lock_file, &lock_path, "exclusive")?;
        generated
    }

    /// Get the cache file path for a given HDT file
    pub fn get_cache_path(hdt_path: impl AsRef<Path>) -> std::path::PathBuf {
        let hdt_path = hdt_path.as_ref();
        let mut cache_path = hdt_path.to_path_buf();

        // Get the original filename
        let file_name = hdt_path.file_name().and_then(|n| n.to_str()).unwrap_or("unknown");

        // Append cache extension: myfile.hdt -> myfile.hdt.index.v5-rust-cache
        let cache_file_name = format!("{file_name}.{CACHE_EXT}");
        cache_path.set_file_name(cache_file_name);

        cache_path
    }

    pub fn write_cache_from_hdt_file(hdt_path: &Path) -> HybridCacheResult<(Self, u64)> {
        use crate::containers::ControlType;
        use std::collections::HashMap;
        use std::io::Seek;

        let mut reader = std::io::BufReader::new(std::fs::File::open(hdt_path)?);
        // Read control info (global header)
        ControlInfo::read(&mut reader)?;

        // Read header and get its size
        let header = Header::read(&mut reader)?;
        let header_size = header.length as u64;

        // Track dictionary offset (before control info)
        let dictionary_offset = reader.stream_position()?;

        // Read dictionary control info
        let _ = ControlInfo::read(&mut reader)?;

        // Track offsets for each dictionary section. We only need the
        // stream positions — the section data is not used during cache
        // generation. Skipping avoids allocating ~1.5 GB for the largest
        // section and saves the CRC32 computation over that data.
        let dict_shared_offset = reader.stream_position()?;
        skip_dict_section(&mut reader)?;

        let dict_subjects_offset = reader.stream_position()?;
        skip_dict_section(&mut reader)?;

        let dict_predicates_offset = reader.stream_position()?;
        skip_dict_section(&mut reader)?;

        let dict_objects_offset = reader.stream_position()?;
        skip_dict_section(&mut reader)?;

        // Track triples section offset
        let triples_offset = reader.stream_position()?;

        // Read triples control info
        let triples_ci = ControlInfo::read(&mut reader)?;

        // Track bitmap_y offset BEFORE reading it
        let bitmap_y_offset = reader.stream_position()?;
        let bitmap_y = Bitmap::read(&mut reader)?;

        // Track bitmap_z offset BEFORE reading it
        let bitmap_z_offset = reader.stream_position()?;
        let bitmap_z = Bitmap::read(&mut reader)?;

        // read sequences
        let sequence_y = Sequence::read(&mut reader)?;

        // Track sequence_z offset BEFORE reading it
        let sequence_z_offset = reader.stream_position()?;
        let sequence_z = Sequence::read(&mut reader)?;

        let order_value = triples_ci.get("order").ok_or(HybridCacheError::MissingProperty("order"))?;
        let order_num = order_value
            .parse::<u32>()
            .map_err(|_| HybridCacheError::InvalidProperty { key: "order", value: order_value.clone() })?;
        let order = Order::try_from(order_num)
            .map_err(|_| HybridCacheError::InvalidProperty { key: "order", value: order_value })?;
        let adjlist_z = AdjList::new(sequence_z, bitmap_z);

        // QWT library panics on empty data, so skip cache for empty HDT files.
        if adjlist_z.is_empty() {
            return Err(HybridCacheError::EmptyHdt);
        }

        let triples_bitmap = TriplesBitmap::new(order, &sequence_y, bitmap_y, adjlist_z);
        drop(sequence_y); // ~156 MB freed; not referenced again after wavelet build

        // Prepare a temporary cache file in the same directory, then atomically
        // rename it onto the final cache path once writing is complete. This
        // ensures readers (which may have the cache mmaped) never see a partial
        // file: a fresh inode appears under the final name, leaving any stale
        // mapping pointing at the old inode until it is dropped.
        let cache_path = Self::get_cache_path(hdt_path);
        let cache_dir = cache_path.parent().unwrap_or_else(|| Path::new("."));
        let mut tmp_cache = tempfile::NamedTempFile::new_in(cache_dir)?;
        let tmp_cache_path = tmp_cache.path().to_path_buf();
        let mut writer = BufWriter::new(tmp_cache.as_file_mut());

        // Create ControlInfo with metadata in properties
        let mut properties = HashMap::new();
        properties.insert("order".to_owned(), (triples_bitmap.order.clone() as u8).to_string());
        properties.insert("numTriples".to_owned(), triples_bitmap.adjlist_z.len().to_string());
        properties.insert("headerSize".to_owned(), header_size.to_string());
        let control_info =
            ControlInfo { control_type: ControlType::Index, format: CACHE_FORMAT.to_owned(), properties };

        // Write ControlInfo
        control_info.write(&mut writer)?;

        // Write wavelet_y using bincode
        bincode::serde::encode_into_std_write(
            &triples_bitmap.wavelet_y,
            &mut writer,
            bincode::config::standard(),
        )?;

        // Write all HDT file offsets
        writer.write_all(&bitmap_y_offset.to_le_bytes())?;
        writer.write_all(&bitmap_z_offset.to_le_bytes())?;
        writer.write_all(&sequence_z_offset.to_le_bytes())?;
        writer.write_all(&dictionary_offset.to_le_bytes())?;
        writer.write_all(&dict_shared_offset.to_le_bytes())?;
        writer.write_all(&dict_subjects_offset.to_le_bytes())?;
        writer.write_all(&dict_predicates_offset.to_le_bytes())?;
        writer.write_all(&dict_objects_offset.to_le_bytes())?;
        writer.write_all(&triples_offset.to_le_bytes())?;

        // Build rank indices for all three bitmaps. Each call pops through
        // the u64 words of the bitmap once, so total cost is ~3 sequential
        // scans — far smaller than the wavelet build that already ran.
        let bitmap_y_rank_index_words =
            build_rank_index(triples_bitmap.bitmap_y.inner().dict.bit_vector().words());
        let bitmap_z_rank_index_words =
            build_rank_index(triples_bitmap.adjlist_z.bitmap.inner().dict.bit_vector().words());
        let op_index_bitmap_rank_index_words =
            build_rank_index(triples_bitmap.op_index.bitmap.inner().dict.bit_vector().words());

        // Compute layout: after the 9 HDT offsets we write 4 more u64s
        // (three rank-index offsets plus the op_index_bitmap offset), then
        // the three rank-index byte arrays, then the op_index bitmap +
        // sequence. Everything downstream of this point is positionally
        // derived, not streamed, because readers need the rank index
        // offsets up-front.
        let header_end = writer.stream_position()?;
        let rank_header_bytes: u64 = 4 * 8;
        let bitmap_y_rank_index_offset = header_end + rank_header_bytes;
        let bitmap_y_rank_index_bytes = (bitmap_y_rank_index_words.len() * 8) as u64;
        let bitmap_z_rank_index_offset = bitmap_y_rank_index_offset + bitmap_y_rank_index_bytes;
        let bitmap_z_rank_index_bytes = (bitmap_z_rank_index_words.len() * 8) as u64;
        let op_index_bitmap_rank_index_offset = bitmap_z_rank_index_offset + bitmap_z_rank_index_bytes;
        let op_index_bitmap_rank_index_bytes = (op_index_bitmap_rank_index_words.len() * 8) as u64;
        let op_index_bitmap_offset = op_index_bitmap_rank_index_offset + op_index_bitmap_rank_index_bytes;

        // Write the 4 cache-file offsets. Their positions inside the cache
        // file are now fixed and recoverable by any reader.
        writer.write_all(&bitmap_y_rank_index_offset.to_le_bytes())?;
        writer.write_all(&bitmap_z_rank_index_offset.to_le_bytes())?;
        writer.write_all(&op_index_bitmap_rank_index_offset.to_le_bytes())?;
        writer.write_all(&op_index_bitmap_offset.to_le_bytes())?;

        // Sanity-check that we're about to write the rank indices at the
        // offset we promised readers they'd live at.
        debug_assert_eq!(writer.stream_position()?, bitmap_y_rank_index_offset);

        for word in &bitmap_y_rank_index_words {
            writer.write_all(&word.to_le_bytes())?;
        }
        debug_assert_eq!(writer.stream_position()?, bitmap_z_rank_index_offset);

        for word in &bitmap_z_rank_index_words {
            writer.write_all(&word.to_le_bytes())?;
        }
        debug_assert_eq!(writer.stream_position()?, op_index_bitmap_rank_index_offset);

        for word in &op_index_bitmap_rank_index_words {
            writer.write_all(&word.to_le_bytes())?;
        }
        debug_assert_eq!(writer.stream_position()?, op_index_bitmap_offset);

        // Write op_index.bitmap, then op_index.sequence.
        triples_bitmap.op_index.bitmap.inner().write(&mut writer)?;
        triples_bitmap.op_index.sequence.inner().write(&mut writer)?;

        // The second tuple element returned from `write_cache_from_hdt_file`
        // is the op_index_bitmap offset, so the caller can construct the
        // MmapBitmap view over the op-index without re-reading the cache.
        let op_index_offset = op_index_bitmap_offset;

        writer.flush()?;
        // Make sure every byte hits the disk before we swing the rename.
        // Without sync_all() a crash between rename and journal commit could
        // leave the new inode visible but empty.
        drop(writer);
        tmp_cache.as_file().sync_all()?;

        // Atomic publish: persist the temp file onto the final cache path.
        tmp_cache.persist(&cache_path).map_err(|error| HybridCacheError::Rename {
            tmp: tmp_cache_path.display().to_string(),
            final_path: cache_path.display().to_string(),
            source: error.error,
        })?;

        // Create and return the cache structure
        let cache = Self {
            control_info,
            wavelet_y: triples_bitmap.wavelet_y.clone(),
            bitmap_y_offset,
            bitmap_z_offset,
            sequence_z_offset,
            dictionary_offset,
            dict_shared_offset,
            dict_subjects_offset,
            dict_predicates_offset,
            dict_objects_offset,
            triples_offset,
            bitmap_y_rank_index_offset,
            bitmap_z_rank_index_offset,
            op_index_bitmap_rank_index_offset,
            op_index_bitmap_offset,
        };

        debug!("Cache generated and saved to: {}", cache_path.display());
        debug!("{cache:#?}");
        Ok((cache, op_index_offset))
    }

    /// Read cache from file, returning the cache structure and the offset to the OpIndex data
    ///
    /// # Returns
    /// Returns a tuple `(HybridCache, u64)` where:
    /// - `HybridCache`: The loaded cache with in-memory structures (wavelet_y only)
    /// - `u64`: File offset in the cache file where the OpIndex data begins (bitmap then sequence).
    ///   Callers can use this offset to construct both bitmap and sequence accessors.
    pub fn read_from_file<P: AsRef<Path>>(path: P) -> HybridCacheResult<(Self, u64)> {
        use crate::containers::ControlType;
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // Read and verify ControlInfo
        let control_info = ControlInfo::read(&mut reader)?;

        // Verify it's an Index type
        if control_info.control_type != ControlType::Index {
            return Err(HybridCacheError::InvalidControlType { found: control_info.control_type });
        }

        // Verify format
        if control_info.format != CACHE_FORMAT {
            return Err(HybridCacheError::UnsupportedFormat {
                expected: CACHE_FORMAT.to_owned(),
                found: control_info.format,
            });
        }

        // Read wavelet_y using bincode
        let wavelet_y: WT = bincode::serde::decode_from_std_read(&mut reader, bincode::config::standard())?;

        // Read HDT file offsets
        let mut bitmap_y_offset_bytes = [0u8; 8];
        reader.read_exact(&mut bitmap_y_offset_bytes)?;
        let bitmap_y_offset = u64::from_le_bytes(bitmap_y_offset_bytes);

        let mut bitmap_z_offset_bytes = [0u8; 8];
        reader.read_exact(&mut bitmap_z_offset_bytes)?;
        let bitmap_z_offset = u64::from_le_bytes(bitmap_z_offset_bytes);

        let mut sequence_z_offset_bytes = [0u8; 8];
        reader.read_exact(&mut sequence_z_offset_bytes)?;
        let sequence_z_offset = u64::from_le_bytes(sequence_z_offset_bytes);

        let mut dictionary_offset_bytes = [0u8; 8];
        reader.read_exact(&mut dictionary_offset_bytes)?;
        let dictionary_offset = u64::from_le_bytes(dictionary_offset_bytes);

        let mut dict_shared_offset_bytes = [0u8; 8];
        reader.read_exact(&mut dict_shared_offset_bytes)?;
        let dict_shared_offset = u64::from_le_bytes(dict_shared_offset_bytes);

        let mut dict_subjects_offset_bytes = [0u8; 8];
        reader.read_exact(&mut dict_subjects_offset_bytes)?;
        let dict_subjects_offset = u64::from_le_bytes(dict_subjects_offset_bytes);

        let mut dict_predicates_offset_bytes = [0u8; 8];
        reader.read_exact(&mut dict_predicates_offset_bytes)?;
        let dict_predicates_offset = u64::from_le_bytes(dict_predicates_offset_bytes);

        let mut dict_objects_offset_bytes = [0u8; 8];
        reader.read_exact(&mut dict_objects_offset_bytes)?;
        let dict_objects_offset = u64::from_le_bytes(dict_objects_offset_bytes);

        let mut triples_offset_bytes = [0u8; 8];
        reader.read_exact(&mut triples_offset_bytes)?;
        let triples_offset = u64::from_le_bytes(triples_offset_bytes);

        // Read the 4 cache-file offsets: three rank index offsets and the
        // op_index_bitmap offset. These must match the positions computed
        // by write_cache_from_hdt_file.
        let mut buf = [0u8; 8];
        reader.read_exact(&mut buf)?;
        let bitmap_y_rank_index_offset = u64::from_le_bytes(buf);
        reader.read_exact(&mut buf)?;
        let bitmap_z_rank_index_offset = u64::from_le_bytes(buf);
        reader.read_exact(&mut buf)?;
        let op_index_bitmap_rank_index_offset = u64::from_le_bytes(buf);
        reader.read_exact(&mut buf)?;
        let op_index_bitmap_offset = u64::from_le_bytes(buf);

        let cache = Self {
            control_info,
            wavelet_y,
            bitmap_y_offset,
            bitmap_z_offset,
            sequence_z_offset,
            dictionary_offset,
            dict_shared_offset,
            dict_subjects_offset,
            dict_predicates_offset,
            dict_objects_offset,
            triples_offset,
            bitmap_y_rank_index_offset,
            bitmap_z_rank_index_offset,
            op_index_bitmap_rank_index_offset,
            op_index_bitmap_offset,
        };

        Ok((cache, op_index_bitmap_offset))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Barrier};
    use std::thread;
    use std::time::{SystemTime, UNIX_EPOCH};

    /// Copy the snikmeta test HDT into a temp directory keyed on the test
    /// name and current pid+timestamp so that concurrent tests cannot
    /// collide on the same cache file.
    fn setup_isolated_hdt(test_name: &str) -> Result<(PathBuf, PathBuf, PathBuf), Box<dyn std::error::Error>> {
        let unique = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let test_dir = std::env::temp_dir()
            .join(format!("hdt-hybrid-cache-test-{test_name}-{}-{unique}", std::process::id()));
        std::fs::create_dir_all(&test_dir)?;

        let hdt_path = test_dir.join("snikmeta.hdt");
        std::fs::copy("tests/resources/snikmeta.hdt", &hdt_path)?;
        let cache_path = HybridCache::get_cache_path(&hdt_path);
        Ok((test_dir, hdt_path, cache_path))
    }

    #[test]
    fn test_from_hdt_path() -> Result<(), Box<dyn std::error::Error>> {
        crate::tests::init();
        let (test_dir, hdt_path, cache_path) = setup_isolated_hdt("single")?;
        let _ = std::fs::remove_file(&cache_path);

        println!("\n=== Test 1: First call (should generate cache) ===");
        let (cache1, offset1) = HybridCache::from_hdt_path(&hdt_path)?;
        assert!(cache_path.exists(), "Cache file should be created");
        println!("Cache size: {} bytes", std::fs::metadata(&cache_path)?.len());

        println!("\n=== Test 2: Second call (should load existing cache) ===");
        let (cache2, offset2) = HybridCache::from_hdt_path(&hdt_path)?;

        // Verify both caches are identical
        assert_eq!(cache1.order()? as u8, cache2.order()? as u8);
        assert_eq!(cache1.wavelet_y.len(), cache2.wavelet_y.len());
        assert_eq!(cache1.bitmap_y_offset, cache2.bitmap_y_offset);
        assert_eq!(cache1.bitmap_z_offset, cache2.bitmap_z_offset);
        assert_eq!(cache1.sequence_z_offset, cache2.sequence_z_offset);
        assert_eq!(offset1, offset2, "OpIndex offsets should match");

        println!("\nBoth caches are identical!");

        std::fs::remove_dir_all(test_dir)?;
        Ok(())
    }

    /// Hammer `from_hdt_path` from many threads with no preexisting cache.
    /// All workers should converge on a single valid cache without panics,
    /// torn writes, or duplicate generation racing.
    #[test]
    fn test_from_hdt_path_parallel_threads() -> Result<(), Box<dyn std::error::Error>> {
        crate::tests::init();
        let (test_dir, hdt_path, cache_path) = setup_isolated_hdt("parallel")?;
        let _ = std::fs::remove_file(&cache_path);

        let workers = 8_usize;
        let barrier = Arc::new(Barrier::new(workers));
        let mut handles = Vec::with_capacity(workers);
        for _ in 0..workers {
            let path = hdt_path.clone();
            let barrier = Arc::clone(&barrier);
            handles.push(thread::spawn(move || -> Result<(), String> {
                barrier.wait();
                HybridCache::from_hdt_path(&path).map(|_| ()).map_err(|e| e.to_string())
            }));
        }

        for handle in handles {
            match handle.join() {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    return Err(std::io::Error::other(format!(
                        "hybrid cache worker failed while loading cache: {e}"
                    ))
                    .into());
                }
                Err(_) => {
                    return Err(std::io::Error::other("hybrid cache worker thread panicked").into());
                }
            }
        }

        assert!(cache_path.exists(), "cache should exist after concurrent loads");
        std::fs::remove_dir_all(test_dir)?;
        Ok(())
    }

    #[test]
    fn test_from_hdt_path_thread_name_with_path_unsafe_chars() -> Result<(), Box<dyn std::error::Error>> {
        crate::tests::init();
        let (test_dir, hdt_path, cache_path) = setup_isolated_hdt("thread-name")?;
        let _ = std::fs::remove_file(&cache_path);

        let handle = thread::Builder::new().name("hdt::cache::worker".to_string()).spawn({
            let path = hdt_path.clone();
            move || HybridCache::from_hdt_path(&path).map(|_| ())
        })?;

        handle.join().map_err(|_| std::io::Error::other("hybrid cache worker thread panicked"))??;

        assert!(cache_path.exists(), "cache should exist after named-thread load");
        std::fs::remove_dir_all(test_dir)?;
        Ok(())
    }
}
