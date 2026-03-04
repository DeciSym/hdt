//! Cache file format for HybridTripleAccess
//!
//! This module provides functionality to serialize/deserialize the in-memory
//! structures used by HybridTripleAccess, allowing them to be prebuilt from
//! TriplesBitmap and reused.
//!
//! Cache file format (.hdt.index.v4-rust-cache):
//! ```text
//! [ControlInfo]                     (HDT ControlInfo structure with type=Index)
//!   - format: "<http://purl.org/HDT/hdt#cacheV4>"
//!   - properties["order"]           (SPO=1, SOP=2, PSO=3, etc.)
//!   - properties["numTriples"]      (total number of triples)
//!   - properties["headerSize"]      (size of HDT header section in bytes)
//! [Wavelet Y]                       (variable - sucds serialized WaveletMatrix)
//! [Bitmap Y Offset: u64]            (8 bytes - offset in HDT file where bitmap_y begins)
//! [Bitmap Z Offset: u64]            (8 bytes - offset in HDT file where bitmap_z begins)
//! [Sequence Z Offset: u64]          (8 bytes - offset in HDT file where sequence_z begins)
//! [Dictionary Offset: u64]          (8 bytes - offset in HDT file where Dictionary section begins)
//! [Dict Shared Offset: u64]         (8 bytes - offset where shared dictionary section begins)
//! [Dict Subjects Offset: u64]       (8 bytes - offset where subjects dictionary section begins)
//! [Dict Predicates Offset: u64]     (8 bytes - offset where predicates dictionary section begins)
//! [Dict Objects Offset: u64]        (8 bytes - offset where objects dictionary section begins)
//! [Triples Offset: u64]             (8 bytes - offset in HDT file where Triples section begins)
//! [Op Index Bitmap]                 (variable - sucds serialized Rank9Sel via Bitmap::write(), offset returned by read_from_file())
//! [Op Index Sequence]               (variable - sucds serialized CompactVector, offset = bitmap_offset + bitmap_size)
//! ```
//!
//! ## Design Rationale
//! - **Stored in cache (in memory)**: wavelet_y - computed structure, expensive to rebuild, always loaded
//! - **Stored in cache (on disk)**: op_index.bitmap, op_index.sequence - can be accessed on-demand or mmapped
//! - **File offsets only**: bitmap_y, bitmap_z - read directly from HDT file on-demand
//! - **File offsets only**: sequence_z - metadata read during FileBasedSequence::new()
//! - **Version 4 changes**: Use ControlInfo structure, moved order/numTriples/headerSize to properties

use crate::containers::AdjListGeneric;
use crate::containers::Bitmap;
use crate::containers::ControlInfo;
use crate::containers::InMemoryBitmap;
use crate::containers::InMemorySequence;
use crate::containers::Sequence;
use crate::header::Header;
use crate::triples::Order;
use crate::triples::TriplesBitmapGeneric;
use bytesize::ByteSize;
use fs2::FileExt;
use log::debug;
use log::warn;
use std::collections::hash_map::DefaultHasher;
use std::fmt;
use std::fs::{File, OpenOptions};
use std::hash::{Hash, Hasher};
use std::io::Seek;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use sucds::Serializable;
use sucds::bit_vectors::Rank9Sel;
use sucds::char_sequences::WaveletMatrix;

pub const CACHE_EXT: &str = "index.v5-rust-cache";
const CACHE_FORMAT: &str = "<http://purl.org/HDT/hdt#cacheV4>";

fn boxed_io_error(message: impl Into<String>) -> Box<dyn std::error::Error> {
    std::io::Error::other(message.into()).into()
}

fn canonical_hdt_path(path: &Path) -> Result<PathBuf, Box<dyn std::error::Error>> {
    path.canonicalize()
        .map_err(|e| boxed_io_error(format!("failed to canonicalize HDT path {}: {e}", path.display())))
}

fn cache_lock_file_path(canonical_hdt_path: &Path) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let mut hasher = DefaultHasher::new();
    canonical_hdt_path.as_os_str().hash(&mut hasher);
    let lock_name = format!("hdt-hybrid-cache-{:016x}.lock", hasher.finish());
    let lock_root = std::env::temp_dir().join("hdt-hybrid-cache-locks");
    std::fs::create_dir_all(&lock_root).map_err(|e| {
        boxed_io_error(format!("failed to create cache lock directory {}: {e}", lock_root.display()))
    })?;
    Ok(lock_root.join(lock_name))
}

fn open_cache_lock_file(canonical_hdt_path: &Path) -> Result<(File, PathBuf), Box<dyn std::error::Error>> {
    let lock_path = cache_lock_file_path(canonical_hdt_path)?;
    let lock_file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&lock_path)
        .map_err(|e| boxed_io_error(format!("failed to open cache lock file {}: {e}", lock_path.display())))?;
    Ok((lock_file, lock_path))
}

fn unlock_cache_lock(
    lock_file: &File, lock_path: &Path, hdt_path: &Path, mode: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    FileExt::unlock(lock_file).map_err(|e| {
        boxed_io_error(format!(
            "failed to release {mode} cache lock {} for {}: {e}",
            lock_path.display(),
            hdt_path.display()
        ))
    })
}

/// Cached structures for HybridTripleAccess
///
/// ## Storage Strategy:
/// - **In cache (in memory)**: wavelet_y - computed/built structures loaded into memory
/// - **In cache (on disk)**: op_index.bitmap and op_index.sequence - written at end of cache file, offsets returned by read_from_file()
/// - **HDT file offsets**: bitmap_y, bitmap_z, sequence_z, dictionary sections - read from HDT file on-demand
/// - **Metadata in ControlInfo**: order, numTriples, headerSize stored in properties
pub struct HybridCache {
    /// Control information containing metadata (order, numTriples, headerSize)
    pub control_info: ControlInfo,
    /// Wavelet matrix (stored in cache file, always loaded into memory)
    pub wavelet_y: WaveletMatrix<Rank9Sel>,
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
}

impl fmt::Debug for HybridCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "in-memory size {}: {{ {} wavelet_y }}",
            ByteSize(self.wavelet_y.size_in_bytes() as u64),
            ByteSize(self.wavelet_y.size_in_bytes() as u64),
        )
    }
}

impl HybridCache {
    /// Get the triple ordering from cache metadata
    pub fn order(&self) -> Result<Order, Box<dyn std::error::Error>> {
        let order_str = self.control_info.get("order").ok_or("order property not found in cache")?;
        let order_value = order_str.parse::<u8>()?;
        Order::try_from(order_value as u32).map_err(|e| format!("Invalid order value: {e}").into())
    }

    /// Get the number of triples from cache metadata
    pub fn num_triples(&self) -> Result<usize, Box<dyn std::error::Error>> {
        let num_triples_str =
            self.control_info.get("numTriples").ok_or("numTriples property not found in cache")?;
        Ok(num_triples_str.parse::<usize>()?)
    }

    /// Get the header size from cache metadata
    pub fn header_size(&self) -> Result<u64, Box<dyn std::error::Error>> {
        let header_size_str =
            self.control_info.get("headerSize").ok_or("headerSize property not found in cache")?;
        Ok(header_size_str.parse::<u64>()?)
    }
}

impl HybridCache {
    /// Smart constructor: Load cache if exists, otherwise create it
    ///
    /// This is the recommended way to create a HybridCache. It automatically:
    /// 1. Checks if a cache file exists for the given HDT file
    /// 2. If found, loads the existing cache
    /// 3. If not found, generates the cache from the HDT file and saves it
    ///
    /// # Arguments
    /// * `hdt_path` - Path to the HDT file
    ///
    /// # Cache File Location
    /// The cache file is stored in the same directory as the HDT file with the naming convention:
    /// `<hdt_filename>.index.v5-rust-cache`
    ///
    /// Cache generation is guarded by a cross-process file lock in the system temp
    /// directory, keyed by canonical HDT path.
    ///
    /// # Example
    /// ```ignore
    /// let cache = HybridCache::from_hdt_path("data/myfile.hdt")?;
    /// // First call: generates cache and saves to "data/myfile.hdt.index.v5-rust-cache"
    /// // Second call: loads existing cache (much faster!)
    /// ```
    /// Load or create cache, returning the cache and the offset to the OpIndex bitmap in the cache file
    ///
    /// # Returns
    /// Returns a tuple `(HybridCache, u64)` where:
    /// - `HybridCache`: The loaded/created cache
    /// - `u64`: File offset in the cache file where the OpIndex bitmap begins
    pub fn from_hdt_path(hdt_path: impl AsRef<Path>) -> Result<(Self, u64), Box<dyn std::error::Error>> {
        let hdt_path = hdt_path.as_ref();
        let canonical_hdt_path = canonical_hdt_path(hdt_path)?;
        let (lock_file, lock_file_path) = open_cache_lock_file(&canonical_hdt_path)?;

        // Construct cache file path
        let cache_path = Self::get_cache_path(hdt_path);

        // Reader path: allow concurrent readers if cache is already present and valid.
        FileExt::lock_shared(&lock_file).map_err(|e| {
            boxed_io_error(format!(
                "failed to acquire shared cache lock {} for {}: {e}",
                lock_file_path.display(),
                hdt_path.display()
            ))
        })?;

        // Check if cache exists and is readable.
        if cache_path.exists() {
            debug!("Found existing cache: {}", cache_path.display());
            match Self::read_from_file(&cache_path) {
                Ok((cache, op_index_bitmap_offset)) => {
                    debug!("Loaded cache successfully");
                    debug!("{cache:#?}");
                    unlock_cache_lock(&lock_file, &lock_file_path, hdt_path, "shared")?;
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

        unlock_cache_lock(&lock_file, &lock_file_path, hdt_path, "shared")?;

        // Writer path: serialize cache regeneration.
        FileExt::lock_exclusive(&lock_file).map_err(|e| {
            boxed_io_error(format!(
                "failed to acquire exclusive cache lock {} for {}: {e}",
                lock_file_path.display(),
                hdt_path.display()
            ))
        })?;

        // Re-check in case another process generated the cache while we were waiting.
        if cache_path.exists() {
            debug!("Re-checking cache after acquiring exclusive lock");
            if let Ok((cache, op_index_bitmap_offset)) = Self::read_from_file(&cache_path) {
                unlock_cache_lock(&lock_file, &lock_file_path, hdt_path, "exclusive")?;
                return Ok((cache, op_index_bitmap_offset));
            }
            warn!("Cache remained unreadable under exclusive lock; regenerating...");
        }

        let generated = Self::try_write_cache_from_hdt_file(hdt_path, &cache_path);
        unlock_cache_lock(&lock_file, &lock_file_path, hdt_path, "exclusive")?;
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

    pub fn write_cache_from_hdt_file(hdt_path: &Path) -> (Self, u64) {
        let cache_path = Self::get_cache_path(hdt_path);
        Self::try_write_cache_from_hdt_file(hdt_path, &cache_path)
            .expect("Failed to create hybrid cache from HDT file")
    }

    fn try_write_cache_from_hdt_file(
        hdt_path: &Path, cache_path: &Path,
    ) -> Result<(Self, u64), Box<dyn std::error::Error>> {
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

        // Track offsets for each dictionary section BEFORE reading them
        let dict_shared_offset = reader.stream_position()?;
        let _ = crate::dict_sect_pfc::DictSectPFC::read(&mut reader, true)?
            .join()
            .map_err(|_| boxed_io_error("dictionary section read thread panicked"))??;

        let dict_subjects_offset = reader.stream_position()?;
        let _ = crate::dict_sect_pfc::DictSectPFC::read(&mut reader, true)?
            .join()
            .map_err(|_| boxed_io_error("dictionary section read thread panicked"))??;

        let dict_predicates_offset = reader.stream_position()?;
        let _ = crate::dict_sect_pfc::DictSectPFC::read(&mut reader, true)?
            .join()
            .map_err(|_| boxed_io_error("dictionary section read thread panicked"))??;

        let dict_objects_offset = reader.stream_position()?;
        let _ = crate::dict_sect_pfc::DictSectPFC::read(&mut reader, true)?
            .join()
            .map_err(|_| boxed_io_error("dictionary section read thread panicked"))??;

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

        let order: Order;
        if let Some(n) = triples_ci.get("order").and_then(|v| v.parse::<u32>().ok()) {
            order = Order::try_from(n)?;
        } else {
            return Err(boxed_io_error("unknown triples order in HDT triples control info"));
        }
        let adjlist_z = AdjListGeneric::new(InMemorySequence::new(sequence_z), InMemoryBitmap::new(bitmap_z));

        let triples_bitmap = TriplesBitmapGeneric::new(order, sequence_y, bitmap_y, adjlist_z);

        // Prepare temporary cache file for writing, then atomically replace target.
        let file_name = cache_path.file_name().and_then(|n| n.to_str()).unwrap_or("hdt-cache");
        let tmp_cache_path = cache_path.with_file_name(format!(
            "{file_name}.tmp-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("unnamed")
        ));
        let file = OpenOptions::new().write(true).create(true).truncate(true).open(&tmp_cache_path)?;
        let mut writer = BufWriter::new(file);

        // Create ControlInfo with metadata in properties

        let mut properties = HashMap::new();
        properties.insert("order".to_owned(), (triples_bitmap.order.clone() as u8).to_string());
        properties.insert("numTriples".to_owned(), triples_bitmap.adjlist_z.len().to_string());
        properties.insert("headerSize".to_owned(), header_size.to_string());
        let control_info =
            ControlInfo { control_type: ControlType::Index, format: CACHE_FORMAT.to_owned(), properties };

        // Write ControlInfo
        control_info.write(&mut writer)?;

        // Write wavelet_y
        triples_bitmap.wavelet_y.serialize_into(&mut writer)?;

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

        let op_index_offset = writer.stream_position()?;

        // Write op_index.bitmap, then op_index.sequence at the END of the file
        // (offset returned by read_from_file(), both can be accessed on-demand)
        triples_bitmap.op_index.bitmap.inner().write(&mut writer)?;

        triples_bitmap.op_index.sequence.inner().serialize_into(&mut writer)?;

        writer.flush()?;
        let file = writer.into_inner()?;
        file.sync_all()?;

        // Replace cache atomically where possible.
        if cache_path.exists() {
            let _ = std::fs::remove_file(cache_path);
        }
        std::fs::rename(&tmp_cache_path, cache_path).map_err(|e| {
            let _ = std::fs::remove_file(&tmp_cache_path);
            boxed_io_error(format!(
                "failed to move cache temp file {} to {}: {e}",
                tmp_cache_path.display(),
                cache_path.display()
            ))
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
    pub fn read_from_file<P: AsRef<Path>>(path: P) -> Result<(Self, u64), Box<dyn std::error::Error>> {
        use crate::containers::ControlType;
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // Read and verify ControlInfo
        let control_info = ControlInfo::read(&mut reader)?;

        // Verify it's an Index type
        if control_info.control_type != ControlType::Index {
            return Err(format!(
                "Invalid cache file: expected Index control type, found {:?}",
                control_info.control_type
            )
            .into());
        }

        // Verify format
        if control_info.format != CACHE_FORMAT {
            return Err(format!(
                "Unsupported cache format: expected {}, found {}",
                CACHE_FORMAT, control_info.format
            )
            .into());
        }

        // Read computed structures (in-memory)
        // Read wavelet_y
        let wavelet_y = WaveletMatrix::deserialize_from(&mut reader)?;

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

        // The OpIndex data (bitmap then sequence) starts right after all the offsets
        let op_index_offset = reader.stream_position()?;

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
        };

        Ok((cache, op_index_offset))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::{Arc, Barrier};
    use std::thread;
    use std::time::{SystemTime, UNIX_EPOCH};

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
        let (test_dir, hdt_path, cache_path) = setup_isolated_hdt("single")?;

        // Clean up any existing cache
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

        // Clean up
        std::fs::remove_dir_all(test_dir)?;

        Ok(())
    }

    #[test]
    fn test_from_hdt_path_parallel_threads() -> Result<(), Box<dyn std::error::Error>> {
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
                Err(_) => return Err(std::io::Error::other("hybrid cache worker thread panicked").into()),
            }
        }

        assert!(cache_path.exists(), "cache should exist after concurrent loads");
        std::fs::remove_dir_all(test_dir)?;
        Ok(())
    }
}
