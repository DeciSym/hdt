//! Trait abstraction for bitmap access - allows in-memory and memory-mapped implementations

use std::fmt::{self, Debug};
use std::sync::Arc;

/// Number of 64-bit words per rank-index block. 8 * 64 = 512 bits per block.
/// Each block contributes one `u64` cumulative count to the rank index, so
/// the index overhead is 1 u64 per 512 bits of bitmap data ≈ 1.56%.
pub const RANK_BLOCK_WORDS: usize = 8;
pub const RANK_BLOCK_BITS: usize = RANK_BLOCK_WORDS * 64;

/// Number of cumulative-count entries needed to index a bitmap with
/// `num_words` 64-bit words. Always `num_words.div_ceil(RANK_BLOCK_WORDS) + 1`
/// so the last entry holds the total `num_ones`.
pub const fn rank_index_entries(num_words: usize) -> usize {
    num_words.div_ceil(RANK_BLOCK_WORDS) + 1
}

/// Serialized byte length of a rank index for a bitmap with `num_words`
/// 64-bit words. Each entry is a little-endian `u64`.
pub const fn rank_index_size_bytes(num_words: usize) -> usize {
    rank_index_entries(num_words) * 8
}

/// Build a rank index over the given bitmap words.
///
/// Returns a `Vec<u64>` of length `num_blocks + 1`, where
/// `result[i] = popcount(words[0..i*RANK_BLOCK_WORDS])`. The last entry is
/// therefore `num_ones`. Callers write this vector to the cache file as a
/// little-endian `u64` array; [`RankIndex`] then views it via mmap.
pub fn build_rank_index(words: &[u64]) -> Vec<u64> {
    let num_blocks = words.len().div_ceil(RANK_BLOCK_WORDS);
    let mut cumulative: Vec<u64> = Vec::with_capacity(num_blocks + 1);
    cumulative.push(0);
    let mut count: u64 = 0;
    let mut w = 0usize;
    while w < words.len() {
        let end = (w + RANK_BLOCK_WORDS).min(words.len());
        for word in &words[w..end] {
            count += word.count_ones() as u64;
        }
        cumulative.push(count);
        w = end;
    }
    cumulative
}

/// View over a precomputed rank index stored in a memory map.
///
/// The index is a dense `[u64]` array, `num_blocks + 1` entries long, laid
/// out contiguously at `offset` bytes inside `mmap`. Entry `i` holds the
/// cumulative count of set bits at bit position `i * RANK_BLOCK_BITS`, and
/// the final entry holds the total `num_ones`.
///
/// The mmap can be a different file than the one holding the bitmap data
/// — for `HdtHybrid`, the bitmap data lives in the HDT file while the rank
/// index lives in the cache file. Each `Arc<Mmap>` keeps its own file alive
/// for the lifetime of the accessor.
#[derive(Clone)]
pub struct RankIndex {
    mmap: Arc<memmap2::Mmap>,
    offset: usize,
    num_blocks: usize,
}

impl fmt::Debug for RankIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RankIndex {{ num_blocks: {}, bytes: {} }}",
            self.num_blocks,
            rank_index_size_bytes(self.num_blocks * RANK_BLOCK_WORDS)
        )
    }
}

impl RankIndex {
    /// Construct a rank index view. `num_bits` is the bitmap this index
    /// describes — used to derive `num_blocks` and to bounds-check the mmap.
    pub fn from_mmap(mmap: Arc<memmap2::Mmap>, offset: u64, num_bits: usize) -> std::io::Result<Self> {
        if offset > usize::MAX as u64 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("rank index offset {offset} exceeds platform usize::MAX"),
            ));
        }
        let offset = offset as usize;
        let num_words = num_bits.div_ceil(64);
        let num_blocks = num_words.div_ceil(RANK_BLOCK_WORDS);
        let byte_len = rank_index_size_bytes(num_words);
        if offset.saturating_add(byte_len) > mmap.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "rank index data extends past end of mmap: needs {byte_len} bytes from offset {offset}, mmap len {}",
                    mmap.len()
                ),
            ));
        }
        Ok(Self { mmap, offset, num_blocks })
    }

    /// Number of 512-bit blocks this index covers.
    pub const fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    /// Read the cumulative count stored at block boundary `block_idx`.
    /// Valid range: `[0, num_blocks]` inclusive (the final entry is
    /// `num_ones`). Panics (debug) on out-of-range access.
    pub fn cumulative(&self, block_idx: usize) -> u64 {
        debug_assert!(block_idx <= self.num_blocks);
        let byte_offset = self.offset + block_idx * 8;
        let bytes: [u8; 8] = self.mmap[byte_offset..byte_offset + 8].try_into().unwrap();
        u64::from_le_bytes(bytes)
    }
}

/// Trait for accessing bitmaps with rank and select support
///
/// This abstraction allows bitmaps to be either:
/// - In-memory with RSNarrow indexes (fast O(1) operations)
/// - Memory-mapped using OS page cache
pub trait BitmapAccess: Debug + Send + Sync {
    /// Returns the position of the k-th one bit (0-indexed), or None if there aren't that many
    fn select1(&self, k: usize) -> Option<usize>;

    /// Returns the number of one bits from the 0-th bit to the pos-th bit (exclusive)
    fn rank(&self, pos: usize) -> usize;

    /// Number of bits in the bitmap
    fn len(&self) -> usize;

    /// Whether the bitmap is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of one bits in the bitmap
    fn num_ones(&self) -> usize;

    /// Access a specific bit (returns true if 1, false if 0).
    ///
    /// # Panics
    /// Implementations must panic if `pos >= self.len()`. This method is
    /// called from query iterators that compute their positions by
    /// construction; an out-of-bounds access is always a programming error
    /// in the caller and should fail loudly rather than masquerade as a
    /// false bit.
    fn access(&self, pos: usize) -> bool;

    /// Size in bytes (memory footprint)
    fn size_in_bytes(&self) -> usize;
}

/// In-memory bitmap implementation with RSNarrow indexing (existing Bitmap)
#[derive(Debug, Clone)]
pub struct InMemoryBitmap {
    inner: crate::containers::Bitmap,
}

impl InMemoryBitmap {
    pub const fn new(bitmap: crate::containers::Bitmap) -> Self {
        Self { inner: bitmap }
    }

    pub fn into_inner(self) -> crate::containers::Bitmap {
        self.inner
    }

    pub const fn inner(&self) -> &crate::containers::Bitmap {
        &self.inner
    }
}

impl BitmapAccess for InMemoryBitmap {
    fn select1(&self, k: usize) -> Option<usize> {
        self.inner.select1(k)
    }

    fn rank(&self, pos: usize) -> usize {
        self.inner.rank(pos)
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn num_ones(&self) -> usize {
        self.inner.num_ones()
    }

    fn access(&self, pos: usize) -> bool {
        self.inner.at_last_sibling(pos)
    }

    fn size_in_bytes(&self) -> usize {
        self.inner.size_in_bytes()
    }
}

/// Memory-mapped bitmap implementation using OS page cache.
///
/// Holds an `Arc<Mmap>` so it can share a single mapping with other
/// components that view the same file (dictionary sections, sequences, ...).
///
/// # Safety contract for callers
///
/// The mapped file must not be modified or truncated for the lifetime of this
/// `MmapBitmap` (and any other accessors built from the same `Arc<Mmap>`).
/// `memmap2::Mmap::map` is `unsafe` for exactly this reason: writing to a
/// mapped file from another process can produce SIGBUS or undefined behavior.
/// HDT files are immutable after generation, so this is fine in practice as
/// long as nothing rewrites the file in place while a query process is live.
///
/// # Performance characteristics
/// - `access()`: O(1) - direct slice indexing
/// - `rank()`:   O(n) - scans words (no rank index)
/// - `select1()`: O(n) - scans words (no select index)
///
/// # Memory usage
/// - Heap: a few `usize` fields plus the `Arc<Mmap>` control block
/// - Bitmap data lives in the OS page cache, demand-paged
pub struct MmapBitmap {
    /// Shared memory map of the underlying file.
    mmap: Arc<memmap2::Mmap>,
    /// Offset to the first byte of bitmap data within the mmap.
    data_offset: usize,
    /// Total number of bits stored in this bitmap.
    num_bits: usize,
    /// Number of 64-bit words covered by the bitmap. The last word may be
    /// byte-aligned (i.e. fewer than 8 bytes if `num_bits % 64 != 0`).
    num_words: usize,
    /// Length of the metadata header (type + vbyte + CRC8) in bytes.
    /// Used by `serialized_size_bytes` so callers can find the next section.
    metadata_size: usize,
    /// Pre-computed number of one bits. Computed at construction time so
    /// `num_ones()` is O(1); the cost is a single sequential scan that the
    /// OS will read ahead efficiently.
    num_ones_cached: usize,
    /// Optional precomputed rank index. When present, `rank()` is O(1) and
    /// `select1()` is O(log num_blocks + 64). When absent, both fall back
    /// to linear scans. Built once at cache-generation time and loaded via
    /// a second mmap (typically the cache file, distinct from the HDT mmap
    /// that holds `data_offset`).
    rank_index: Option<RankIndex>,
}

impl fmt::Debug for MmapBitmap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} bits, {} ones, mmaped from file",
            self.num_bits, self.num_ones_cached
        )
    }
}

/// Errors that can occur while parsing a bitmap header out of a memory map.
#[derive(thiserror::Error, Debug)]
pub enum MmapBitmapError {
    #[error("bitmap offset {offset} is past end of mmap (len {len})")]
    OffsetOutOfBounds { offset: u64, len: usize },
    /// The serialized offset is larger than this platform's `usize`. On 32-bit
    /// targets this fires for any offset above `u32::MAX` (~4 GiB). On 64-bit
    /// it cannot fire — `u64::MAX as u64 == usize::MAX as u64`.
    #[error("bitmap offset {offset} exceeds platform usize::MAX ({usize_max}); this cache file requires a 64-bit target")]
    OffsetTooLargeForPlatform { offset: u64, usize_max: u64 },
    #[error("unsupported bitmap type {0}, expected 1")]
    UnsupportedType(u8),
    #[error("truncated bitmap header at offset {0}")]
    Truncated(u64),
    #[error("invalid CRC8 over bitmap header: computed {computed}, stored {stored}")]
    InvalidCrc8 { computed: u8, stored: u8 },
    #[error("bitmap data extends past end of mmap: needs {needed} bytes from offset {offset}, mmap len {len}")]
    DataPastEnd { offset: usize, needed: usize, len: usize },
    #[error("vbyte decode error: {0}")]
    Vbyte(#[from] std::io::Error),
}

impl From<MmapBitmapError> for std::io::Error {
    fn from(e: MmapBitmapError) -> Self {
        std::io::Error::new(std::io::ErrorKind::InvalidData, e)
    }
}

impl MmapBitmap {
    /// Convenience: open the file at `file_path` and mmap it, then parse the
    /// bitmap header at `bitmap_offset`. Most callers should mmap the file
    /// once themselves and call [`Self::from_mmap`] so multiple components
    /// can share one mapping.
    pub fn new(file_path: impl AsRef<std::path::Path>, bitmap_offset: u64) -> std::io::Result<Self> {
        let file = std::fs::File::open(file_path.as_ref())?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        Self::from_mmap(Arc::new(mmap), bitmap_offset)
    }

    /// Parse a bitmap header that begins at `bitmap_offset` within the given
    /// shared mmap, validating CRC8 and bounds. The returned `MmapBitmap`
    /// holds a clone of the `Arc`, so the mapping lives as long as any
    /// accessor that references it.
    pub fn from_mmap(mmap: Arc<memmap2::Mmap>, bitmap_offset: u64) -> std::io::Result<Self> {
        use crate::containers::vbyte::read_vbyte;
        use std::io::Cursor;

        // Reject offsets that cannot be addressed on this platform before any
        // truncating cast. On 64-bit `usize::MAX as u64 == u64::MAX` so this
        // never fires; on 32-bit it surfaces a clean error instead of silently
        // wrapping when offsets exceed 4 GiB.
        if bitmap_offset > usize::MAX as u64 {
            return Err(MmapBitmapError::OffsetTooLargeForPlatform {
                offset: bitmap_offset,
                usize_max: usize::MAX as u64,
            }
            .into());
        }

        let mmap_len = mmap.len();
        if (bitmap_offset as usize) > mmap_len {
            return Err(MmapBitmapError::OffsetOutOfBounds { offset: bitmap_offset, len: mmap_len }.into());
        }

        // Parse the header: [type:u8] [num_bits:vbyte] [crc8:u8]. CRC8 covers
        // type byte and the vbyte bytes (matching Bitmap::read).
        let mut cursor = Cursor::new(&mmap[bitmap_offset as usize..]);
        let mut history: Vec<u8> = Vec::with_capacity(8);

        let type_byte = read_one_byte(&mut cursor, bitmap_offset)?;
        history.push(type_byte);
        if type_byte != 1 {
            return Err(MmapBitmapError::UnsupportedType(type_byte).into());
        }

        let (num_bits, vbyte_bytes) =
            read_vbyte(&mut cursor).map_err(|_| MmapBitmapError::Truncated(bitmap_offset))?;
        history.extend_from_slice(&vbyte_bytes);

        let stored_crc = read_one_byte(&mut cursor, bitmap_offset)?;

        let crc8 = crc::Crc::<u8>::new(&crc::CRC_8_SMBUS);
        let mut digest = crc8.digest();
        digest.update(&history);
        let computed_crc = digest.finalize();
        if computed_crc != stored_crc {
            return Err(MmapBitmapError::InvalidCrc8 { computed: computed_crc, stored: stored_crc }.into());
        }

        let metadata_size = cursor.position() as usize;
        let data_offset = bitmap_offset as usize + metadata_size;
        let num_words = if num_bits == 0 { 0 } else { ((num_bits - 1) / 64) + 1 };

        // Bounds check: every read we will ever perform must lie inside mmap.
        let data_size = bitmap_data_size_bytes(num_bits, num_words);
        if data_offset.saturating_add(data_size) > mmap_len {
            return Err(MmapBitmapError::DataPastEnd { offset: data_offset, needed: data_size, len: mmap_len }.into());
        }

        // Pre-compute num_ones. The constructor reads every byte once, which
        // is the same work the legacy Bitmap::read does. Subsequent calls to
        // num_ones() are then O(1).
        let num_ones_cached =
            count_ones_in_slice(&mmap[data_offset..data_offset + data_size], num_bits);

        Ok(Self {
            mmap,
            data_offset,
            num_bits,
            num_words,
            metadata_size,
            num_ones_cached,
            rank_index: None,
        })
    }

    /// Like [`Self::from_mmap`] but also attaches a precomputed rank index
    /// (typically stored in a separate cache file). The rank index turns
    /// `rank` into an O(1) operation and `select1` into
    /// O(log num_blocks + 64), which is what makes `HdtHybrid` viable for
    /// queries on large bitmaps.
    ///
    /// The `rank_mmap` can be the same `Arc<Mmap>` as `bitmap_mmap` (if
    /// the bitmap and its rank index share a file) or a different one.
    pub fn from_mmap_with_rank_index(
        bitmap_mmap: Arc<memmap2::Mmap>, bitmap_offset: u64, rank_mmap: Arc<memmap2::Mmap>, rank_offset: u64,
    ) -> std::io::Result<Self> {
        let mut bitmap = Self::from_mmap(bitmap_mmap, bitmap_offset)?;
        let rank_index = RankIndex::from_mmap(rank_mmap, rank_offset, bitmap.num_bits)?;
        bitmap.rank_index = Some(rank_index);
        Ok(bitmap)
    }

    /// Read a 64-bit word from the bitmap data. Always returns a value (the
    /// constructor verified that all reads up to `num_words` are in bounds);
    /// the last word may have fewer than 64 valid bits.
    fn read_word(&self, word_index: usize) -> u64 {
        debug_assert!(word_index < self.num_words);
        let byte_offset = self.data_offset + word_index * 8;

        if word_index + 1 == self.num_words {
            // Last word is byte-aligned, may be shorter than 8 bytes.
            let last_word_bits = ((self.num_bits - 1) % 64) + 1;
            let last_word_bytes = last_word_bits.div_ceil(8);
            let mut word = 0u64;
            for byte_idx in 0..last_word_bytes {
                word |= (self.mmap[byte_offset + byte_idx] as u64) << (byte_idx * 8);
            }
            word
        } else {
            let bytes: [u8; 8] = self.mmap[byte_offset..byte_offset + 8].try_into().unwrap();
            u64::from_le_bytes(bytes)
        }
    }

    /// Find the k-th one bit in a 64-bit word.
    fn select1_in_word(word: u64, k: usize) -> Option<usize> {
        let mut remaining = word;
        for _ in 0..k {
            // Clear the lowest set bit.
            if remaining == 0 {
                return None;
            }
            remaining &= remaining - 1;
        }
        if remaining == 0 {
            return None;
        }
        Some(remaining.trailing_zeros() as usize)
    }

    /// Calculate the total serialized size of this bitmap on disk, including
    /// the type byte, vbyte length, CRC8, the byte-aligned data and the
    /// trailing CRC32. Used by `new_hybrid_cache` to find the offset of the
    /// section that follows this bitmap in the cache file.
    pub const fn serialized_size_bytes(&self) -> usize {
        let data_size = bitmap_data_size_bytes(self.num_bits, self.num_words);
        // metadata + data + CRC32 trailer
        self.metadata_size + data_size + 4
    }

    /// O(1) rank using the precomputed rank index.
    ///
    /// `pos < self.num_bits` is the caller's precondition (checked in the
    /// trait method before delegation). The block-level cumulative count
    /// eliminates the linear scan; we only popcount up to 7 full words plus
    /// one partial word within the target block.
    ///
    /// The full-word popcounts are batched into a single contiguous slice
    /// read — one bounds check for up to 56 bytes instead of 7 separate
    /// `read_word()` calls. `chunks_exact(8)` lets the compiler
    /// SIMD-vectorize the popcount loop. The target word (which might be
    /// the byte-aligned last word of the bitmap) still uses `read_word()`
    /// for its partial-word edge-case handling.
    fn rank_with_index(&self, idx: &RankIndex, pos: usize) -> usize {
        let word_idx = pos / 64;
        let block_idx = word_idx / RANK_BLOCK_WORDS;
        let word_in_block = word_idx % RANK_BLOCK_WORDS;
        let bit_in_word = pos % 64;

        let mut count = idx.cumulative(block_idx) as usize;

        // Batch-load full words in one slice. All words before word_idx are
        // guaranteed to be full 8-byte words (only the absolute last word of
        // the bitmap can be byte-aligned, and it's handled via read_word below).
        if word_in_block > 0 {
            let block_start = self.data_offset + block_idx * RANK_BLOCK_WORDS * 8;
            let bytes = word_in_block * 8;
            let slice = &self.mmap[block_start..block_start + bytes];
            for chunk in slice.chunks_exact(8) {
                let word = u64::from_le_bytes(chunk.try_into().unwrap());
                count += word.count_ones() as usize;
            }
        }

        // Partial target word (might be the last word of the bitmap).
        if bit_in_word > 0 {
            let word = self.read_word(word_idx);
            let mask = (1u64 << bit_in_word) - 1;
            count += (word & mask).count_ones() as usize;
        }
        count
    }

    /// Fallback O(n) rank used when no rank index is attached.
    fn rank_scan(&self, pos: usize) -> usize {
        let word_index = pos / 64;
        let bit_in_word = pos % 64;
        let mut count = 0usize;

        for idx in 0..word_index {
            count += self.read_word(idx).count_ones() as usize;
        }

        let word = self.read_word(word_index);
        let mask = if bit_in_word == 0 { 0 } else { (1u64 << bit_in_word) - 1 };
        count += (word & mask).count_ones() as usize;

        count
    }

    /// O(log num_blocks + 64) select1 using the precomputed rank index.
    ///
    /// Binary-search the block table to find the block where the k-th set
    /// bit lives, then linear-scan the (≤ 8) words of that block.
    /// `k < num_ones_cached` is the caller's precondition.
    fn select1_with_index(&self, idx: &RankIndex, k: usize) -> Option<usize> {
        let target = k as u64;
        // Invariant: cumulative(lo) <= target, cumulative(hi) > target.
        // We want the largest block_idx with cumulative(block_idx) <= target.
        let mut lo = 0usize;
        let mut hi = idx.num_blocks();
        while lo < hi {
            let mid = lo + (hi - lo).div_ceil(2);
            if mid == 0 {
                break;
            }
            if idx.cumulative(mid) <= target {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        let block_idx = lo;
        let block_start_count = idx.cumulative(block_idx) as usize;
        let target_in_block = k - block_start_count;

        // Linear scan within the block — at most RANK_BLOCK_WORDS words.
        let start_word = block_idx * RANK_BLOCK_WORDS;
        let end_word = (start_word + RANK_BLOCK_WORDS).min(self.num_words);
        let mut ones_seen = 0usize;
        for word_idx in start_word..end_word {
            let word = self.read_word(word_idx);
            let ones_in_word = word.count_ones() as usize;
            if ones_seen + ones_in_word > target_in_block {
                let target_in_word = target_in_block - ones_seen;
                let bit_pos = Self::select1_in_word(word, target_in_word)?;
                return Some(word_idx * 64 + bit_pos);
            }
            ones_seen += ones_in_word;
        }
        None
    }

    /// Fallback O(n) select1 used when no rank index is attached.
    fn select1_scan(&self, k: usize) -> Option<usize> {
        let mut ones_seen = 0usize;

        for word_idx in 0..self.num_words {
            let word = self.read_word(word_idx);
            let ones_in_word = word.count_ones() as usize;

            if ones_seen + ones_in_word > k {
                let target_in_word = k - ones_seen;
                let bit_pos = Self::select1_in_word(word, target_in_word)?;
                return Some(word_idx * 64 + bit_pos);
            }

            ones_seen += ones_in_word;
        }

        None
    }

    /// Verify the CRC32 trailer over the bitmap data. This forces all data
    /// pages into memory, so it is *not* called from the constructor. Use it
    /// in long-running services that want to detect on-disk corruption
    /// proactively.
    pub fn verify_crc32(&self) -> Result<(), std::io::Error> {
        let data_size = bitmap_data_size_bytes(self.num_bits, self.num_words);
        let data = &self.mmap[self.data_offset..self.data_offset + data_size];
        let trailer_offset = self.data_offset + data_size;
        if trailer_offset + 4 > self.mmap.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "missing CRC32 trailer for bitmap",
            ));
        }
        let trailer: [u8; 4] = self.mmap[trailer_offset..trailer_offset + 4].try_into().unwrap();
        let stored = u32::from_le_bytes(trailer);

        let crc32 = crc::Crc::<u32>::new(&crc::CRC_32_ISCSI);
        let mut digest = crc32.digest();
        digest.update(data);
        let computed = digest.finalize();
        if computed != stored {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("invalid CRC32 over bitmap data: computed {computed}, stored {stored}"),
            ));
        }
        Ok(())
    }
}

impl BitmapAccess for MmapBitmap {
    fn select1(&self, k: usize) -> Option<usize> {
        if k >= self.num_ones_cached {
            return None;
        }
        if let Some(idx) = &self.rank_index {
            return self.select1_with_index(idx, k);
        }
        self.select1_scan(k)
    }

    fn rank(&self, pos: usize) -> usize {
        if pos >= self.num_bits {
            return self.num_ones_cached;
        }
        if let Some(idx) = &self.rank_index {
            return self.rank_with_index(idx, pos);
        }
        self.rank_scan(pos)
    }

    fn len(&self) -> usize {
        self.num_bits
    }

    fn num_ones(&self) -> usize {
        self.num_ones_cached
    }

    fn access(&self, pos: usize) -> bool {
        assert!(
            pos < self.num_bits,
            "MmapBitmap::access out of bounds: pos {} >= len {}",
            pos,
            self.num_bits
        );
        let byte_offset = self.data_offset + pos / 8;
        let bit_in_byte = pos % 8;
        (self.mmap[byte_offset] >> bit_in_byte) & 1 == 1
    }

    fn size_in_bytes(&self) -> usize {
        // Heap footprint only: a few words plus the Arc control block.
        // The mmap data lives in the OS page cache and is not counted.
        std::mem::size_of::<Self>()
    }
}

/// Number of bytes the bitmap data occupies on disk for `num_bits` /
/// `num_words`. The last word is byte-aligned.
const fn bitmap_data_size_bytes(num_bits: usize, num_words: usize) -> usize {
    if num_words == 0 {
        return 0;
    }
    let full_words = num_words - 1;
    let last_word_bits = ((num_bits - 1) % 64) + 1;
    let last_word_bytes = last_word_bits.div_ceil(8);
    full_words * 8 + last_word_bytes
}

/// Count one bits in a byte-aligned bitmap data slice for `num_bits` total.
fn count_ones_in_slice(data: &[u8], num_bits: usize) -> usize {
    if num_bits == 0 {
        return 0;
    }
    let num_words = ((num_bits - 1) / 64) + 1;
    let mut count = 0usize;

    // Full 8-byte words.
    for word_idx in 0..(num_words - 1) {
        let byte_offset = word_idx * 8;
        let bytes: [u8; 8] = data[byte_offset..byte_offset + 8].try_into().unwrap();
        count += u64::from_le_bytes(bytes).count_ones() as usize;
    }

    // Last (possibly short) word.
    let last_word_bits = ((num_bits - 1) % 64) + 1;
    let last_word_bytes = last_word_bits.div_ceil(8);
    let byte_offset = (num_words - 1) * 8;
    let mut last_word = 0u64;
    for i in 0..last_word_bytes {
        last_word |= (data[byte_offset + i] as u64) << (i * 8);
    }
    let mask = if last_word_bits == 64 { u64::MAX } else { (1u64 << last_word_bits) - 1 };
    count += (last_word & mask).count_ones() as usize;
    count
}

fn read_one_byte<T: AsRef<[u8]>>(cursor: &mut std::io::Cursor<T>, base_offset: u64) -> Result<u8, MmapBitmapError> {
    use std::io::Read;
    let mut buf = [0u8];
    cursor.read_exact(&mut buf).map_err(|_| MmapBitmapError::Truncated(base_offset))?;
    Ok(buf[0])
}

#[cfg(test)]
mod rank_index_tests {
    use super::*;
    use crate::containers::Bitmap;
    use qwt::BitVectorMut;

    /// Build a bitmap with every `stride`-th bit set, padded to a multiple
    /// of 64 bits with zeros. Returns the `Bitmap` plus the total number
    /// of padded bits so callers can iterate the full range.
    fn build_test_bitmap(num_bits: usize, stride: usize) -> (Bitmap, usize) {
        let mut bv = BitVectorMut::new();
        for i in 0..num_bits {
            bv.push(i % stride == 0);
        }
        // Pad to next multiple of 64 with zeros.
        let padded = num_bits.div_ceil(64) * 64;
        for _ in num_bits..padded {
            bv.push(false);
        }
        (Bitmap::from(bv), padded)
    }

    /// Serialize a bitmap + its rank index into a single anonymous mmap
    /// and return the mmap plus the offsets of each section. This mimics
    /// the runtime layout where the bitmap lives in one region and its
    /// rank index in another (possibly the same) mmap.
    fn build_bitmap_mmap(bitmap: &Bitmap) -> (Arc<memmap2::Mmap>, u64, u64) {
        // Serialize bitmap to bytes via the real writer so we exercise the
        // exact header/CRC format that MmapBitmap::from_mmap expects.
        let mut bitmap_bytes = Vec::new();
        bitmap.write(&mut bitmap_bytes).unwrap();

        // Build the rank index and serialize as little-endian u64 array.
        let rank_words = build_rank_index(bitmap.dict.bit_vector().words());
        let mut rank_bytes = Vec::with_capacity(rank_words.len() * 8);
        for w in &rank_words {
            rank_bytes.extend_from_slice(&w.to_le_bytes());
        }

        // Combine into one anonymous mmap, bitmap first then rank index.
        let bitmap_offset = 0u64;
        let rank_offset = bitmap_bytes.len() as u64;
        let total = bitmap_bytes.len() + rank_bytes.len();
        let mut anon = memmap2::MmapMut::map_anon(total).unwrap();
        anon[..bitmap_bytes.len()].copy_from_slice(&bitmap_bytes);
        anon[bitmap_bytes.len()..].copy_from_slice(&rank_bytes);
        let mmap = Arc::new(anon.make_read_only().unwrap());

        (mmap, bitmap_offset, rank_offset)
    }

    /// The constructor used by `Hdt::new_hybrid_cache`; this directly
    /// exercises `rank_with_index` and `select1_with_index`.
    fn bitmap_with_rank(bitmap: &Bitmap) -> MmapBitmap {
        let (mmap, bitmap_offset, rank_offset) = build_bitmap_mmap(bitmap);
        MmapBitmap::from_mmap_with_rank_index(
            Arc::clone(&mmap),
            bitmap_offset,
            mmap,
            rank_offset,
        )
        .unwrap()
    }

    #[test]
    fn rank_matches_in_memory_every_position() {
        // 5000 bits × stride 3 → ~1667 ones → spans multiple rank blocks
        // (5000 / 512 ≈ 10 blocks), exercises both cumulative lookups and
        // partial-block popcounts across word boundaries.
        let (in_mem, padded_bits) = build_test_bitmap(5000, 3);
        let mmap_bitmap = bitmap_with_rank(&in_mem);

        assert_eq!(mmap_bitmap.len(), in_mem.len());
        assert_eq!(mmap_bitmap.num_ones(), in_mem.num_ones());

        // Compare rank at every position except the final boundary
        // (in-memory Bitmap::rank panics on pos == len).
        for pos in 0..padded_bits {
            let expected = in_mem.rank(pos);
            let actual = mmap_bitmap.rank(pos);
            assert_eq!(actual, expected, "rank({pos}) mismatch");
        }
        // At pos == len, MmapBitmap returns num_ones_cached.
        assert_eq!(mmap_bitmap.rank(padded_bits), in_mem.num_ones());
    }

    #[test]
    fn select1_matches_in_memory_every_index() {
        let (in_mem, _padded_bits) = build_test_bitmap(5000, 3);
        let mmap_bitmap = bitmap_with_rank(&in_mem);

        let num_ones = in_mem.num_ones();
        for k in 0..num_ones {
            let expected = in_mem.select1(k);
            let actual = mmap_bitmap.select1(k);
            assert_eq!(actual, expected, "select1({k}) mismatch");
        }
        // Past the end: both should return None.
        assert_eq!(mmap_bitmap.select1(num_ones), None);
        assert_eq!(mmap_bitmap.select1(num_ones + 1), None);
    }

    #[test]
    fn rank_and_select1_across_multiple_block_boundaries() {
        // Larger bitmap with a denser pattern to exercise binary search
        // through several rank blocks (12000 / 512 ≈ 23 blocks).
        let (in_mem, padded_bits) = build_test_bitmap(12000, 2);
        let mmap_bitmap = bitmap_with_rank(&in_mem);

        // Spot-check rank at every block boundary, and a handful of
        // positions within each block.
        for block in 0..(padded_bits / RANK_BLOCK_BITS) {
            let boundary = block * RANK_BLOCK_BITS;
            assert_eq!(mmap_bitmap.rank(boundary), in_mem.rank(boundary));
            assert_eq!(mmap_bitmap.rank(boundary + 1), in_mem.rank(boundary + 1));
            assert_eq!(mmap_bitmap.rank(boundary + 63), in_mem.rank(boundary + 63));
            assert_eq!(mmap_bitmap.rank(boundary + 64), in_mem.rank(boundary + 64));
            assert_eq!(mmap_bitmap.rank(boundary + 257), in_mem.rank(boundary + 257));
        }

        // Spot-check select1 at regular intervals.
        let num_ones = in_mem.num_ones();
        for k in (0..num_ones).step_by(37) {
            assert_eq!(mmap_bitmap.select1(k), in_mem.select1(k), "select1({k})");
        }
    }
}
