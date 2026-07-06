//! Trait abstraction for sequence access - allows in-memory and memory-mapped implementations

use bytesize::ByteSize;
use std::fmt::{self, Debug};
use std::sync::Arc;

pub const USIZE_BITS: usize = usize::BITS as usize;

/// Trait for accessing integer sequences
///
/// This abstraction allows sequences to be either:
/// - In-memory (Sequence struct)
/// - Memory-mapped from disk
pub trait SequenceAccess: Debug + Send + Sync {
    /// Get the value at the given index
    fn get(&self, index: usize) -> usize;

    /// Number of entries in the sequence
    fn len(&self) -> usize;

    /// Whether the sequence is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of bits per entry
    fn bits_per_entry(&self) -> usize;

    /// Size in bytes (memory footprint)
    fn size_in_bytes(&self) -> usize;
}

/// In-memory sequence implementation (existing Sequence)
#[derive(Debug)]
pub struct InMemorySequence {
    inner: crate::containers::Sequence,
}

impl InMemorySequence {
    pub const fn new(sequence: crate::containers::Sequence) -> Self {
        Self { inner: sequence }
    }

    pub fn into_inner(self) -> crate::containers::Sequence {
        self.inner
    }

    pub const fn inner(&self) -> &crate::containers::Sequence {
        &self.inner
    }
}

impl SequenceAccess for InMemorySequence {
    fn get(&self, index: usize) -> usize {
        self.inner.get(index)
    }

    fn len(&self) -> usize {
        self.inner.entries
    }

    fn bits_per_entry(&self) -> usize {
        self.inner.bits_per_entry
    }

    fn size_in_bytes(&self) -> usize {
        self.inner.size_in_bytes()
    }
}

/// Memory-mapped sequence implementation.
///
/// Holds an `Arc<Mmap>` so multiple components can share a single mapping
/// of the same file. Bit-packed values are extracted with branch-light
/// slice indexing — no locks, no buffered seeks, fully thread-safe.
///
/// # Safety contract for callers
///
/// As with [`crate::containers::MmapBitmap`], the underlying file must not
/// be modified or truncated while any `MmapSequence` references it.
pub struct MmapSequence {
    /// Shared memory map of the underlying file.
    mmap: Arc<memmap2::Mmap>,
    /// Offset to the first byte of packed sequence data within the mmap.
    data_offset: usize,
    /// Number of entries.
    entries: usize,
    /// Number of bits per entry. Validated to be in `1..=64` (or 0 for empty).
    bits_per_entry: usize,
    /// Length of the metadata header (type + bits + vbyte(entries) + CRC8).
    metadata_size: usize,
}

impl fmt::Debug for MmapSequence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}, {} entries, {} bits per entry, mmaped from file",
            ByteSize(self.size_in_bytes() as u64),
            self.entries,
            self.bits_per_entry,
        )
    }
}

/// Errors that can occur while parsing a sequence header out of a memory map.
#[derive(thiserror::Error, Debug)]
pub enum MmapSequenceError {
    #[error("sequence offset {offset} is past end of mmap (len {len})")]
    OffsetOutOfBounds { offset: u64, len: usize },
    /// The serialized offset is larger than this platform's `usize`. On 32-bit
    /// targets this fires for any offset above `u32::MAX` (~4 GiB). On 64-bit
    /// it cannot fire — `u64::MAX as u64 == usize::MAX as u64`.
    #[error(
        "sequence offset {offset} exceeds platform usize::MAX ({usize_max}); this cache file requires a 64-bit target"
    )]
    OffsetTooLargeForPlatform { offset: u64, usize_max: u64 },
    #[error("unsupported sequence type {0}, expected 1 (Log64)")]
    UnsupportedType(u8),
    #[error("entry size of {0} bit too large (>64 bit)")]
    EntrySizeTooLarge(usize),
    #[error("truncated sequence header at offset {0}")]
    Truncated(u64),
    #[error("invalid CRC8 over sequence header: computed {computed}, stored {stored}")]
    InvalidCrc8 { computed: u8, stored: u8 },
    #[error("sequence data extends past end of mmap: needs {needed} bytes from offset {offset}, mmap len {len}")]
    DataPastEnd { offset: usize, needed: usize, len: usize },
}

impl From<MmapSequenceError> for std::io::Error {
    fn from(e: MmapSequenceError) -> Self {
        std::io::Error::new(std::io::ErrorKind::InvalidData, e)
    }
}

impl MmapSequence {
    /// Convenience: open the file at `file_path`, mmap it, and parse the
    /// sequence header at `sequence_offset`.
    pub fn new(file_path: impl AsRef<std::path::Path>, sequence_offset: u64) -> std::io::Result<Self> {
        let file = std::fs::File::open(file_path.as_ref())?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        Self::from_mmap(Arc::new(mmap), sequence_offset)
    }

    /// Parse a sequence header that begins at `sequence_offset` within the
    /// given shared mmap. Validates CRC8 over the header bytes, that
    /// `bits_per_entry <= 64`, and that the entire data range fits inside
    /// the mmap.
    pub fn from_mmap(mmap: Arc<memmap2::Mmap>, sequence_offset: u64) -> std::io::Result<Self> {
        use crate::containers::vbyte::read_vbyte;
        use std::io::Cursor;

        // Reject offsets that cannot be addressed on this platform before any
        // truncating cast. On 64-bit `usize::MAX as u64 == u64::MAX` so this
        // never fires; on 32-bit it surfaces a clean error instead of silently
        // wrapping when offsets exceed 4 GiB.
        if sequence_offset > usize::MAX as u64 {
            return Err(MmapSequenceError::OffsetTooLargeForPlatform {
                offset: sequence_offset,
                usize_max: usize::MAX as u64,
            }
            .into());
        }

        let mmap_len = mmap.len();
        if (sequence_offset as usize) > mmap_len {
            return Err(MmapSequenceError::OffsetOutOfBounds { offset: sequence_offset, len: mmap_len }.into());
        }

        let mut cursor = Cursor::new(&mmap[sequence_offset as usize..]);
        let mut history: Vec<u8> = Vec::with_capacity(8);

        let type_byte = read_one_byte(&mut cursor, sequence_offset)?;
        history.push(type_byte);
        if type_byte != 1 {
            return Err(MmapSequenceError::UnsupportedType(type_byte).into());
        }

        let bits_per_entry_byte = read_one_byte(&mut cursor, sequence_offset)?;
        history.push(bits_per_entry_byte);
        let bits_per_entry = bits_per_entry_byte as usize;
        if bits_per_entry > 64 {
            return Err(MmapSequenceError::EntrySizeTooLarge(bits_per_entry).into());
        }

        let (entries, vbyte_bytes) =
            read_vbyte(&mut cursor).map_err(|_| MmapSequenceError::Truncated(sequence_offset))?;
        history.extend_from_slice(&vbyte_bytes);

        let stored_crc = read_one_byte(&mut cursor, sequence_offset)?;

        let crc8 = crc::Crc::<u8>::new(&crc::CRC_8_SMBUS);
        let mut digest = crc8.digest();
        digest.update(&history);
        let computed_crc = digest.finalize();
        if computed_crc != stored_crc {
            return Err(MmapSequenceError::InvalidCrc8 { computed: computed_crc, stored: stored_crc }.into());
        }

        let metadata_size = cursor.position() as usize;
        let data_offset = sequence_offset as usize + metadata_size;
        let data_size = sequence_data_size_bytes(entries, bits_per_entry);
        if data_offset.saturating_add(data_size) > mmap_len {
            return Err(
                MmapSequenceError::DataPastEnd { offset: data_offset, needed: data_size, len: mmap_len }.into()
            );
        }

        Ok(Self { mmap, data_offset, entries, bits_per_entry, metadata_size })
    }

    /// Read the bit-packed value at `index`. The constructor verified that
    /// every in-bounds index is reachable, so this never reads past the
    /// mmap end.
    ///
    /// The hot path (bits_per_entry + bit_in_byte ≤ 64, i.e. the value fits
    /// in a single u64 load) avoids per-byte iteration entirely: one slice
    /// copy → one `u64::from_le_bytes` → shift + mask. This is the common
    /// case for HDT sequences where bits_per_entry ≤ 56.
    ///
    /// The cold path (57-64 bits_per_entry with a non-zero bit_in_byte,
    /// requiring up to 9 bytes) falls back to a 16-byte stack copy.
    fn read_value(&self, index: usize) -> usize {
        debug_assert!(index < self.entries);
        debug_assert!(self.bits_per_entry <= 64);

        if self.bits_per_entry == 0 {
            return 0;
        }

        let bit_offset = index * self.bits_per_entry;
        let byte_offset = self.data_offset + (bit_offset / 8);
        let bit_in_byte = bit_offset % 8;
        let bits_needed = self.bits_per_entry + bit_in_byte;
        let bytes_needed = bits_needed.div_ceil(8); // 1..=9

        let slice = &self.mmap[byte_offset..byte_offset + bytes_needed];

        let raw: u128 = if bytes_needed <= 8 {
            // Hot path: single u64 load covers the value.
            let mut buf = [0u8; 8];
            buf[..bytes_needed].copy_from_slice(slice);
            u64::from_le_bytes(buf) as u128
        } else {
            // Cold path: 9 bytes needed (bits_per_entry 57-64 with misaligned start).
            let mut buf = [0u8; 16];
            buf[..bytes_needed].copy_from_slice(slice);
            u128::from_le_bytes(buf)
        };

        let mask = (1u128 << self.bits_per_entry) - 1;
        ((raw >> bit_in_byte) & mask) as usize
    }

    /// Calculate the total serialized size of this sequence on disk: header
    /// (type + bits + vbyte(entries) + CRC8) + byte-aligned packed data +
    /// CRC32 trailer. Used by callers that need to find the offset of the
    /// next section in a cache file.
    pub const fn serialized_size_bytes(&self) -> usize {
        let data_size = sequence_data_size_bytes(self.entries, self.bits_per_entry);
        // metadata + data + CRC32 trailer
        self.metadata_size + data_size + 4
    }

    /// Verify the CRC32 trailer over the packed sequence data. This forces
    /// every page of the sequence into memory; do not call from a hot path.
    pub fn verify_crc32(&self) -> Result<(), std::io::Error> {
        let data_size = sequence_data_size_bytes(self.entries, self.bits_per_entry);
        let data = &self.mmap[self.data_offset..self.data_offset + data_size];
        let trailer_offset = self.data_offset + data_size;
        if trailer_offset + 4 > self.mmap.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "missing CRC32 trailer for sequence",
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
                format!("invalid CRC32 over sequence data: computed {computed}, stored {stored}"),
            ));
        }
        Ok(())
    }
}

impl SequenceAccess for MmapSequence {
    fn get(&self, index: usize) -> usize {
        assert!(index < self.entries, "index {} out of bounds for sequence of len {}", index, self.entries);
        self.read_value(index)
    }

    fn len(&self) -> usize {
        self.entries
    }

    fn bits_per_entry(&self) -> usize {
        self.bits_per_entry
    }

    fn size_in_bytes(&self) -> usize {
        // Heap footprint only: a few words plus the Arc control block.
        // The mmap data lives in the OS page cache and is not counted.
        std::mem::size_of::<Self>()
    }
}

/// Number of bytes that `entries * bits_per_entry` bits occupy on disk
/// (byte-aligned).
const fn sequence_data_size_bytes(entries: usize, bits_per_entry: usize) -> usize {
    if entries == 0 || bits_per_entry == 0 {
        return 0;
    }
    (entries * bits_per_entry).div_ceil(8)
}

fn read_one_byte<T: AsRef<[u8]>>(
    cursor: &mut std::io::Cursor<T>, base_offset: u64,
) -> Result<u8, MmapSequenceError> {
    use std::io::Read;
    let mut buf = [0u8];
    cursor.read_exact(&mut buf).map_err(|_| MmapSequenceError::Truncated(base_offset))?;
    Ok(buf[0])
}
