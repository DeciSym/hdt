use crate::ControlInfo;
use crate::containers::{
    AdjList, AdjListGeneric, Bitmap, BitmapAccess, InMemoryBitmap, InMemorySequence, MmapBitmap, MmapSequence,
    Sequence, SequenceAccess, bitmap, control_info, sequence,
};
use bytesize::ByteSize;
use log::error;
use qwt::QWT512;
use qwt::mem_dbg::{MemSize, SizeFlags};
use qwt::{AccessUnsigned, BitVector, BitVectorMut, bitvector::rs_narrow::RSNarrow};
use std::cmp::Ordering;
use std::fmt;
use std::io::BufRead;

mod subject_iter;
pub use subject_iter::SubjectIter;
mod predicate_iter;
pub use predicate_iter::PredicateIter;
mod predicate_object_iter;
pub use predicate_object_iter::PredicateObjectIter;
mod object_iter;
pub use object_iter::ObjectIter;

#[cfg(feature = "cache")]
pub mod hybrid_cache;
#[cfg(feature = "cache")]
pub use hybrid_cache::{CACHE_EXT, HybridCache, HybridCacheError, HybridCacheResult};

pub type Result<T> = core::result::Result<T, Error>;

/// Order of the triple sections.
/// Only SPO is tested, others probably don't work correctly.
#[allow(missing_docs)]
#[repr(u8)]
#[derive(Debug, Default, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "cache", derive(serde::Deserialize, serde::Serialize))]
pub enum Order {
    #[default]
    Unknown = 0,
    SPO = 1,
    SOP = 2,
    PSO = 3,
    POS = 4,
    OSP = 5,
    OPS = 6,
}

impl TryFrom<u32> for Order {
    type Error = Error;

    fn try_from(original: u32) -> Result<Self> {
        match original {
            0 => Ok(Order::Unknown),
            1 => Ok(Order::SPO),
            2 => Ok(Order::SOP),
            3 => Ok(Order::PSO),
            4 => Ok(Order::POS),
            5 => Ok(Order::OSP),
            6 => Ok(Order::OPS),
            n => Err(Error::UnrecognizedTriplesOrder(n)),
        }
    }
}

/// Generic inverse index from object id to positions in the object adjacency list.
/// This object-based index allows to traverse from the leaves and support ??O and ?PO queries.
/// Used for logarithmic (?) time access instead of linear time sequential search.
/// See Martínez-Prieto, M., M. Arias, and J. Fernández (2012). Exchange and Consumption of Huge RDF Data. Pages 8--10.
///
/// Generic over:
/// - S: SequenceAccess for the sequence (`InMemorySequence` or `MmapSequence`)
/// - B: BitmapAccess for the bitmap (`InMemoryBitmap` or `MmapBitmap`)
pub struct OpIndexGeneric<S: SequenceAccess, B: BitmapAccess> {
    /// Compact integer vector of object positions.
    /// "[...] integer sequence: SoP, which stores, for each object, a sorted list of references to the predicate-subject pairs (sorted by predicate) related to it."
    pub sequence: S,
    /// Bitmap with a one bit for every new object to allow finding the starting point for a given object id.
    pub bitmap: B,
}

/// Type alias for traditional in-memory OpIndex.
/// This maintains backward compatibility with existing code.
pub type OpIndex = OpIndexGeneric<InMemorySequence, InMemoryBitmap>;

/// Type alias for the mmap-based OpIndex used by `HdtHybrid`.
pub type OpIndexMmap = OpIndexGeneric<MmapSequence, MmapBitmap>;

impl<S: SequenceAccess, B: BitmapAccess> fmt::Debug for OpIndexGeneric<S, B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "total size {} {{", ByteSize(self.size_in_bytes() as u64))?;
        writeln!(
            f,
            "    sequence: {} with {} bits,",
            ByteSize(self.sequence.size_in_bytes() as u64),
            self.sequence.bits_per_entry()
        )?;
        write!(f, "    bitmap: {:#?}\n}}", self.bitmap)
    }
}

impl<S: SequenceAccess, B: BitmapAccess> OpIndexGeneric<S, B> {
    /// Create a new OpIndex with the given sequence and bitmap.
    ///
    /// # Panics
    /// Panics if the bitmap can't back the sequence. Same band as
    /// [`crate::containers::AdjListGeneric::new`]: bitmap must be at least
    /// `sequence.len()` bits long and at most 63 bits longer.
    pub fn new(sequence: S, bitmap: B) -> Self {
        let s_len = sequence.len();
        // See AdjListGeneric::new for the empty-bitmap special case.
        if s_len > 0 {
            let b_len = bitmap.len();
            assert!(
                b_len >= s_len && b_len - s_len < 64,
                "OpIndexGeneric: bitmap.len() ({b_len}) cannot back sequence.len() ({s_len})"
            );
        }
        Self { sequence, bitmap }
    }

    /// Size in bytes on the heap.
    pub fn size_in_bytes(&self) -> usize {
        self.sequence.size_in_bytes() + self.bitmap.size_in_bytes()
    }

    /// Find the first position in the OP index of the given object ID.
    pub fn find(&self, o: Id) -> usize {
        self.bitmap.select1(o - 1).unwrap() as usize
    }

    /// Find the last position in the object index of the given object ID.
    pub fn last(&self, o: Id) -> usize {
        self.bitmap.select1(o).map_or_else(|| self.bitmap.len() - 1, |index| index as usize - 1)
    }

    /// Get the value at the given index
    pub fn get(&self, index: usize) -> usize {
        self.sequence.get(index)
    }

    /// Number of entries in the sequence
    pub fn len(&self) -> usize {
        self.sequence.len()
    }

    /// Whether the sequence is empty
    pub fn is_empty(&self) -> bool {
        self.sequence.is_empty()
    }
}

type WT = QWT512<usize>;

/// Generic `BitmapTriples` variant of the triples section.
///
/// Generic over:
/// - S: SequenceAccess (`InMemorySequence` or `MmapSequence`)
/// - B: BitmapAccess (`InMemoryBitmap` or `MmapBitmap`)
///
/// ## Main Variants:
/// 1. **All in-memory**: `TriplesBitmap` (fast, high memory)
/// 2. **All mmaped**: `TriplesBitmapMmap` (low memory, OS page cache, used by `HdtHybrid`)
pub struct TriplesBitmapGeneric<S: SequenceAccess, B: BitmapAccess> {
    order: Order,
    /// bitmap to find positions in the wavelet matrix (generic: in-memory or file-based)
    pub bitmap_y: B,
    /// adjacency list storing the object IDs (generic: in-memory or file-based)
    pub adjlist_z: AdjListGeneric<S, B>,
    /// Index for object-based access. Generic over both sequence and bitmap access
    pub op_index: OpIndexGeneric<S, B>,
    /// wavelet matrix for predicate-based access
    pub wavelet_y: WT,
}

/// Type alias for the traditional TriplesBitmap with all in-memory structures.
/// This maintains backward compatibility with existing code.
pub type TriplesBitmap = TriplesBitmapGeneric<InMemorySequence, InMemoryBitmap>;

/// Type alias for the mmap-based TriplesBitmap used by `HdtHybrid`.
pub type TriplesBitmapMmap = TriplesBitmapGeneric<MmapSequence, MmapBitmap>;

#[derive(Debug)]
pub enum Level {
    Y,
    Z,
}

/// The error type for the triples bitmap read and write function.
#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("failed to read control info")]
    ControlInfo(#[from] control_info::Error),
    #[error("bitmap error in the {0:?} level")]
    Bitmap(Level, #[source] bitmap::Error),
    #[error("sequence read error")]
    Sequence(Level, #[source] sequence::Error),
    #[error("unspecified triples order")]
    UnspecifiedTriplesOrder,
    #[error("unknown triples order")]
    UnknownTriplesOrder,
    #[error("unrecognized triples order {0}")]
    UnrecognizedTriplesOrder(u32),
    #[error("unknown triples format {0}")]
    UnknownTriplesFormat(String),
    #[error("triple lists are not supported yet")]
    TriplesList,
    #[error("({0},{1},{2}) none of the components of a triple may be 0.")]
    TripleComponentZero(usize, usize, usize),
    #[error("unspecified external library error")]
    External(#[from] Box<dyn std::error::Error + Send + Sync + 'static>),
    #[error("cache decode error")]
    #[cfg(feature = "cache")]
    Decode(#[from] bincode::error::DecodeError),
    #[error("IO error")]
    Io(#[from] std::io::Error),
    #[error("Header error")]
    Header(#[from] crate::header::Error),
    #[error("Four section dictionary error")]
    FourSectDict(#[from] crate::four_sect_dict::Error),
    #[error("Bitmap error")]
    BitmapError(#[from] bitmap::Error),
    #[error("Sequence error")]
    SequenceError(#[from] sequence::Error),
    /// One of the cross-component invariants of `TriplesBitmapGeneric`
    /// failed. The most likely cause is a stale or corrupted hybrid cache:
    /// the cached `wavelet_y`/`bitmap_y`/`adjlist_z`/`op_index` were built
    /// from a different HDT file than the one currently being mapped.
    #[error("triples bitmap components are inconsistent: {0}")]
    InconsistentComponents(String),
}

impl<S: SequenceAccess, B: BitmapAccess> fmt::Debug for TriplesBitmapGeneric<S, B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "total size {}", ByteSize(self.size_in_bytes() as u64))?;
        writeln!(f, "adjlist_z {:#?}", self.adjlist_z)?;
        writeln!(f, "op_index {:#?}", self.op_index)?;
        write!(f, "wavelet_y {}", ByteSize(self.wavelet_y.mem_size(SizeFlags::default()) as u64))
    }
}

// Generic implementation for all TriplesBitmapGeneric variants
impl<S: SequenceAccess, B: BitmapAccess> TriplesBitmapGeneric<S, B> {
    /// Create a new TriplesBitmapGeneric from independently-loaded
    /// components. Used by the hybrid/streaming load path where the four
    /// components come from separately-mapped sections of an HDT file plus
    /// a cached wavelet tree.
    ///
    /// Validates the cross-component invariants:
    /// - `bitmap_y.len()` is in `[wavelet_y.len(), wavelet_y.len() + 64)`.
    ///   Both index the predicate sequence; the wavelet stores the predicate
    ///   at each position and the bitmap marks subject boundaries within the
    ///   same range. The bitmap is padded up to a multiple of 64 bits, so we
    ///   accept up to 63 bits of slack.
    /// - `op_index.len() == adjlist_z.len()` — the inverse object index has
    ///   one entry per triple, same as the Z-level adjacency list. Both are
    ///   raw sequence lengths, no padding.
    ///
    /// (`adjlist_z` and `op_index` already enforce their own intra-component
    /// invariants via [`AdjListGeneric::new`] / [`OpIndexGeneric::new`].)
    ///
    /// # Errors
    /// Returns [`Error::InconsistentComponents`] if any invariant fails.
    /// The most likely cause in production is a stale cache whose offsets
    /// were generated from a different version of the HDT file.
    pub fn from_components(
        order: Order, bitmap_y: B, adjlist_z: AdjListGeneric<S, B>, op_index: OpIndexGeneric<S, B>, wavelet_y: WT,
    ) -> Result<Self> {
        // Same empty-bitmap caveat as AdjListGeneric::new: skip the bitmap_y
        // length check entirely if the wavelet is empty, to avoid the qwt
        // empty-RSNarrow subtract-overflow.
        let w_len = wavelet_y.len();
        if w_len > 0 {
            let by_len = bitmap_y.len();
            if by_len < w_len || by_len - w_len >= 64 {
                return Err(Error::InconsistentComponents(format!(
                    "bitmap_y.len() ({by_len}) cannot back wavelet_y.len() ({w_len}) — cache likely stale"
                )));
            }
        }
        if op_index.len() != adjlist_z.len() {
            return Err(Error::InconsistentComponents(format!(
                "op_index.len() ({}) != adjlist_z.len() ({}) — cache likely stale",
                op_index.len(),
                adjlist_z.len()
            )));
        }
        Ok(Self { order, bitmap_y, adjlist_z, op_index, wavelet_y })
    }

    /// Size in bytes on the heap.
    pub fn size_in_bytes(&self) -> usize {
        self.adjlist_z.size_in_bytes()
            + self.op_index.size_in_bytes()
            + self.wavelet_y.mem_size(SizeFlags::default())
    }

    /// Position in the wavelet index of the first predicate for the given subject ID.
    pub fn find_y(&self, subject_id: Id) -> usize {
        if subject_id == 0 {
            return 0;
        }
        self.bitmap_y.select1(subject_id - 1).unwrap_or_else(|| {
            panic!("invalid s_id {subject_id}, there are only {} subjects", self.bitmap_y.num_ones())
        }) as usize
            + 1
    }

    /// Position in the wavelet index of the last predicate for the given subject ID.
    pub fn last_y(&self, subject_id: usize) -> usize {
        self.find_y(subject_id + 1) - 1
    }

    /// Binary search in the wavelet matrix.
    fn bin_search_y(&self, element: usize, begin: usize, end: usize) -> Option<usize> {
        let mut low = begin;
        let mut high = end;

        while low < high {
            let mid = usize::midpoint(low, high);
            match self.wavelet_y.get(mid).unwrap().cmp(&element) {
                Ordering::Less => low = mid + 1,
                Ordering::Greater => high = mid,
                Ordering::Equal => return Some(mid),
            }
        }
        None
    }

    /// Search the wavelet matrix for the position of a given subject, predicate pair.
    pub fn search_y(&self, subject_id: usize, property_id: usize) -> Option<usize> {
        self.bin_search_y(property_id, self.find_y(subject_id), self.last_y(subject_id) + 1)
    }

    /// Transform the given IDs of the layers in triple section order to a triple ID.
    /// Warning: At the moment only SPO is properly supported anyways, in which case this is equivalent to `[x,y,z]`.
    /// Other orders may lead to undefined behaviour.
    pub const fn coord_to_triple(&self, x: Id, y: Id, z: Id) -> Result<TripleId> {
        if x == 0 || y == 0 || z == 0 {
            return Err(Error::TripleComponentZero(x, y, z));
        }
        match self.order {
            Order::SPO => Ok([x, y, z]),
            Order::SOP => Ok([x, z, y]),
            Order::PSO => Ok([y, x, z]),
            Order::POS => Ok([y, z, x]),
            Order::OSP => Ok([z, x, y]),
            Order::OPS => Ok([z, y, x]),
            Order::Unknown => Err(Error::UnknownTriplesOrder),
        }
    }

    /// Number of triples
    pub fn len(&self) -> usize {
        self.adjlist_z.len()
    }

    /// Whether there are no triples
    pub fn is_empty(&self) -> bool {
        self.adjlist_z.is_empty()
    }

    /// Get the order of the triples
    pub const fn order(&self) -> &Order {
        &self.order
    }
}

/// Sort (object, predicate, pos_y) entries by (object, predicate) and
/// stream them into the two components of an op-index: a position sequence
/// (`cv`) and a bitmap marking object-group boundaries.
///
/// Takes ownership of `entries` so the ~6 GB flat array is freed before the
/// caller assembles the final `Sequence`/`Bitmap` structures (which have
/// their own transient allocations).
///
/// The previous implementation allocated one `Vec<u32>` per distinct object
/// (`vec![Vec::<u32>::with_capacity(4); max_object]`), which for 200M
/// objects consumed ~8 GB of heap in 200M separate allocations. This flat
/// approach uses one contiguous allocation of N × 12 bytes (~6 GB for
/// 500M triples), sorts in-place, and streams in order.
fn build_op_index_from_entries(mut entries: Vec<(u32, u32, u32)>) -> (Vec<u32>, BitVectorMut) {
    // Sort key must include pos_y as a tiebreaker so the ordering is fully
    // determined: entries with the same (object, predicate) that were
    // collected in pos_z order have monotonically non-decreasing pos_y
    // (because bitmap.rank is monotone), which matches the insertion-order
    // preservation of the old stable sort_by_cached_key.
    //
    // par_sort splits into chunks, sorts each in parallel, then merges.
    // For 500M entries this is the single largest time sink in cache
    // generation (~60-120s single-threaded, ~15-30s with rayon).
    use rayon::slice::ParallelSliceMut;
    entries.par_sort_unstable_by_key(|&(obj, pred, pos_y)| (obj, pred, pos_y));

    let n = entries.len();
    let mut bitmap = BitVectorMut::new();
    let mut cv = Vec::<u32>::with_capacity(n);
    let mut prev_object = 0u32;
    for &(object, _, pos_y) in &entries {
        bitmap.push(object != prev_object);
        prev_object = object;
        cv.push(pos_y);
    }
    // `entries` dropped here — frees the sorted array before Sequence::new_from_u32.
    (cv, bitmap)
}

// Specialized implementation for in-memory TriplesBitmap
// These methods build indexes and thus always use in-memory structures
impl TriplesBitmap {
    /// Builds the necessary indexes and constructs TriplesBitmap.
    pub fn new(order: Order, sequence_y: &Sequence, bitmap_y: Bitmap, adjlist_z: AdjList) -> Self {
        use rayon::prelude::*;
        let wavelet_y = WT::from_iter(sequence_y);

        let entries = adjlist_z.sequence.entries;
        // Collect (object, predicate, pos_y) tuples for the op-index build.
        // Each iteration does one sequence.get + one bitmap.rank + one wavelet.get;
        // all are read-only so we parallelize with rayon for ~6-8× speedup.
        let pairs: Vec<(u32, u32, u32)> = (0..entries)
            .into_par_iter()
            .filter_map(|pos_z| {
                let object = adjlist_z.sequence.get(pos_z);
                if object == 0 {
                    error!("ERROR: There is a zero value in the Z level.");
                    return None;
                }
                let pos_y = adjlist_z.bitmap.rank(pos_z) as u32;
                let predicate = wavelet_y.get(pos_y as usize).unwrap() as u32;
                Some((object as u32, predicate, pos_y))
            })
            .collect();

        // Sort + stream → op-index components. `pairs` is consumed and freed.
        let (cv, bitmap_index_bitvector) = build_op_index_from_entries(pairs);

        let bv = BitVector::from(bitmap_index_bitvector);
        let bitmap_index = Bitmap { dict: RSNarrow::from(bv) };
        let op_index_sequence = InMemorySequence::new(Sequence::new_from_u32(&cv));
        let op_index = OpIndexGeneric::new(op_index_sequence, InMemoryBitmap::new(bitmap_index));

        // Wrap adjlist_z components in generic wrappers
        let adjlist_z_generic =
            AdjListGeneric::new(InMemorySequence::new(adjlist_z.sequence), InMemoryBitmap::new(adjlist_z.bitmap));

        Self { order, bitmap_y: InMemoryBitmap::new(bitmap_y), adjlist_z: adjlist_z_generic, op_index, wavelet_y }
    }

    /// Creates a TriplesBitmap from cached components.
    /// The wavelet_y is provided from cache, and the op_index is rebuilt.
    #[cfg(feature = "cache")]
    pub fn from_cache(
        order: Order, bitmap_y: Bitmap, adjlist_z: AdjListGeneric<InMemorySequence, InMemoryBitmap>, wavelet_y: WT,
    ) -> Self {
        use rayon::prelude::*;
        let entries = adjlist_z.sequence.len();
        let pairs: Vec<(u32, u32, u32)> = (0..entries)
            .into_par_iter()
            .filter_map(|pos_z| {
                let object = adjlist_z.sequence.get(pos_z);
                if object == 0 {
                    error!("ERROR: There is a zero value in the Z level.");
                    return None;
                }
                let pos_y = adjlist_z.bitmap.rank(pos_z) as u32;
                let predicate = wavelet_y.get(pos_y as usize).unwrap() as u32;
                Some((object as u32, predicate, pos_y))
            })
            .collect();

        let (cv, bitmap_index_bitvector) = build_op_index_from_entries(pairs);

        let bv = BitVector::from(bitmap_index_bitvector);
        let bitmap_index = Bitmap { dict: RSNarrow::from(bv) };
        let op_index_sequence = InMemorySequence::new(Sequence::new_from_u32(&cv));
        let op_index = OpIndexGeneric::new(op_index_sequence, InMemoryBitmap::new(bitmap_index));

        Self { order, bitmap_y: InMemoryBitmap::new(bitmap_y), adjlist_z, op_index, wavelet_y }
    }

    /// Creates a new TriplesBitmap from a list of sorted RDF triples.
    ///
    /// Takes the triples by value so the encoded-triple buffer (24 B ×
    /// num_triples) can be freed as soon as the y/z arrays and bitmaps are
    /// built, instead of staying resident through the memory-heavy op-index
    /// construction in [`TriplesBitmap::new`].
    pub fn from_triples(triples: Vec<TripleId>) -> Self {
        let mut y_bitmap = BitVectorMut::new();
        let mut z_bitmap = BitVectorMut::new();
        let mut array_y = Vec::new();
        let mut array_z = Vec::new();

        let mut last_x = 0;
        let mut last_y = 0;
        let mut last_z = 0;

        for (i, triple) in triples.iter().enumerate() {
            let [x, y, z] = *triple;

            assert!(!(x == 0 || y == 0 || z == 0), "triple IDs should never be zero");

            if i == 0 {
                array_y.push(y);
            } else if x != last_x {
                assert!(x == last_x + 1, "the subjects must be correlative.");
                y_bitmap.push(true);
                array_y.push(y);
                z_bitmap.push(true);
            } else if y != last_y {
                assert!(y >= last_y, "the predicates must be in increasing order.");
                y_bitmap.push(false);
                array_y.push(y);
                z_bitmap.push(true);
            } else {
                assert!(z >= last_z, "the objects must be in increasing order");
                z_bitmap.push(false);
            }
            array_z.push(z);

            last_x = x;
            last_y = y;
            last_z = z;
        }
        // The encoded triples have been fully read into the y/z arrays and
        // bitmaps; free them (~24 B × num_triples) before the op-index build
        // peak in TriplesBitmap::new rather than holding them until return.
        drop(triples);
        y_bitmap.push(true);
        let n = y_bitmap.len();
        y_bitmap.extend_with_zeros(n.div_ceil(64) * 64 - n);
        z_bitmap.push(true);
        let bitmap_y = Bitmap::from(y_bitmap);
        let bitmap_z = Bitmap::from(z_bitmap);
        let sequence_y = Sequence::new(&array_y);
        let sequence_z = Sequence::new(&array_z);
        // The plain usize arrays are now redundant with the bit-packed
        // sequences; free them (~8 B × len each) before the op-index build.
        drop(array_y);
        drop(array_z);
        let adjlist_z = AdjList::new(sequence_z, bitmap_z);
        TriplesBitmap::new(Order::SPO, &sequence_y, bitmap_y, adjlist_z)
    }

    /// read the whole triple section including control information
    pub fn read_sect<R: BufRead>(reader: &mut R) -> Result<Self> {
        let triples_ci = ControlInfo::read(reader)?;

        match &triples_ci.format[..] {
            "<http://purl.org/HDT/hdt#triplesBitmap>" => TriplesBitmap::read(reader, &triples_ci),
            "<http://purl.org/HDT/hdt#triplesList>" => Err(Error::TriplesList),
            f => Err(Error::UnknownTriplesFormat(f.to_owned())),
        }
    }

    fn read<R: BufRead>(reader: &mut R, triples_ci: &ControlInfo) -> Result<Self> {
        let order: Order;
        if let Some(n) = triples_ci.get("order").and_then(|v| v.parse::<u32>().ok()) {
            order = Order::try_from(n)?;
        } else {
            return Err(Error::UnspecifiedTriplesOrder);
        }

        // read bitmaps
        let bitmap_y = Bitmap::read(reader).map_err(|e| Error::Bitmap(Level::Y, e))?;
        let bitmap_z = Bitmap::read(reader).map_err(|e| Error::Bitmap(Level::Z, e))?;

        // read sequences
        let sequence_y = Sequence::read(reader).map_err(|e| Error::Sequence(Level::Y, e))?;
        let sequence_z = Sequence::read(reader).map_err(|e| Error::Sequence(Level::Z, e))?;
        let adjlist_z = AdjList::new(sequence_z, bitmap_z);

        let triples_bitmap = TriplesBitmap::new(order, &sequence_y, bitmap_y, adjlist_z);
        Ok(triples_bitmap)
    }

    pub fn write(&self, write: &mut impl std::io::Write) -> Result<()> {
        ControlInfo::bitmap_triples(self.order.clone() as u32, self.adjlist_z.len() as u32).write(write)?;
        self.bitmap_y.inner().write(write).map_err(|e| Error::Bitmap(Level::Y, e))?;
        self.adjlist_z.bitmap.inner().write(write).map_err(|e| Error::Bitmap(Level::Z, e))?;
        let y = self.wavelet_y.iter().collect::<Vec<_>>();
        Sequence::new(&y).write(write).map_err(|e| Error::Sequence(Level::Y, e))?;
        self.adjlist_z.sequence.inner().write(write).map_err(|e| Error::Sequence(Level::Z, e))?;
        Ok(())
    }
}

impl<'a, S: SequenceAccess, B: BitmapAccess> IntoIterator for &'a TriplesBitmapGeneric<S, B> {
    type Item = TripleId;
    type IntoIter = SubjectIter<'a, S, B>;

    fn into_iter(self) -> Self::IntoIter {
        SubjectIter::new(self)
    }
}

/// Subject, predicate or object ID, starting at 1.
///
/// Subjects and predicate share IDs, starting at 1, for common values.
/// A value of 0 indicates either not found (as a return value) or all of them (in a triple pattern).
/// In the official documentation, u32 is used, however here, usize is used.
/// While u32 caps out at 4 billion, more is not supported by the format anyways so this can probably be changed to u32.
pub type Id = usize;

/// Type for a triple encoded as numeric IDs for subject, predicate and object, respectively.
/// See <https://www.rdfhdt.org/hdt-binary-format/#triples>.
/// Subject index starting at 1 in the combined shared and subject section.
/// Predicate index starting at 1 in the predicate section.
/// Predicate index starting at 1 in the combined shared and object section.
/// When used as a triple, 0 values are invalid.
/// When used as a pattern, 0 values in a position match all values.
pub type TripleId = [Id; 3];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::header::Header;
    use crate::tests::init;
    use crate::{FourSectDict, IdKind};
    use fs_err::File;
    use pretty_assertions::assert_eq;
    use std::io::BufReader;

    /// Iterator over all triples with a given ID in the specified position (subject, predicate or object).
    fn triples_with_id<'a>(t: &'a TriplesBitmap, id: usize, k: IdKind) -> Box<dyn Iterator<Item = TripleId> + 'a> {
        match k {
            IdKind::Subject => Box::new(SubjectIter::with_s(t, id)),
            IdKind::Predicate => Box::new(PredicateIter::new(t, id)),
            IdKind::Object => Box::new(ObjectIter::new(t, id)),
        }
    }

    #[test]
    fn read_triples() -> color_eyre::Result<()> {
        init();
        let file = File::open("tests/resources/snikmeta.hdt")?;
        let mut reader = BufReader::new(file);
        ControlInfo::read(&mut reader)?;
        Header::read(&mut reader)?;
        let _dict = FourSectDict::read(&mut reader)?;
        let triples = TriplesBitmap::read_sect(&mut reader)?;
        let v: Vec<TripleId> = triples.into_iter().collect::<Vec<TripleId>>();
        assert_eq!(v.len(), 328);
        assert_eq!(v[0][0], 1);
        assert_eq!(v[2][0], 1);
        assert_eq!(v[3][0], 2);
        let lens = [48, 23, 175];
        let [num_subjects, _num_predicates, _num_objects] = lens;
        let mut filtered: Vec<TripleId>;
        let funs = [|t: TripleId| t[0], |t: TripleId| t[1], |t: TripleId| t[2]];
        for j in 0..IdKind::KINDS.len() {
            for i in 1..=lens[j] {
                filtered = v.iter().filter(|tid| funs[j](**tid) == i).copied().collect();
                filtered.sort_unstable();
                let mut triples_with_id =
                    triples_with_id(&triples, i, IdKind::KINDS[j]).collect::<Vec<TripleId>>();
                triples_with_id.sort_unstable();
                assert_eq!(filtered, triples_with_id, "triples_with({},{:?})", i, IdKind::KINDS[j]);
            }
        }

        // SubjectIter
        assert_eq!(0, SubjectIter::empty(&triples).count());
        // SPO
        assert_eq!(vec![[14, 14, 154]], SubjectIter::with_pattern(&triples, [14, 14, 154]).collect::<Vec<_>>());
        // SP
        assert_eq!(vec![[14, 14, 154]], SubjectIter::with_pattern(&triples, [14, 14, 0]).collect::<Vec<_>>());
        // S??
        for i in 1..num_subjects {
            assert_eq!(
                SubjectIter::with_s(&triples, i).collect::<Vec<_>>(),
                SubjectIter::with_pattern(&triples, [i, 0, 0]).collect::<Vec<_>>()
            );
        }
        // ??? (all triples)
        assert_eq!(v, SubjectIter::with_pattern(&triples, [0, 0, 0]).collect::<Vec<_>>());
        // SP? where S and P are in the graph, but not together
        assert_eq!(0, SubjectIter::with_pattern(&triples, [12, 14, 154]).count());
        Ok(())
    }
}
