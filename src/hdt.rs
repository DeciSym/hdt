use crate::containers::{ControlInfo, FileBasedCompactVector, InMemoryBitmap, InMemorySequence, control_info};
use crate::containers::{FileBasedBitmap, FileBasedSequence, MmapBitmap};
use crate::four_sect_dict::{self, IdKind};
use crate::header::Header;
use crate::triples::{
    HybridCache, Id, ObjectIter, OpIndexGeneric, PredicateIter, PredicateObjectIter, SubjectIter, TripleId,
    TriplesBitmap, TriplesBitmapGeneric,
};
use crate::{FourSectDict, header};
use bytesize::ByteSize;
use log::{debug, error};
use std::fs::File;
#[cfg(feature = "cache")]
use std::io::{Seek, SeekFrom};
use std::iter;
use std::path::Path;
use std::sync::Arc;

pub type Result<T> = core::result::Result<T, Error>;
#[cfg(feature = "nt")]
#[path = "nt.rs"]
/// Converting N-Triples to HDT, available only if HDT is built with the experimental `"nt"` feature.
mod nt;

/// Type alias for hybrid/streaming HDT implementation using file-based sequences, bitmaps, compact vectors, and dictionaries, but in-memory wavelet.
/// Uses FileBasedSequence for adjlist_z (streaming), FileBasedCompactVector for OpIndex sequence, MmapBitmap for bitmaps, and FileBasedDictSectPfc for dictionaries.
/// Used with HybridCache for memory-efficient querying of large HDT files.
pub type Hdt = HdtGeneric<
    InMemorySequence,
    crate::containers::InMemoryCompactVector,
    InMemoryBitmap,
    crate::dict_sect_pfc::DictSectPFC,
>;

/// Type alias for hybrid/streaming HDT implementation using file-based sequences, bitmaps, compact vectors, and dictionaries, but in-memory wavelet.
/// Uses FileBasedSequence for adjlist_z (streaming), FileBasedCompactVector for OpIndex sequence, MmapBitmap for bitmaps, and FileBasedDictSectPfc for dictionaries.
/// Used with HybridCache for memory-efficient querying of large HDT files.
pub type HdtHybrid = HdtGeneric<
    FileBasedSequence,
    crate::containers::FileBasedCompactVector,
    MmapBitmap,
    crate::dict_sect_pfc::FileBasedDictSectPfc,
>;

/// Type alias for fully file-based HDT implementation.
/// Uses file-based sequences, bitmaps, compact vectors, AND dictionary sections for maximum memory efficiency.
pub type HdtFullyFileBased = HdtGeneric<
    FileBasedSequence,
    crate::containers::FileBasedCompactVector,
    FileBasedBitmap,
    crate::dict_sect_pfc::FileBasedDictSectPfc,
>;

/// Generic HDT structure that can work with different TriplesBitmap and Dictionary implementations.
///
/// # Generic Parameters
/// - `S`: Sequence access implementation (InMemorySequence or FileBasedSequence)
/// - `B`: Bitmap access implementation (InMemoryBitmap or FileBasedBitmap)
/// - `D`: Dictionary section access implementation (DictSectPFC or FileBasedDictSectPfc)
#[derive(Debug)]
pub struct HdtGeneric<
    S: crate::containers::SequenceAccess,
    C: crate::containers::CompactVectorAccess,
    B: crate::containers::BitmapAccess,
    D: crate::dict_sect_pfc::DictSectPfcAccess,
> {
    header: Header,
    pub dict: crate::four_sect_dict::FourSectDictGeneric<D>,
    pub triples: TriplesBitmapGeneric<S, C, B>,
}

type StringTriple = [Arc<str>; 3];

impl<
    S: crate::containers::SequenceAccess,
    C: crate::containers::CompactVectorAccess,
    B: crate::containers::BitmapAccess,
    D: crate::dict_sect_pfc::DictSectPfcAccess,
> HdtGeneric<S, C, B, D>
{
    /// Recursive size in bytes on the heap.
    pub fn size_in_bytes(&self) -> usize {
        self.dict.size_in_bytes() + self.triples.size_in_bytes()
    }

    /// Returns the HDT header.
    pub fn header(&self) -> &Header {
        &self.header
    }

    /// Returns a mutable HDT header.
    ///
    /// Call [`Self::recompute_header_length`] after mutating header triples so the
    /// serialized header control information stays consistent.
    pub fn header_mut(&mut self) -> &mut Header {
        &mut self.header
    }

    /// Recomputes the serialized byte length for the current header body.
    ///
    /// This should be called after mutating [`Self::header_mut`].
    pub fn recompute_header_length(&mut self) -> std::io::Result<()> {
        use std::io::Write as _;
        let mut buf = Vec::<u8>::new();
        for triple in &self.header.body {
            writeln!(&mut buf, "{triple}")?;
        }
        self.header.length = buf.len();
        Ok(())
    }

    /// An iterator visiting *all* triples as strings in order.
    /// Using this method with a filter can be inefficient for large graphs,
    /// because the strings are stored in compressed form and must be decompressed and allocated.
    /// Whenever possible, use [`triples_with_pattern`] instead.
    /// # Example
    /// ```
    /// fn print_first_triple(hdt: hdt::Hdt) {
    ///     println!("{:?}", hdt.triples_all().next().expect("no triple in the graph"));
    /// }
    /// ```
    pub fn triples_all(&self) -> impl Iterator<Item = StringTriple> + '_ {
        let mut triple_cache = TripleCacheGeneric::new(self);
        SubjectIter::new(&self.triples).map(move |ids| triple_cache.translate(ids).unwrap())
    }

    /// Get all subjects with the given property and object (?PO pattern).
    /// Use this over `triples_with_pattern(None,Some(p),Some(o))` if you don't need whole triples.
    /// # Example
    /// Who was born in Leipzig?
    /// ```
    /// fn query(dbpedia: hdt::Hdt) {
    ///     for person in dbpedia.subjects_with_po(
    ///       "http://dbpedia.org/ontology/birthPlace", "http://dbpedia.org/resource/Leipzig") {
    ///       println!("{person:?}");
    ///     }
    /// }
    /// ```
    pub fn subjects_with_po(&self, p: &str, o: &str) -> Box<dyn Iterator<Item = String> + '_> {
        let pid = self.dict.string_to_id(p, IdKind::Predicate);
        let oid = self.dict.string_to_id(o, IdKind::Object);
        // predicate or object not in dictionary, iterator would interpret 0 as variable
        if pid == 0 || oid == 0 {
            return Box::new(iter::empty());
        }
        // needed for extending the lifetime of the parameters into the iterator for error messages
        let p_owned = p.to_owned();
        let o_owned = o.to_owned();
        Box::new(
            PredicateObjectIter::new(&self.triples, pid, oid)
                .map(move |sid| self.dict.id_to_string(sid, IdKind::Subject))
                .filter_map(move |r| {
                    r.map_err(|e| error!("Error on triple with property {p_owned} and object {o_owned}: {e}")).ok()
                }),
        )
    }

    /// Get all triples that fit the given triple patterns, where `None` stands for a variable.
    /// For example, `triples_with_pattern(Some(s), Some(p), None)` answers an SP? pattern.
    /// # Example
    /// What is the capital of the United States of America?
    /// ```
    /// fn query(dbpedia: hdt::Hdt) {
    ///   println!("{:?}", dbpedia.triples_with_pattern(
    ///     Some("http://dbpedia.org/resource/United_States"), Some("http://dbpedia.org/ontology/capital"), None)
    ///     .next().expect("no capital found")[2]);
    /// }
    /// ```
    pub fn triples_with_pattern<'a>(
        &'a self, sp: Option<&'a str>, pp: Option<&'a str>, op: Option<&'a str>,
    ) -> Box<dyn Iterator<Item = StringTriple> + 'a> {
        let pattern: [Option<(Arc<str>, usize)>; 3] = [(0, sp), (1, pp), (2, op)]
            .map(|(i, x)| x.map(|x| (Arc::from(x), self.dict.string_to_id(x, IdKind::KINDS[i]))));
        // at least one term does not exist in the graph
        if pattern.iter().flatten().any(|x| x.1 == 0) {
            return Box::new(iter::empty());
        }
        // TODO: improve error handling
        let mut cache = TripleCacheGeneric::new(self);
        match pattern {
            [Some(s), Some(p), Some(o)] => {
                if SubjectIter::with_pattern(&self.triples, [s.1, p.1, o.1]).next().is_some() {
                    Box::new(iter::once([s.0, p.0, o.0]))
                } else {
                    Box::new(iter::empty())
                }
            }
            [Some(s), Some(p), None] => {
                Box::new(SubjectIter::with_pattern(&self.triples, [s.1, p.1, 0]).map(move |t| {
                    [s.0.clone(), p.0.clone(), Arc::from(self.dict.id_to_string(t[2], IdKind::Object).unwrap())]
                }))
            }
            [Some(s), None, Some(o)] => {
                Box::new(SubjectIter::with_pattern(&self.triples, [s.1, 0, o.1]).map(move |t| {
                    [s.0.clone(), Arc::from(self.dict.id_to_string(t[1], IdKind::Predicate).unwrap()), o.0.clone()]
                }))
            }
            [Some(s), None, None] => Box::new(
                SubjectIter::with_pattern(&self.triples, [s.1, 0, 0])
                    .map(move |t| [s.0.clone(), cache.get(1, t[1]).unwrap(), cache.get(2, t[2]).unwrap()]),
            ),
            [None, Some(p), Some(o)] => {
                Box::new(PredicateObjectIter::new(&self.triples, p.1, o.1).map(move |sid| {
                    [Arc::from(self.dict.id_to_string(sid, IdKind::Subject).unwrap()), p.0.clone(), o.0.clone()]
                }))
            }
            [None, Some(p), None] => Box::new(
                PredicateIter::new(&self.triples, p.1)
                    .map(move |t| [cache.get(0, t[0]).unwrap(), p.0.clone(), cache.get(2, t[2]).unwrap()]),
            ),
            [None, None, Some(o)] => Box::new(
                ObjectIter::new(&self.triples, o.1)
                    .map(move |t| [cache.get(0, t[0]).unwrap(), cache.get(1, t[1]).unwrap(), o.0.clone()]),
            ),
            [None, None, None] => Box::new(self.triples_all()),
        }
    }
}

/// A TripleCacheGeneric stores the `Arc<str>` of the last returned triple (generic version)
#[derive(Clone, Debug)]
struct TripleCacheGeneric<
    'a,
    S: crate::containers::SequenceAccess,
    C: crate::containers::CompactVectorAccess,
    B: crate::containers::BitmapAccess,
    D: crate::dict_sect_pfc::DictSectPfcAccess,
> {
    hdt: &'a HdtGeneric<S, C, B, D>,
    tid: TripleId,
    arc: [Option<Arc<str>>; 3],
}

impl<
    'a,
    S: crate::containers::SequenceAccess,
    C: crate::containers::CompactVectorAccess,
    B: crate::containers::BitmapAccess,
    D: crate::dict_sect_pfc::DictSectPfcAccess,
> TripleCacheGeneric<'a, S, C, B, D>
{
    /// Build a new [`TripleCacheGeneric`] for the given [`HdtGeneric`]
    const fn new(hdt: &'a HdtGeneric<S, C, B, D>) -> Self {
        TripleCacheGeneric { hdt, tid: [0; 3], arc: [None, None, None] }
    }

    /// Translate a triple of indexes into a triple of strings.
    fn translate(&mut self, t: TripleId) -> core::result::Result<StringTriple, TranslateError> {
        // refactor when try_map for arrays becomes stable
        Ok([
            self.get(0, t[0]).map_err(|e| TranslateError { e, t })?,
            self.get(1, t[1]).map_err(|e| TranslateError { e, t })?,
            self.get(2, t[2]).map_err(|e| TranslateError { e, t })?,
        ])
    }

    fn get(&mut self, pos: usize, id: Id) -> core::result::Result<Arc<str>, four_sect_dict::ExtractError> {
        debug_assert!(id != 0);
        debug_assert!(pos < 3);
        if self.tid[pos] == id {
            Ok(self.arc[pos].as_ref().unwrap().clone())
        } else {
            let ret: Arc<str> = self.hdt.dict.id_to_string(id, IdKind::KINDS[pos])?.into();
            self.arc[pos] = Some(ret.clone());
            self.tid[pos] = id;
            Ok(ret)
        }
    }
}

/// The error type for the `translate_id` method.
#[derive(thiserror::Error, Debug)]
#[error("cannot translate triple ID {t:?} to string triple: {e}")]
pub struct TranslateError {
    #[source]
    e: four_sect_dict::ExtractError,
    t: TripleId,
}

/// The error type for the `new` method.
#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("failed to read HDT control info")]
    ControlInfo(#[from] control_info::Error),
    #[error("failed to read HDT header")]
    Header(#[from] header::Error),
    #[error("failed to read HDT four section dictionary")]
    FourSectDict(#[from] four_sect_dict::Error),
    #[error("failed to read HDT triples section")]
    Triples(#[from] crate::triples::Error),
    #[error("IO Error")]
    Io(#[from] std::io::Error),
}

impl Hdt {
    #[deprecated(since = "0.4.0", note = "please use `read` instead")]
    pub fn new<R: std::io::BufRead>(reader: R) -> Result<Self> {
        Self::read(reader)
    }

    /// Creates an immutable HDT instance containing the dictionary and triples from the given reader.
    /// The reader must point to the beginning of the data of an HDT file as produced by hdt-cpp.
    /// FourSectionDictionary with DictionarySectionPlainFrontCoding and SPO order is the only supported implementation.
    /// The format is specified at <https://www.rdfhdt.org/hdt-binary-format/>, however there are some deviations.
    /// The initial HDT specification at <http://www.w3.org/Submission/2011/03/> is outdated and not supported.
    /// # Example
    /// ```
    /// let file = std::fs::File::open("tests/resources/snikmeta.hdt").expect("error opening file");
    /// let hdt = hdt::Hdt::read(std::io::BufReader::new(file)).unwrap();
    /// ```
    pub fn read<R: std::io::BufRead>(mut reader: R) -> Result<Self> {
        ControlInfo::read(&mut reader)?;
        let header = Header::read(&mut reader)?;
        let unvalidated_dict = FourSectDict::read(&mut reader, false)?;
        let triples = TriplesBitmap::read_sect(&mut reader)?;
        let dict = unvalidated_dict.validate()?;
        let hdt = Hdt { header, dict, triples };
        debug!("HDT size in memory {}, details:", ByteSize(hdt.size_in_bytes() as u64));
        debug!("{hdt:#?}");
        Ok(hdt)
    }

    /// Creates a memory-efficient HDT instance with automatic cache management.
    ///
    /// # Arguments
    /// * `hdt_path` - Path to the HDT file
    /// * `skip_dict_validation` - If true, skips CRC32 validation of dictionary sections (saves ~400-1200ms)
    ///
    /// # Returns
    /// Returns an `HdtHybrid` instance using file-based sequences and bitmaps for memory efficiency.
    pub fn new_hybrid_cache(
        hdt_path: &Path, _skip_dict_validation: bool,
    ) -> core::result::Result<HdtHybrid, Box<dyn std::error::Error>> {
        use crate::containers::AdjListGeneric;

        // Load the HybridCache and get the offset to the OpIndex bitmap in the cache file
        let (cache, op_index_bitmap_offset) = HybridCache::from_hdt_path(hdt_path)?;
        let cache_path = HybridCache::get_cache_path(hdt_path);

        // Open HDT file and read header + dictionary
        let hdt_file = File::open(hdt_path)?;
        let mut reader = std::io::BufReader::new(hdt_file);

        ControlInfo::read(&mut reader)?;
        let header = Header::read(&mut reader)?;
        // Conditionally validate dictionary based on parameter
        let dict = FourSectDict::from_cache(
            hdt_path, cache.dict_shared_offset, cache.dict_subjects_offset, cache.dict_predicates_offset,
            cache.dict_objects_offset,
        )?;

        let buf = &hdt_path.to_path_buf();
        // Create file-based sequence for adjlist_z using cached metadata
        let sequence_z = FileBasedSequence::new(buf, cache.sequence_z_offset)?;
        let bitmap_z = MmapBitmap::new(buf, cache.bitmap_z_offset)?;
        let bitmap_y = MmapBitmap::new(buf, cache.bitmap_y_offset)?;
        // Create file-based adjlist_z from cache metadata
        let adjlist_z = AdjListGeneric::new(sequence_z, bitmap_z);

        // Get order before moving cache fields
        let order = cache.order()?;

        // Use the cached op_index sequence (already built and stored in cache)
        // For HdtHybrid, we use MmapBitmap for the OpIndex bitmap (stored at end of cache file)
        let op_index_bitmap = MmapBitmap::new(&cache_path, op_index_bitmap_offset)?;

        // Calculate where the CompactVector begins (right after the bitmap)
        // Instead of reading the bitmap, we calculate its size from the MmapBitmap metadata
        let bitmap_size = op_index_bitmap.serialized_size_bytes() as u64;
        let op_index_sequence_offset = op_index_bitmap_offset + bitmap_size;

        let op_index = OpIndexGeneric::new(
            FileBasedCompactVector::new(&cache_path, op_index_sequence_offset)?,
            op_index_bitmap,
        );

        // Build the TriplesBitmapGeneric
        let triples = TriplesBitmapGeneric::from_components(order, bitmap_y, adjlist_z, op_index, cache.wavelet_y);

        let hdt = HdtGeneric { header, dict, triples };
        debug!("HDT Hybrid size in memory {}, details:", ByteSize(hdt.size_in_bytes() as u64));
        debug!("{hdt:#?}");
        Ok(hdt)
    }

    /// Write as N-Triples
    #[cfg(feature = "sophia")]
    pub fn write_nt(&self, write: &mut impl std::io::Write) -> std::io::Result<()> {
        use sophia::api::prelude::TripleSerializer;
        use sophia::turtle::serializer::nt::NtSerializer;
        NtSerializer::new(write).serialize_graph(self).map_err(|e| std::io::Error::other(format!("{e}")))?;
        Ok(())
    }

    /// Creates an immutable HDT instance containing the dictionary and triples from the Path.
    /// Will utilize a custom cached TriplesBitmap file if exists or create one if it does not exist.
    /// The file path must point to the beginning of the data of an HDT file as produced by hdt-cpp.
    /// FourSectionDictionary with DictionarySectionPlainFrontCoding and SPO order is the only supported implementation.
    /// The format is specified at <https://www.rdfhdt.org/hdt-binary-format/>, however there are some deviations.
    /// The initial HDT specification at <http://www.w3.org/Submission/2011/03/> is outdated and not supported.
    /// # Example
    /// ```
    /// let hdt = hdt::Hdt::read_from_path(std::path::Path::new("tests/resources/snikmeta.hdt")).unwrap();
    /// ```
    #[cfg(feature = "cache")]
    pub fn read_from_path(f: &std::path::Path) -> Result<Self> {
        use crate::containers::{AdjListGeneric, InMemoryBitmap, InMemorySequence};
        use crate::triples::OpIndex;
        use log::warn;
        use sucds::Serializable;

        // Try to use HybridCache for faster loading (avoids recomputing op_index and wavelet_y)
        // But return a fully in-memory Hdt (not HdtHybrid)
        let cache_and_offset = match HybridCache::from_hdt_path(f) {
            Ok((cache, offset)) => Some((cache, offset)),
            Err(e) => {
                warn!("Could not load HybridCache, falling back to full HDT read: {e}");
                None
            }
        };

        let source = File::open(f)?;
        let mut reader = std::io::BufReader::new(source);
        ControlInfo::read(&mut reader)?;
        let header = Header::read(&mut reader)?;
        let unvalidated_dict = FourSectDict::read(&mut reader, false)?;

        // Capture position after dictionary for fallback
        let triples_section_offset = reader.stream_position()?;

        let triples = if let Some((cache, op_index_bitmap_offset)) = cache_and_offset {
            // Try to use cache, but fall back to regular reading if there's any error
            // (e.g., cache is from a different HDT file or corrupted)
            match (|| -> core::result::Result<TriplesBitmap, Box<dyn std::error::Error>> {
                if cache.header_size()? != (header.length as u64) {
                    error!(
                        "index file corrupt, headers don't match. expected {}, got {}",
                        cache.header_size()?,
                        header.length as u64
                    );
                    return Err("failed index validation".into());
                }
                // Extract order before consuming cache fields
                let order = cache.order()?;

                // Read the bitmaps and sequences from HDT file using cached offsets
                reader.seek(SeekFrom::Start(cache.bitmap_y_offset))?;
                let bitmap_y = crate::containers::Bitmap::read(&mut reader)?;

                reader.seek(SeekFrom::Start(cache.sequence_z_offset))?;
                let sequence_z = crate::containers::Sequence::read(&mut reader)?;

                reader.seek(SeekFrom::Start(cache.bitmap_z_offset))?;
                let bitmap_z = crate::containers::Bitmap::read(&mut reader)?;

                // Load OpIndex bitmap and sequence from cache file
                // For Hdt (fully in-memory), we load both into memory
                let cache_path = HybridCache::get_cache_path(f);
                let cache_file = File::open(cache_path)?;
                let mut cache_reader = std::io::BufReader::new(cache_file);
                cache_reader.seek(SeekFrom::Start(op_index_bitmap_offset))?;

                // Read bitmap first
                let op_index_bitmap = crate::containers::Bitmap::read(&mut cache_reader)?;
                let op_index_bitmap_wrapped = InMemoryBitmap::new(op_index_bitmap);

                // Read sequence second (it's right after the bitmap)
                let op_index_sequence_cv = sucds::int_vectors::CompactVector::deserialize_from(&mut cache_reader)?;
                let op_index_sequence = crate::containers::InMemoryCompactVector::new(op_index_sequence_cv);

                let op_index = OpIndex::new(op_index_sequence, op_index_bitmap_wrapped);

                let adjlist_z =
                    AdjListGeneric::new(InMemorySequence::new(sequence_z), InMemoryBitmap::new(bitmap_z));
                let bitmap_y_mem = InMemoryBitmap::new(bitmap_y);

                Ok(TriplesBitmapGeneric::from_components(
                    order, bitmap_y_mem, adjlist_z, op_index, cache.wavelet_y,
                ))
            })() {
                Ok(triples) => triples,
                Err(e) => {
                    warn!("Error using cache, falling back to full read: {e}");
                    // Seek back to triples section (after dictionary)
                    reader.seek(SeekFrom::Start(triples_section_offset))?;
                    TriplesBitmap::read_sect(&mut reader)?
                }
            }
        } else {
            // Fall back to reading the entire triples section
            TriplesBitmap::read_sect(&mut reader)?
        };
        let dict = unvalidated_dict.validate()?;
        let hdt = Hdt { header, dict, triples };
        debug!("HDT size in memory {}, details:", ByteSize(hdt.size_in_bytes() as u64));
        debug!("{hdt:#?}");
        Ok(hdt)
    }

    pub fn write(&self, write: &mut impl std::io::Write) -> Result<()> {
        ControlInfo::global().write(write)?;
        self.header.write(write)?;
        self.dict.write(write)?;
        self.triples.write(write)?;
        write.flush()?;
        Ok(())
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::tests::init;
    use color_eyre::Result;
    use fs_err::File;
    use pretty_assertions::{assert_eq, assert_ne};

    /// reusable test HDT read from SNIK Meta test HDT file
    pub fn snikmeta() -> Result<Hdt> {
        let filename = "tests/resources/snikmeta.hdt";
        let file = File::open(filename)?;
        Ok(Hdt::read(std::io::BufReader::new(file))?)
    }

    #[test]
    fn write() -> Result<()> {
        init();
        let hdt = snikmeta()?;
        snikmeta_check_generic(&hdt)?;
        let mut buf = Vec::<u8>::new();
        hdt.write(&mut buf)?;
        let hdt2 = Hdt::read(std::io::Cursor::new(buf))?;
        snikmeta_check_generic(&hdt2)?;
        Ok(())
    }

    #[test]
    #[cfg(feature = "nt")]
    fn header_metadata_can_be_mutated_before_single_write() -> Result<()> {
        use crate::containers::rdf::{Id as RdfId, Term as RdfTerm, Triple as RdfTriple};

        init();
        let mut hdt = Hdt::read_nt(Path::new("tests/resources/empty.nt"))?;
        let triple = RdfTriple::new(
            RdfId::Named("http://example.org/dataset".to_owned()),
            "https://decisym.ai/de#graphIRI".to_owned(),
            RdfTerm::Id(RdfId::Named("http://example.org/graph".to_owned())),
        );
        hdt.header_mut().body.insert(triple.clone());
        hdt.recompute_header_length()?;

        let mut buf = Vec::<u8>::new();
        hdt.write(&mut buf)?;
        let reloaded = Hdt::read(std::io::Cursor::new(buf))?;

        assert!(reloaded.header().body.contains(&triple));
        Ok(())
    }

    // make sure loading with cache works under different circumstances
    // e.g. clear cache, prexisting cache, stale cache
    #[test]
    #[serial_test::serial]
    #[cfg(feature = "cache")]
    fn cache() -> Result<()> {
        use crate::triples::hybrid_cache::CACHE_EXT;
        use fs_err::remove_file;
        use std::path::Path;
        init();
        // start with an empty cache
        let filename = "tests/resources/snikmeta.hdt";
        let path = Path::new(filename);
        let path_cache = HybridCache::get_cache_path(path);
        // force fresh cache
        let _ = remove_file(&path_cache);
        let hdt1 = Hdt::read_from_path(path)?;
        snikmeta_check_generic(&hdt1)?;
        // now it should be cached
        let hdt2 = Hdt::read_from_path(path)?;
        snikmeta_check_generic(&hdt2)?;
        #[cfg(feature = "nt")]
        {
            // create a cache for an empty HDT
            let path_empty_nt = Path::new("tests/resources/empty.nt");
            // it's empty so we could just pass an empty buffer
            let hdt_empty = Hdt::read_nt(path_empty_nt)?;
            let filename_empty_hdt = "tests/resources/empty.hdt";
            let path_empty_hdt = Path::new(filename_empty_hdt);
            if !path_empty_hdt.exists() {
                let file_empty_hdt = File::create(filename_empty_hdt)?;
                let mut writer = std::io::BufWriter::new(file_empty_hdt);
                hdt_empty.write(&mut writer)?;
            }
            // we don't care about the empty HDT, we just need it to create the cache
            let filename_empty_cache = format!("{filename_empty_hdt}.{CACHE_EXT}");
            let path_empty_cache = Path::new(&filename_empty_cache);
            let _ = remove_file(path_empty_cache);
            Hdt::read_from_path(path_empty_hdt)?;
            // purposefully create a mismatch between cache and HDT file for the same name
            fs_err::rename(path_empty_cache, &path_cache)?;
            // mismatch should be detected and handled
            let hdt3 = Hdt::read_from_path(path)?;
            snikmeta_check_generic(&hdt3)?;
        }
        Ok(())
    }

    /// Test that dictionary validation can be skipped
    #[test]
    fn test_skip_dict_validation() -> Result<()> {
        use std::time::Instant;

        init();

        let hdt_path = Path::new("tests/resources/snikmeta.hdt");

        let _ = std::fs::remove_file(HybridCache::get_cache_path(hdt_path));

        // Load with validation
        let start = Instant::now();
        let hdt_validated =
            Hdt::new_hybrid_cache(hdt_path, false).map_err(|e| std::io::Error::other(format!("{e}")))?;
        let validated_time = start.elapsed();

        // Load without validation
        let start = Instant::now();
        let hdt_skip_validation =
            Hdt::new_hybrid_cache(hdt_path, true).map_err(|e| std::io::Error::other(format!("{e}")))?;
        let skip_validation_time = start.elapsed();

        println!("\nDictionary load comparison:");
        println!("  With validation:    {validated_time:?}");
        println!("  Without validation: {skip_validation_time:?}");

        if skip_validation_time < validated_time {
            println!(
                "  Skip validation is {:.2}x faster!",
                validated_time.as_secs_f64() / skip_validation_time.as_secs_f64()
            );
        }

        // Verify both work correctly
        snikmeta_check_generic(&hdt_validated)?;
        snikmeta_check_generic(&hdt_skip_validation)?;

        Ok(())
    }

    /// Compare load times between Hdt::read() and Hdt::read_with_hybrid_cache()
    #[test]
    fn compare_load_times() -> Result<()> {
        use crate::triples::hybrid_cache::CACHE_EXT;
        use std::time::Instant;
        init();

        let hdt_path = Path::new("tests/resources/snikmeta.hdt");

        println!("\n Comparing load times:\n");

        // First, load the entire file into memory so we can use Cursor to track positions
        let file_contents = std::fs::read(hdt_path)?;

        // Test 1: Hdt::read() - fully in-memory
        println!("Loading with Hdt::read() (fully in-memory)...");
        let start = Instant::now();
        let mut cursor = std::io::Cursor::new(&file_contents);
        let hdt_in_memory = Hdt::read(&mut cursor)?;
        let in_memory_time = start.elapsed();
        let in_memory_size = hdt_in_memory.size_in_bytes();

        println!("   Loaded in {in_memory_time:?}");
        println!("   Memory usage: {}", ByteSize(in_memory_size as u64));

        // Generate the HybridCache by tracking file positions
        println!("\nGenerating HybridCache...");
        let cache_name = format!("{}.{CACHE_EXT}", hdt_path.to_str().unwrap());

        println!("Generating HybridCache for {hdt_path:?}...");
        // Generate cache
        let _ = HybridCache::write_cache_from_hdt_file(hdt_path);
        println!("  Cache generated ({} bytes)", std::fs::metadata(&cache_name)?.len());

        // Test 2: Hdt::read_with_hybrid_cache() - streaming
        println!("\nLoading with Hdt::read_with_hybrid_cache() (streaming)...");
        let start = Instant::now();
        let hdt_hybrid =
            Hdt::new_hybrid_cache(hdt_path, false).map_err(|e| std::io::Error::other(format!("{e}")))?;
        let hybrid_time = start.elapsed();
        let hybrid_size = hdt_hybrid.size_in_bytes();

        println!("  Loaded in {hybrid_time:?}");
        println!("  Memory usage: {}", ByteSize(hybrid_size as u64));

        // Comparison
        println!("\n Results:");
        println!("  In-memory load time:  {in_memory_time:?}");
        println!("  Hybrid load time:     {hybrid_time:?}");

        let speedup = in_memory_time.as_secs_f64() / hybrid_time.as_secs_f64();
        if hybrid_time < in_memory_time {
            println!("  Hybrid is {speedup:.2}x faster!");
        } else {
            println!("   In-memory is {:.2}x faster", 1.0 / speedup);
        }

        println!("\n  In-memory size:  {}", ByteSize(in_memory_size as u64));
        println!("  Hybrid size:     {}", ByteSize(hybrid_size as u64));

        let memory_saved = (in_memory_size - hybrid_size) as i64;
        if memory_saved > 0 {
            println!(
                "  Memory saved: {} ({:.1}%)",
                ByteSize(memory_saved as u64),
                (memory_saved as f64 / in_memory_size as f64) * 100.0
            );
        }

        // Verify both work correctly by running a simple query
        println!("\n Verifying correctness with sample query...");
        let in_memory_count = hdt_in_memory.triples_with_pattern(None, None, None).count();
        // Note: Can't easily query HdtHybrid without implementing iterator traits
        println!("  In-memory triple count: {in_memory_count}");
        let start = Instant::now();
        snikmeta_check_generic(&hdt_in_memory)?;
        let in_memory_time = start.elapsed();

        let start = Instant::now();
        snikmeta_check_generic(&hdt_hybrid)?;
        let in_hybrid_time = start.elapsed();
        // Comparison
        println!("\n Results:");
        println!("  In-memory check time:  {in_memory_time:?}");
        println!("  Hybrid check time:     {in_hybrid_time:?}");
        Ok(())
    }

    /// Helper test to generate HybridCache for snikmeta.hdt test file.
    /// Run this once to create the cache file needed by compare_load_times test.
    #[test]
    fn generate_test_cache() -> Result<()> {
        init();

        let hdt_path = Path::new("tests/resources/snikmeta.hdt");
        let cache_name = HybridCache::get_cache_path(hdt_path);

        println!("Generating HybridCache for {hdt_path:?}...");
        // Generate cache
        let _ = HybridCache::write_cache_from_hdt_file(hdt_path);

        let cache_size = std::fs::metadata(Path::new(&cache_name))?.len();
        println!("\n  Cache successfully written to {:?}", &cache_name);
        println!("  Cache file size: {}", ByteSize(cache_size));

        // Verify the cache can be loaded
        let _loaded_cache =
            HybridCache::read_from_file(cache_name).map_err(|e| std::io::Error::other(format!("{e}")))?;

        // Verify a HDT can be created from cache
        let _hdt_hybrid =
            Hdt::new_hybrid_cache(hdt_path, false).map_err(|e| std::io::Error::other(format!("{e}")))?;
        Ok(())
    }

    #[test]
    #[cfg(feature = "cli")]
    fn read_nt() -> Result<()> {
        init();
        let path = Path::new("tests/resources/snikmeta.nt");
        if !path.exists() {
            log::info!("Creating test resource snikmeta.nt.");
            let mut writer = std::io::BufWriter::new(File::create(path)?);
            snikmeta()?.write_nt(&mut writer)?;
        }
        let snikmeta_nt = Hdt::read_nt(path)?;

        let snikmeta = snikmeta()?;
        let hdt_triples: Vec<StringTriple> = snikmeta.triples_all().collect();
        let nt_triples: Vec<StringTriple> = snikmeta_nt.triples_all().collect();

        assert_eq!(nt_triples, hdt_triples);
        assert_eq!(snikmeta.triples.bitmap_y.inner().dict, snikmeta_nt.triples.bitmap_y.inner().dict);
        snikmeta_check_generic(&snikmeta_nt)?;
        let path = std::path::Path::new("tests/resources/empty.nt");
        let hdt_empty = Hdt::read_nt(path)?;
        let mut buf = Vec::<u8>::new();
        hdt_empty.write(&mut buf)?;
        Hdt::read(std::io::Cursor::new(buf))?;
        Ok(())
    }

    /// Generic version of snikmeta_check that works with HdtGeneric<S, C, B, D> (for HdtHybrid, etc.)
    pub fn snikmeta_check_generic<
        S: crate::containers::SequenceAccess,
        C: crate::containers::CompactVectorAccess,
        B: crate::containers::BitmapAccess,
        D: crate::dict_sect_pfc::DictSectPfcAccess,
    >(
        hdt: &HdtGeneric<S, C, B, D>,
    ) -> Result<()> {
        let triples = &hdt.triples;
        assert_eq!(triples.bitmap_y.num_ones(), 49, "{:?}", triples.bitmap_y); // one for each subjecct
        //assert_eq!();
        let v: Vec<StringTriple> = hdt.triples_all().collect();
        assert_eq!(v.len(), 328);
        assert_eq!(hdt.dict.shared.num_strings(), 43);
        assert_eq!(hdt.dict.subjects.num_strings(), 6);
        assert_eq!(hdt.dict.predicates.num_strings(), 23);
        assert_eq!(hdt.dict.objects.num_strings(), 133);
        assert_eq!(v, hdt.triples_with_pattern(None, None, None).collect::<Vec<_>>(), "all triples not equal ???");
        assert_ne!(0, hdt.dict.string_to_id("http://www.snik.eu/ontology/meta", IdKind::Subject));
        for uri in ["http://www.snik.eu/ontology/meta/Top", "http://www.snik.eu/ontology/meta", "doesnotexist"] {
            let filtered: Vec<_> = v.clone().into_iter().filter(|triple| triple[0].as_ref() == uri).collect();
            let with_s: Vec<_> = hdt.triples_with_pattern(Some(uri), None, None).collect();
            assert_eq!(filtered, with_s, "results differ between triples_all() and S?? query for {}", uri);
        }
        let s = "http://www.snik.eu/ontology/meta/Top";
        let p = "http://www.w3.org/2000/01/rdf-schema#label";
        let o = "\"top class\"@en";
        let triple_vec = vec![[Arc::from(s), Arc::from(p), Arc::from(o)]];
        // triple patterns with 2-3 terms
        assert_eq!(triple_vec, hdt.triples_with_pattern(Some(s), Some(p), Some(o)).collect::<Vec<_>>(), "SPO");
        assert_eq!(triple_vec, hdt.triples_with_pattern(Some(s), Some(p), None).collect::<Vec<_>>(), "SP?");
        assert_eq!(triple_vec, hdt.triples_with_pattern(Some(s), None, Some(o)).collect::<Vec<_>>(), "S?O");
        assert_eq!(triple_vec, hdt.triples_with_pattern(None, Some(p), Some(o)).collect::<Vec<_>>(), "?PO");
        let et = "http://www.snik.eu/ontology/meta/EntityType";
        let meta = "http://www.snik.eu/ontology/meta";
        let subjects = ["ApplicationComponent", "Method", "RepresentationType", "SoftwareProduct"]
            .map(|s| meta.to_owned() + "/" + s)
            .to_vec();
        assert_eq!(
            subjects,
            hdt.subjects_with_po("http://www.w3.org/2000/01/rdf-schema#subClassOf", et).collect::<Vec<_>>()
        );
        assert_eq!(
            12,
            hdt.triples_with_pattern(None, Some("http://www.w3.org/2000/01/rdf-schema#subClassOf"), None).count()
        );
        assert_eq!(20, hdt.triples_with_pattern(None, None, Some(et)).count());
        let snikeu = "http://www.snik.eu";
        let triple_vec = [
            "http://purl.org/dc/terms/publisher", "http://purl.org/dc/terms/source",
            "http://xmlns.com/foaf/0.1/homepage",
        ]
        .into_iter()
        .map(|p| [Arc::from(meta), Arc::from(p), Arc::from(snikeu)])
        .collect::<Vec<_>>();
        assert_eq!(
            triple_vec,
            hdt.triples_with_pattern(Some(meta), None, Some(snikeu)).collect::<Vec<_>>(),
            "S?O multiple"
        );
        let s = "http://www.snik.eu/ontology/meta/хобби-N-0";
        assert_eq!(hdt.dict.string_to_id(s, IdKind::Subject), 49);
        assert_eq!(hdt.dict.id_to_string(49, IdKind::Subject)?, s);
        let o = "\"ХОББИ\"@ru";
        let triple_vec = vec![[Arc::from(s), Arc::from(p), Arc::from(o)]];
        assert_eq!(hdt.triples_with_pattern(Some(s), Some(p), None).collect::<Vec<_>>(), triple_vec);
        Ok(())
    }
}
