#![allow(missing_docs)]
// temporary while we figure out what should be public in the end
/// Four section dictionary.
use crate::dict_sect_pfc;
use crate::triples::Id;
use crate::{ControlInfo, DictSectPFC};
use std::io::BufRead;
use std::sync::Arc;
#[cfg(not(any(target_arch = "wasm32", target_arch = "wasm64")))]
use std::thread::JoinHandle;
use thiserror::Error;

pub type Result<T> = core::result::Result<T, Error>;

/// Position in an RDF triple.
#[derive(Debug, Clone, Copy)]
pub enum IdKind {
    /// IRI or blank node in the first position of a triple.
    Subject,
    /// IRI in the second position of a triple.
    Predicate,
    /// IRI, blank node or literal in the third position of a triple.
    Object,
}

impl IdKind {
    pub const KINDS: [IdKind; 3] = [IdKind::Subject, IdKind::Predicate, IdKind::Object];
}

/// Generic four section dictionary with plain front coding.
/// Dictionary with shared, subject, predicate and object sections.
/// Types specified as <http://purl.org/HDT/hdt#dictionaryFour>.
/// See <https://www.rdfhdt.org/hdt-internals/#dictionary>.
///
/// Generic over dictionary section implementation (D) to support both in-memory and file-based access.
#[derive(Debug)]
#[cfg_attr(test, derive(PartialEq))]
pub struct FourSectDictGeneric<D: dict_sect_pfc::DictSectPfcAccess> {
    /// The shared section contains URIs that occur both in subject and object position. Its IDs start at one.
    pub shared: D,
    /// URIs that only occur as subjects. Their IDs start at the last ID of the shared section + 1.
    pub subjects: D,
    /// The predicate section has its own separate numbering starting from 1.
    pub predicates: D,
    /// URIs and literals that only occur as objects . Their IDs start at the last ID of the shared section + 1.
    pub objects: D,
}

/// Type alias for the standard in-memory four section dictionary
pub type FourSectDict = FourSectDictGeneric<DictSectPFC>;

/// Type alias for the memory-mapped four section dictionary used by `HdtHybrid`.
pub type FourSectDictMmap = FourSectDictGeneric<dict_sect_pfc::MmapDictSectPfc>;

/// Designates one of the four sections.
#[derive(Debug)]
pub enum SectKind {
    /// section for terms that appear as both subject and object
    Shared,
    /// section for terms that only appear as subjects
    Subject,
    /// section for terms that only appear as predicates
    Predicate,
    /// sections for terms that only appear as objects
    Object,
}

/// Wraps an extraction error with additional information on which dictionary section it occurred in.
#[derive(Error, Debug)]
#[error("four sect dict error id_to_string({id},IdKind::{id_kind:?}) in the {sect_kind:?} section, caused by {e}")]
pub struct ExtractError {
    #[source]
    e: dict_sect_pfc::ExtractError,
    id: Id,
    id_kind: IdKind,
    sect_kind: SectKind,
}

#[derive(Error, Debug)]
#[error("four sect dict section error in the {sect_kind:?} section")]
pub struct DictSectError {
    #[source]
    e: dict_sect_pfc::Error,
    sect_kind: SectKind,
}

#[derive(Error, Debug)]
pub enum Error {
    #[error("failed to read FourSectDict control info")]
    ControlInfo(#[from] crate::containers::control_info::Error),
    #[error("failed to read FourSectDict section")]
    DictSect(#[from] DictSectError),
    #[error("failed to read FourSectDict: {0}")]
    Other(String),
}

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct EncodedTripleId {
    pub subject: Id,
    pub predicate: Id,
    pub object: Id,
}

/// Generic implementation for all dictionary section types
impl<D: dict_sect_pfc::DictSectPfcAccess> FourSectDictGeneric<D> {
    /// Get the string value of a given ID of a given type.
    /// String representation of URIs, literals and blank nodes is defined in <https://www.w3.org/Submission/2011/SUBM-HDT-20110330/#dictionaryEncoding>>..
    /// Resolve an ID and kind to the section holding it, the section-local ID, and the section
    /// tag used in error messages. Shared by `id_to_string`, `id_to_arc` and `id_into_buf`.
    #[inline]
    fn section_for(&self, id: Id, id_kind: IdKind) -> (&D, Id, SectKind) {
        use SectKind::*;
        let shared_size = self.shared.num_strings() as Id;
        match id_kind {
            IdKind::Predicate => (&self.predicates, id, Predicate),
            IdKind::Subject if id > shared_size => (&self.subjects, id - shared_size, Subject),
            IdKind::Object if id > shared_size => (&self.objects, id - shared_size, Object),
            // subjects and objects with id <= shared_size live in the shared section
            _ => (&self.shared, id, Shared),
        }
    }

    /// Deliberately dispatches per section instead of going through [`Self::section_for`]: the
    /// merged call site costs ~3% on `subjects_with_po`, because the section pointer becomes a
    /// runtime select and stops being hoistable out of the caller's loop. The buffer-based
    /// extractors below don't show this, so they keep sharing `section_for`.
    pub fn id_to_string(&self, id: Id, id_kind: IdKind) -> core::result::Result<String, ExtractError> {
        use SectKind::*;
        let shared_size = self.shared.num_strings() as Id;
        let d = id.saturating_sub(shared_size);
        match id_kind {
            IdKind::Subject => {
                if id <= shared_size {
                    self.shared.extract(id).map_err(|e| ExtractError { e, id, id_kind, sect_kind: Shared })
                } else {
                    self.subjects.extract(d).map_err(|e| ExtractError { e, id, id_kind, sect_kind: Subject })
                }
            }
            IdKind::Predicate => {
                self.predicates.extract(id).map_err(|e| ExtractError { e, id, id_kind, sect_kind: Predicate })
            }
            IdKind::Object => {
                if id <= shared_size {
                    self.shared.extract(id).map_err(|e| ExtractError { e, id, id_kind, sect_kind: Shared })
                } else {
                    self.objects.extract(d).map_err(|e| ExtractError { e, id, id_kind, sect_kind: Object })
                }
            }
        }
    }

    /// Like [`id_to_string`](Self::id_to_string) but decodes directly into an `Arc<str>`, reusing
    /// `buf` as decompression scratch. Building the `Arc` in a single allocation (instead of
    /// `id_to_string(id, kind)?.into()`, which allocates a `Vec`, then a `String`, then the `Arc`)
    /// is the hot-path saver for `triples_with_pattern` result materialization.
    pub fn id_to_arc(
        &self, id: Id, id_kind: IdKind, buf: &mut Vec<u8>,
    ) -> core::result::Result<Arc<str>, ExtractError> {
        let (sect, sid, sect_kind) = self.section_for(id, id_kind);
        sect.extract_arc(sid, buf).map_err(|e| ExtractError { e, id, id_kind, sect_kind })
    }

    /// Like [`id_to_arc`](Self::id_to_arc) but leaves the decoded bytes in `buf` (cleared first)
    /// instead of allocating a return value. Used by the Sophia `Graph` adapter, which parses the
    /// bytes straight into a term and so never needs an intermediate `Arc`.
    pub(crate) fn id_into_buf(
        &self, id: Id, id_kind: IdKind, buf: &mut Vec<u8>,
    ) -> core::result::Result<(), ExtractError> {
        let (sect, sid, sect_kind) = self.section_for(id, id_kind);
        sect.extract_into(sid, buf).map_err(|e| ExtractError { e, id, id_kind, sect_kind })
    }

    /// Get the ID for a given string or 0 if not found.
    /// String representation of URIs, literals and blank nodes is defined in <https://www.w3.org/Submission/2011/SUBM-HDT-20110330/#dictionaryEncoding>>..
    pub fn string_to_id(&self, s: &str, id_kind: IdKind) -> Id {
        let shared_size = self.shared.num_strings();
        match id_kind {
            IdKind::Subject => {
                let mut id = self.shared.string_to_id(s);
                if id == 0 {
                    id = self.subjects.string_to_id(s);
                    if id > 0 {
                        id += shared_size as Id;
                    }
                }
                id
            }
            IdKind::Predicate => self.predicates.string_to_id(s),
            IdKind::Object => {
                let mut id = self.shared.string_to_id(s);
                if id == 0 {
                    id = self.objects.string_to_id(s);
                    if id > 0 {
                        id += shared_size as Id;
                    }
                }
                id
            }
        }
    }

    /// size in bytes of the in memory four section dictionary
    pub fn size_in_bytes(&self) -> usize {
        self.shared.size_in_bytes()
            + self.subjects.size_in_bytes()
            + self.predicates.size_in_bytes()
            + self.objects.size_in_bytes()
    }
}

/// Implementation specific to in-memory FourSectDict
impl FourSectDict {
    /// read the whole dictionary section including control information
    pub fn read<R: BufRead>(reader: &mut R) -> Result<UnvalidatedFourSectDict> {
        use SectKind::*;
        let dict_ci = ControlInfo::read(reader)?;
        if dict_ci.format != "<http://purl.org/HDT/hdt#dictionaryFour>" {
            return Err(Error::Other("Implementation only supports four section dictionaries".to_owned()));
        }
        let mut f = |sect_kind| DictSectPFC::read(reader).map_err(|e| DictSectError { e, sect_kind });
        Ok(UnvalidatedFourSectDict([f(Shared)?, f(Subject)?, f(Predicate)?, f(Object)?]))
    }

    /// write the whole Dictionary including control info and all sections
    pub fn write(&self, write: &mut impl std::io::Write) -> Result<()> {
        use SectKind::*;
        ControlInfo::four_sect_dict().write(write)?;
        self.shared.write(write).map_err(|e| DictSectError { e, sect_kind: Shared })?;
        self.subjects.write(write).map_err(|e| DictSectError { e, sect_kind: Subject })?;
        self.predicates.write(write).map_err(|e| DictSectError { e, sect_kind: Predicate })?;
        self.objects.write(write).map_err(|e| DictSectError { e, sect_kind: Object })?;
        Ok(())
    }

    /// Build a memory-mapped four-section dictionary from a shared mmap
    /// and the four section offsets recorded in the cache.
    #[cfg(not(any(target_arch = "wasm32", target_arch = "wasm64")))]
    pub fn from_mmap(
        mmap: Arc<memmap2::Mmap>, shared_offset: u64, subject_offset: u64, predicate_offset: u64,
        object_offset: u64,
    ) -> Result<FourSectDictMmap> {
        use crate::four_sect_dict::SectKind::*;
        use dict_sect_pfc::MmapDictSectPfc;
        Ok(FourSectDictGeneric {
            shared: MmapDictSectPfc::from_mmap(Arc::clone(&mmap), shared_offset)
                .map_err(|e| DictSectError { e, sect_kind: Shared })?,
            subjects: MmapDictSectPfc::from_mmap(Arc::clone(&mmap), subject_offset)
                .map_err(|e| DictSectError { e, sect_kind: Subject })?,
            predicates: MmapDictSectPfc::from_mmap(Arc::clone(&mmap), predicate_offset)
                .map_err(|e| DictSectError { e, sect_kind: Predicate })?,
            objects: MmapDictSectPfc::from_mmap(mmap, object_offset)
                .map_err(|e| DictSectError { e, sect_kind: Object })?,
        })
    }
}

/// A wrapper to ensure prevent using FourSectDict before its checksums have been validated
#[cfg(not(any(target_arch = "wasm32", target_arch = "wasm64")))]
pub struct UnvalidatedFourSectDict([JoinHandle<dict_sect_pfc::Result<DictSectPFC>>; 4]);

/// WASM version without JoinHandle
#[cfg(any(target_arch = "wasm32", target_arch = "wasm64"))]
pub struct UnvalidatedFourSectDict([DictSectPFC; 4]);

impl UnvalidatedFourSectDict {
    /// Validates the checksums of all dictionary sections in parallel.
    /// Dict validation takes around 1200 ms on a single thread with an 1.5 GB HDT file on an i9-12900k.
    #[cfg(not(any(target_arch = "wasm32", target_arch = "wasm64")))]
    pub fn validate(self) -> Result<FourSectDict> {
        use SectKind::*;
        let r: Vec<_> = [Shared, Subject, Predicate, Object]
            .into_iter()
            .zip(self.0)
            .map(|(sect_kind, handle)| handle.join().unwrap().map_err(|e| DictSectError { e, sect_kind }))
            .collect::<std::result::Result<Vec<DictSectPFC>, DictSectError>>()?;
        let [shared, subjects, predicates, objects]: [DictSectPFC; 4] = r.try_into().unwrap();
        Ok(FourSectDict { shared, subjects, predicates, objects })
    }

    /// WASM version - sections are already validated during read
    #[cfg(any(target_arch = "wasm32", target_arch = "wasm64"))]
    pub fn validate(self) -> Result<FourSectDict> {
        let [shared, subjects, predicates, objects] = self.0;
        Ok(FourSectDict { shared, subjects, predicates, objects })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::header::Header;
    use crate::tests::init;
    use fs_err::File;
    use pretty_assertions::assert_eq;
    use std::io::BufReader;

    #[test]
    fn read_write_dict() -> color_eyre::Result<()> {
        init();
        let file = File::open("tests/resources/snikmeta.hdt")?;
        let mut reader = BufReader::new(file);
        ControlInfo::read(&mut reader)?;
        Header::read(&mut reader)?;

        let dict = FourSectDict::read(&mut reader)?.validate()?;
        assert_eq!(dict.shared.num_strings(), 43, "wrong number of strings in the shared section");
        assert_eq!(dict.subjects.num_strings(), 6, "wrong number of strings in the subject section");
        assert_eq!(dict.predicates.num_strings(), 23, "wrong number of strings in the predicates section");
        assert_eq!(dict.objects.num_strings(), 133, "wrong number of strings in the objects section");
        assert_eq!(dict.string_to_id("_:b1", IdKind::Subject), 1);
        assert_eq!("http://www.snik.eu/ontology/meta/uses", dict.id_to_string(43, IdKind::Subject)?);
        assert_eq!("http://www.snik.eu/ontology/meta/Chapter", dict.id_to_string(3, IdKind::Subject)?);
        assert_eq!("http://www.snik.eu/ontology/meta/DataSetType", dict.id_to_string(5, IdKind::Subject)?);
        for id in 1..dict.shared.num_strings() {
            let s = dict.id_to_string(id, IdKind::Subject)?;
            let back = dict.string_to_id(&s, IdKind::Subject);
            assert_eq!(id, back, "shared id {} -> subject {} -> id {}", id, s, back);

            let s = dict.id_to_string(id, IdKind::Object)?;
            let back = dict.string_to_id(&s, IdKind::Object);
            assert_eq!(id, back, "shared id {} -> object {} -> id {}", id, s, back);
        }
        for (sect, kind, name, offset) in [
            (&dict.subjects, IdKind::Subject, "subject", dict.shared.num_strings()),
            (&dict.objects, IdKind::Object, "object", dict.shared.num_strings()),
            (&dict.predicates, IdKind::Predicate, "predicate", 0),
        ] {
            for id in offset + 1..offset + sect.num_strings() {
                let s = dict.id_to_string(id, kind)?;
                let back = dict.string_to_id(&s, kind);
                assert_eq!(id, back, "{} id {} -> {} {} -> id {}", name, id, name, s, back);
            }
        }
        let mut buf = Vec::new();
        dict.write(&mut buf)?;
        let dict2 = FourSectDict::read(&mut std::io::Cursor::new(buf))?.validate()?;
        assert_eq!(dict, dict2);
        Ok(())
    }
}
