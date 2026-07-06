// Copyright (c) 2026, Decisym, LLC
// Licensed under the BSD 3-Clause License (see LICENSE file in the project root).
//
//! Polymorphic HDT handle that picks the best backend for a given file.
//!
//! [`HdtAny`] holds either an [`HdtHybrid`] (mmap + persisted op-index cache)
//! or a fully in-memory [`Hdt`]. The hybrid form is preferred because it
//! shares mapped pages across processes and avoids reparsing dictionaries on
//! every load, but it cannot be built for every HDT file — most notably,
//! files that contain zero triples cannot have a wavelet-tree based op-index
//! (the underlying QWT library panics on empty data). For those cases
//! [`HdtAny::open`] transparently falls back to the in-memory variant so the
//! caller doesn't have to handle the asymmetry.
//!
//! Prototype scope: this exposes the small subset of the [`HdtGeneric`] query
//! surface that consumers like the `de` SPARQL evaluator need today — header
//! and size accessors, dictionary lookups, and the `triples_*_with_pattern`
//! family. It is deliberately *not* a full mirror of `HdtGeneric`; if a
//! caller needs lower-level access to `dict` / `triples` they should match on
//! the variant directly.
//!
//! [`HdtGeneric`]: crate::HdtGeneric

use std::path::Path;
use std::sync::Arc;

use crate::four_sect_dict::{ExtractError, IdKind};
use crate::hdt::{Error, Hdt, HdtHybrid};
use crate::header::Header;
use crate::triples::hybrid_cache::HybridCacheError;
use crate::triples::{Id, TripleId};

type StringTriple = [Arc<str>; 3];

/// Default file-size threshold (in bytes) below which
/// [`HdtAny::open_with_threshold`] skips the hybrid backend and loads the
/// HDT fully in memory. 1 MiB is well above the size at which cache-build
/// cost outweighs query speedup, while leaving a comfortable margin so that
/// realistic working datasets still pick up the hybrid backend.
pub const DEFAULT_SMALL_FILE_THRESHOLD: usize = 1 << 20;

/// Either flavor of an HDT graph loaded from a file.
///
/// Construct with [`HdtAny::open`]. The variant chosen depends on whether
/// the file admits a hybrid cache; callers that don't care about the backend
/// should treat this type as opaque and use the forwarded query methods.
#[derive(Debug)]
pub enum HdtAny {
    /// Hybrid (mmap + cache) representation. Preferred for non-empty files.
    Hybrid(HdtHybrid),
    /// Fully in-memory representation. Used as a fallback for files where
    /// the hybrid cache cannot be created (e.g. zero-triple HDTs).
    InMemory(Hdt),
}

impl HdtAny {
    /// Load an HDT from `path`, choosing the backend by file size.
    ///
    /// Files smaller than `threshold` bytes are loaded fully in memory via
    /// [`Hdt::read`]; the hybrid cache machinery is bypassed entirely so no
    /// sidecar cache file is created. Files at or above the threshold try
    /// the hybrid backend first and fall back to an in-memory load only if
    /// the hybrid path fails with a recoverable error (currently:
    /// [`HybridCacheError::EmptyHdt`]). All other errors propagate.
    ///
    /// `threshold = None` uses [`DEFAULT_SMALL_FILE_THRESHOLD`] (1 MiB).
    /// Pass `Some(0)` to force the hybrid path for any non-empty file, or
    /// `Some(usize::MAX)` to force in-memory.
    ///
    /// File-size lookup uses [`std::fs::metadata`]; if the metadata call
    /// fails (e.g. transient FS error) the function defers to the hybrid
    /// path so that the subsequent open surfaces the real error rather than
    /// a misleading routing decision.
    pub fn open_with_threshold(path: &Path, threshold: Option<usize>) -> Result<Self, Error> {
        let threshold = threshold.unwrap_or(DEFAULT_SMALL_FILE_THRESHOLD);

        let size_below_threshold = std::fs::metadata(path)
            .is_ok_and(|m| usize::try_from(m.len()).unwrap_or(usize::MAX) < threshold);

        if size_below_threshold {
            log::debug!(
                "HdtAny::open_with_threshold using in-memory backend for {} (size below {threshold} byte threshold)",
                path.display(),
            );
            return Ok(Self::InMemory(Self::read_in_memory(path)?));
        }

        match Hdt::new_hybrid_cache(path) {
            Ok(h) => Ok(Self::Hybrid(h)),
            Err(e) if Self::is_hybrid_recoverable(&e) => {
                log::warn!(
                    "hybrid HDT load not viable for {}, falling back to in-memory: {e}",
                    path.display()
                );
                Ok(Self::InMemory(Self::read_in_memory(path)?))
            }
            Err(e) => Err(e),
        }
    }

    /// In-memory load. Uses [`Hdt::read_from_path`] which itself falls back
    /// to [`Hdt::read`] when the cache machinery cannot be used (e.g. empty
    /// HDTs), so this single helper covers both the small-file shortcut and
    /// the recoverable-error fallback.
    fn read_in_memory(path: &Path) -> Result<Hdt, Error> {
        Hdt::read_from_path(path)
    }

    const fn is_hybrid_recoverable(e: &Error) -> bool {
        matches!(e, Error::Cache(HybridCacheError::EmptyHdt))
    }

    /// True if this handle is using the hybrid (mmap+cache) backend.
    pub const fn is_hybrid(&self) -> bool {
        matches!(self, Self::Hybrid(_))
    }

    pub const fn header(&self) -> &Header {
        match self {
            Self::Hybrid(h) => h.header(),
            Self::InMemory(h) => h.header(),
        }
    }

    pub fn size_in_bytes(&self) -> usize {
        match self {
            Self::Hybrid(h) => h.size_in_bytes(),
            Self::InMemory(h) => h.size_in_bytes(),
        }
    }

    /// Resolve a string term to its dictionary id; returns `0` when absent.
    pub fn string_to_id(&self, s: &str, kind: IdKind) -> Id {
        match self {
            Self::Hybrid(h) => h.dict.string_to_id(s, kind),
            Self::InMemory(h) => h.dict.string_to_id(s, kind),
        }
    }

    /// Resolve a dictionary id back to a string term.
    pub fn id_to_string(&self, id: Id, kind: IdKind) -> Result<String, ExtractError> {
        match self {
            Self::Hybrid(h) => h.dict.id_to_string(id, kind),
            Self::InMemory(h) => h.dict.id_to_string(id, kind),
        }
    }

    pub fn triples_all(&self) -> Box<dyn Iterator<Item = StringTriple> + '_> {
        match self {
            Self::Hybrid(h) => Box::new(h.triples_all()),
            Self::InMemory(h) => Box::new(h.triples_all()),
        }
    }

    pub fn triples_with_pattern<'a>(
        &'a self, sp: Option<&'a str>, pp: Option<&'a str>, op: Option<&'a str>,
    ) -> Box<dyn Iterator<Item = StringTriple> + 'a> {
        match self {
            Self::Hybrid(h) => h.triples_with_pattern(sp, pp, op),
            Self::InMemory(h) => h.triples_with_pattern(sp, pp, op),
        }
    }

    pub fn triple_ids_with_pattern<'a>(
        &'a self, sp: Option<&'a str>, pp: Option<&'a str>, op: Option<&'a str>,
    ) -> Box<dyn Iterator<Item = TripleId> + 'a> {
        match self {
            Self::Hybrid(h) => h.triple_ids_with_pattern(sp, pp, op),
            Self::InMemory(h) => h.triple_ids_with_pattern(sp, pp, op),
        }
    }

    pub fn triple_ids_with_id_pattern(
        &self, pattern: TripleId,
    ) -> Box<dyn Iterator<Item = TripleId> + '_> {
        match self {
            Self::Hybrid(h) => h.triple_ids_with_id_pattern(pattern),
            Self::InMemory(h) => h.triple_ids_with_id_pattern(pattern),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn init() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    #[test]
    fn forced_hybrid_threshold_picks_hybrid_for_non_empty() -> Result<(), Error> {
        init();
        // threshold=Some(0) forces the hybrid path for any non-empty file,
        // independently of the small-file default.
        let any = HdtAny::open_with_threshold(Path::new("tests/resources/snikmeta.hdt"), Some(0))?;
        assert!(any.is_hybrid(), "forced threshold should select hybrid");
        // Sanity: query surface forwards.
        assert_eq!(any.triples_with_pattern(None, None, None).count(), 328);
        assert_ne!(0, any.string_to_id("http://www.snik.eu/ontology/meta", IdKind::Subject));
        Ok(())
    }

    #[test]
    fn default_threshold_picks_in_memory_for_small_file() -> Result<(), Error> {
        init();
        // snikmeta.hdt is ~10 KiB, well below the 1 MiB default threshold,
        // so the default policy should route it to the in-memory backend.
        let any = HdtAny::open_with_threshold(Path::new("tests/resources/snikmeta.hdt"), None)?;
        assert!(!any.is_hybrid(), "small file under default threshold should use in-memory");
        assert_eq!(any.triples_with_pattern(None, None, None).count(), 328);
        Ok(())
    }

    #[test]
    fn empty_hdt_loads_via_in_memory() -> Result<(), Error> {
        init();
        // Even with the threshold forced to 0 (which would prefer hybrid),
        // empty HDTs hit the EmptyHdt fallback and end up in-memory.
        let any = HdtAny::open_with_threshold(Path::new("tests/resources/empty.hdt"), Some(0))?;
        assert!(!any.is_hybrid(), "empty HDT should fall back to in-memory");
        assert_eq!(any.triples_all().count(), 0);
        assert_eq!(any.triples_with_pattern(None, None, None).count(), 0);
        Ok(())
    }
}
