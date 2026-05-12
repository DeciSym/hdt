use crate::containers::ControlInfo;
use crate::containers::rdf::{Id, Literal, Term, Triple};
use oxrdf::{NamedOrBlankNode, Term as OxTerm};
use oxttl::NTriplesParser;
use std::collections::BTreeSet;
use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;
use std::path::Path;

pub type Result<T> = core::result::Result<T, Error>;

/// Metadata about the dataset, see <https://www.rdfhdt.org/hdt-binary-format/#header>.
#[derive(Debug, Clone)]
pub struct Header {
    /// Header data format. Only "ntriples" is supported.
    pub format: String,
    /// The number of bytes of the header data.
    pub length: usize,
    /// Triples describing the dataset.
    pub body: BTreeSet<Triple>,
}

/// The error type for the `read` method.
#[derive(thiserror::Error, Debug)]
#[error("failed to read HDT header")]
pub enum Error {
    #[error("{0}")]
    Other(String),
    Io(#[from] std::io::Error),
    ControlInfo(#[from] crate::containers::control_info::Error),
    #[error("invalid header format {0}, only 'ntriples' is supported")]
    InvalidHeaderFormat(String),
    #[error("invalid header length '{0}'")]
    InvalidHeaderLength(String),
    #[error("missing header length")]
    MissingHeaderLength,
}

impl Header {
    /// Reads the header section directly from an HDT file path.
    ///
    /// This reads and validates the leading global control info chunk, then
    /// parses the header section.
    pub fn read_from_hdt_path(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        ControlInfo::read(&mut reader)?;
        Self::read(&mut reader)
    }

    /// Reader needs to be positioned directly after the global control information.
    pub fn read<R: BufRead>(reader: &mut R) -> Result<Self> {
        let header_ci = ControlInfo::read(reader)?;
        if header_ci.format != "ntriples" {
            return Err(Error::InvalidHeaderFormat(header_ci.format));
        }

        let ls = header_ci.get("length").ok_or(Error::MissingHeaderLength)?;
        let length = ls.parse::<usize>().map_err(|_| Error::InvalidHeaderLength(ls))?;

        let mut body_buffer: Vec<u8> = vec![0; length];
        reader.read_exact(&mut body_buffer)?;
        let mut body = BTreeSet::new();

        for parsed in NTriplesParser::new().for_slice(&body_buffer) {
            let Ok(triple) = parsed else { continue };

            let subject = match triple.subject {
                NamedOrBlankNode::NamedNode(iri) => Id::Named(iri.into_string()),
                NamedOrBlankNode::BlankNode(id) => Id::Blank(id.into_string()),
            };

            let predicate = triple.predicate.into_string();

            let object = match triple.object {
                OxTerm::NamedNode(iri) => Term::Id(Id::Named(iri.into_string())),
                OxTerm::BlankNode(id) => Term::Id(Id::Blank(id.into_string())),
                OxTerm::Literal(lit) => {
                    // oxrdf normalizes "..."^^xsd:string to a plain literal, so the
                    // destructured datatype is None exactly when the literal had no
                    // explicit non-string datatype.
                    let (form, datatype, lang) = lit.destruct();
                    Term::Literal(match (datatype, lang) {
                        (_, Some(lan)) => Literal::new_lang(form, lan),
                        (Some(dt), None) => Literal::new_typed(form, dt.into_string()),
                        (None, None) => Literal::new(form),
                    })
                }
            };

            body.insert(Triple::new(subject, predicate, object));
        }
        Ok(Header { format: header_ci.format, length, body })
    }

    pub fn write(&self, write: &mut impl std::io::Write) -> Result<()> {
        ControlInfo::header(self.length).write(write)?;
        for triple in &self.body {
            writeln!(write, "{triple}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::init;

    #[test]
    fn read_header() -> color_eyre::Result<()> {
        init();
        let header = Header::read_from_hdt_path(Path::new("tests/resources/yago_header.hdt"))?;
        assert_eq!(header.format, "ntriples");
        assert_eq!(header.length, 1891);
        assert_eq!(header.body.len(), 22);
        Ok(())
    }
}
