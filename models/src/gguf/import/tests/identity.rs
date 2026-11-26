//! A file's identity as the import tells its own hard links on every
//! system: a hard link is the source it links, a copy is not, nor is the
//! source at its own path the import's link to it.

use std::fs;

use super::{super::identity, root};
use crate::gguf::{SourceFile, tests::TestResult};

#[test]
fn a_hard_link_is_its_source_and_a_copy_is_not() -> TestResult {
    let root = root("hard-link")?;
    let source = root.join("model.gguf");
    fs::write(&source, b"shard bytes")?;
    let source = fs::canonicalize(&source)?;
    let output = root.join("output");
    fs::create_dir_all(&output)?;
    let (link, copy) = (output.join("model.gguf"), output.join("copy.gguf"));
    fs::hard_link(&source, &link)?;
    let _bytes = fs::copy(&source, &copy)?;

    assert!(identity::same_file(&source, &link));
    assert!(!identity::same_file(&source, &copy));
    assert!(!identity::same_file(&source, &output.join("missing.gguf")));
    let recorded = SourceFile::of(&source)?;
    assert!(recorded.is_linked_at(&link));
    assert!(!recorded.is_linked_at(&copy));
    assert!(!recorded.is_linked_at(&source));
    fs::remove_dir_all(root)?;
    Ok(())
}
