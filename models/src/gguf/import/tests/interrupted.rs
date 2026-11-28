//! Writes cut short: the import reads stale and no loader reads it, and
//! every file an import made stays recorded, through further writes cut
//! short too, until a write completes.

use std::{fs, path::Path};

use serde_json::json;

use super::{ImportManifest, import, root};
use crate::{
    error::{ModelsError, Result},
    gguf::{
        MANIFEST, files_in,
        import::files::{CUT_SHORT, Step},
        tests::TestResult,
    },
    layout::ModelLayout,
};

/// `name` with `bytes` in its own folder of `root`.
fn shard(root: &Path, name: &str, bytes: &[u8]) -> TestResult<std::path::PathBuf> {
    let folder = root.join(name.trim_end_matches(".gguf"));
    fs::create_dir_all(&folder)?;
    fs::write(folder.join(name), bytes)?;
    Ok(folder.join(name))
}

/// Runs `write` cut short after `step`.
fn cut_short(step: Step, write: impl FnOnce() -> Result<()>) {
    CUT_SHORT.set(Some(step));
    let written = write();
    CUT_SHORT.set(None);
    assert!(written.is_err(), "the write was not cut short");
}

fn current(directory: &Path) -> bool {
    ImportManifest::read(directory).is_some_and(|manifest| manifest.is_current(directory))
}

/// A complete manifest beside a manifest a later write left aside: the
/// import reads stale, loaders refuse it, and the next write completes it.
#[test]
fn a_manifest_left_aside_makes_the_import_stale() -> TestResult {
    let root = root("left-aside")?;
    let first = shard(&root, "first.gguf", b"first")?;
    let output = root.join("output");
    import(vec![first.clone()], 1).write(&output)?;
    fs::copy(output.join(MANIFEST), output.join(".import.json.partial"))?;
    assert!(!current(&output));
    assert!(matches!(files_in(&output), Err(ModelsError::InterruptedImport(_))));
    assert!(matches!(ModelLayout::inspect(&output), Err(ModelsError::InterruptedImport(_))));
    import(vec![first], 1).write(&output)?;
    assert!(current(&output) && !output.join(".import.json.partial").exists());
    fs::remove_dir_all(root)?;
    Ok(())
}

/// A write cut short with its files in place, then another cut short with
/// only its manifest aside: the next write still finds every file both
/// made, and leaves only its own.
#[test]
fn a_write_cut_short_twice_leaves_nothing_unrecorded() -> TestResult {
    let root = root("twice")?;
    let shards =
        ["x.gguf", "y.gguf", "z.gguf", "w.gguf"].map(|name| shard(&root, name, name.as_bytes()));
    let [x, y, z, w] = shards;
    let output = root.join("output");
    import(vec![x?], 1).write(&output)?;
    let y = y?;
    cut_short(Step::Moved, || import(vec![y], 1).write(&output));
    assert!(output.join("y.gguf").symlink_metadata().is_ok());
    let z = z?;
    cut_short(Step::Aside, || import(vec![z], 1).write(&output));
    import(vec![w?], 1).write(&output)?;
    for gone in ["x.gguf", "y.gguf", "z.gguf", ".z.gguf.partial"] {
        assert!(output.join(gone).symlink_metadata().is_err(), "{gone} stayed");
    }
    assert_eq!(files_in(&output)?, [output.join("w.gguf")]);
    assert!(current(&output));
    fs::remove_dir_all(root)?;
    Ok(())
}

/// A vision write cut short with its processor configuration in place,
/// then a text write cut short before removing it: a later vision import
/// replaces that configuration as the import's own.
#[test]
fn a_processor_configuration_of_a_write_cut_short_is_replaced() -> TestResult {
    let root = root("twice-vision")?;
    let names = ["a.gguf", "b.gguf", "c.gguf", "d.gguf", "mmproj.gguf"];
    let [a, b, c, d, mmproj] = names.map(|name| shard(&root, name, name.as_bytes()));
    let (mmproj, output) = (mmproj?, root.join("output"));
    let vision = |text, size| {
        let mut vision = import(vec![text], 1);
        vision.vision_shards = vec![mmproj.clone()];
        vision.preprocessor_config = Some(json!({ "size": size }));
        vision
    };
    import(vec![a?], 1).write(&output)?;
    let b = b?;
    cut_short(Step::Moved, || vision(b, 1).write(&output));
    let c = c?;
    cut_short(Step::Aside, || import(vec![c], 1).write(&output));
    vision(d?, 2).write(&output)?;
    let written = fs::read_to_string(output.join("preprocessor_config.json"))?;
    assert!(written.contains('2'), "{written}");
    assert!(current(&output));
    fs::remove_dir_all(root)?;
    Ok(())
}
