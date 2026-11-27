//! Readers of an import while it is republished: every lookup finds a
//! whole version, the one before or the one after, never none, and every
//! inspection one version, or that the directory kept changing.

use std::{
    fs,
    sync::atomic::{AtomicBool, Ordering},
    time::Duration,
};

use super::{
    import,
    republish::{Scratch, exchanges, shard},
};
use crate::{
    error::ModelsError,
    gguf::{import::publish::Exchange, tests::TestResult},
    layout::ModelLayout,
};

/// How a readers' test republishes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Publishing {
    /// With a pause before each publish, as any publisher leaves, in which
    /// an inspection that met one reads again in time.
    Paused,
    /// Back to back, which can change the directory under every read of an
    /// inspection.
    Nonstop,
}

impl Publishing {
    fn pause(self) -> Duration {
        match self {
            Self::Paused => Duration::from_millis(1),
            Self::Nonstop => Duration::ZERO,
        }
    }

    /// How many times, for about half a second to a second.
    const fn publishes(self) -> u32 {
        match self {
            Self::Paused => 300,
            Self::Nonstop => 3000,
        }
    }
}

/// Whether `layout` is a whole version as published: every derived file
/// and exactly one of the shards, never a mix of two versions.
fn whole(layout: &ModelLayout) -> bool {
    layout.tokenizer_path.is_some()
        && layout.tokenizer_config_path.is_some()
        && layout.generation_config_path.is_some()
        && layout.weights.len() == 1
}

/// Reads with `read` until `done`: how many times, and the failures.
fn reading(done: &AtomicBool, read: impl Fn() -> Result<(), String>) -> (u32, Vec<String>) {
    let (mut reads, mut failures) = (0_u32, Vec::new());
    while !done.load(Ordering::Acquire) {
        reads += 1;
        if let Err(failure) = read() {
            failures.push(failure);
        }
    }
    (reads, failures)
}

/// Republishes as `publishing` tells, switching between two shards, while
/// one thread looks up the destination's `config.json` and another
/// inspects it. Every publish exchanges; every lookup finds the file, and
/// every inspection a whole version, or, published nonstop, that the
/// directory kept changing.
fn republished_while_read(label: &str, publishing: Publishing) -> TestResult {
    let changing = publishing == Publishing::Nonstop;
    let root = Scratch::new(label)?;
    let shards = [shard(&root, "a.gguf")?, shard(&root, "b.gguf")?];
    let output = root.join("imports/model");
    import(vec![shards[0].clone()], 0).publish(&output)?;
    let (done, config) = (AtomicBool::new(false), output.join("config.json"));
    let (published, readers) = std::thread::scope(|scope| {
        let readers = [
            scope.spawn(|| {
                reading(&done, || {
                    fs::metadata(&config).map(drop).map_err(|error| format!("config.json: {error}"))
                })
            }),
            scope.spawn(|| {
                reading(&done, || match ModelLayout::inspect(&output) {
                    Ok(layout) if whole(&layout) => Ok(()),
                    Err(ModelsError::ChangedWhileRead(_)) if changing => Ok(()),
                    Ok(layout) => Err(format!("a partial layout: {layout:?}")),
                    Err(error) => Err(error.to_string()),
                })
            }),
        ];
        let published = shards.iter().cycle().zip(1..=publishing.publishes()).try_for_each(
            |(shard, version)| {
                std::thread::sleep(publishing.pause());
                import(vec![shard.clone()], version).publish(&output)
            },
        );
        done.store(true, Ordering::Release);
        (published, readers.map(std::thread::ScopedJoinHandle::join))
    });
    published?;
    let exchanged = exchanges();
    assert!(
        exchanged.iter().all(|exchange| *exchange == Exchange::Done),
        "the temporary directory's file system refused an exchange: {exchanged:?}"
    );
    assert_eq!(exchanged.len(), usize::try_from(publishing.publishes())?);
    for reader in readers {
        let (reads, failures) = reader.map_err(|_| "a reader panicked")?;
        assert!(
            failures.is_empty(),
            "{} of {reads} reads failed, first: {}",
            failures.len(),
            failures.first().map_or("", String::as_str),
        );
    }
    Ok(())
}

/// Paused between publishes, every inspection finds a whole version.
#[test]
fn readers_find_a_whole_import_while_it_is_republished() -> TestResult {
    republished_while_read("readers", Publishing::Paused)
}

/// Published nonstop, an inspection finds a whole version or says the
/// directory kept changing, never giving a partial layout.
#[test]
fn readers_never_find_a_partial_import_while_it_is_republished_nonstop() -> TestResult {
    republished_while_read("readers-nonstop", Publishing::Nonstop)
}
