//! Rows a backend must refuse before it indexes anything with them.

use libmir::decision::{DecisionCheckpoint, DecisionModel, DecisionRow};

use super::{BACKENDS, Failure, reference, rows};

/// Each malformed variant of `base`, named.
fn malformed(
    base: &DecisionRow,
    vocab: usize,
    max_len: usize,
) -> Result<Vec<(&'static str, DecisionRow)>, Failure> {
    let with = |change: &dyn Fn(&mut DecisionRow)| {
        let mut row = base.clone();
        change(&mut row);
        row
    };
    let vocab = u32::try_from(vocab)?;
    Ok(vec![
        (
            "empty row",
            with(&|row| {
                row.tokens.clear();
                row.markers.clear();
            }),
        ),
        ("token == vocab", with(&|row| row.tokens[1] = vocab)),
        ("token u32::MAX", with(&|row| row.tokens[1] = u32::MAX)),
        ("marker == len", with(&|row| row.markers.push(row.tokens.len()))),
        ("marker usize::MAX", with(&|row| row.markers.push(usize::MAX))),
        ("len max_len + 1", with(&|row| row.tokens.resize(max_len + 1, row.tokens[0]))),
    ])
}

#[test]
#[ignore = "needs the Laya checkpoint in LIBMIR_LAYA_MODEL; run `make laya-parity`"]
fn every_backend_rejects_malformed_rows_and_keeps_answering() -> Result<(), Failure> {
    let root = std::env::var_os("LIBMIR_LAYA_MODEL")
        .ok_or("LIBMIR_LAYA_MODEL must name the Laya checkpoint directory")?;
    let checkpoint = DecisionCheckpoint::inspect(&root)?;
    let reference = reference()?;
    for &(backend, _) in BACKENDS {
        let model = DecisionModel::load(&root, backend)?;
        let (_, built) = rows(&model, &reference)?.swap_remove(0);
        let base = built[0].clone();
        let expected = model.logits(std::slice::from_ref(&base))?;
        for (why, row) in malformed(&base, checkpoint.encoder.vocab_size, checkpoint.agent.max_len)?
        {
            let result = model.logits(&[base.clone(), row]);
            assert!(result.is_err(), "{backend:?}: {why} was accepted");
        }
        assert_eq!(model.logits(std::slice::from_ref(&base))?, expected, "{backend:?}");
    }
    Ok(())
}
