use std::io::Write;

use super::*;

#[derive(Clone, Copy, Debug)]
enum Retirement {
    Release,
    Retain,
}

#[test]
#[ignore = "real-model cohort shrink control with inactive rows retained or released"]
fn isolates_retirement_from_scalar_packed_arithmetic()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    let mut model = fixture::load_path(std::env::var("MIRMIR_BENCH_MODEL")?, 10)?;
    let retained = run(&mut model, Retirement::Retain)?;
    let released = run(&mut model, Retirement::Release)?;
    assert_eq!(retained, released, "retiring inactive rows changes the survivor trajectory");
    Ok(())
}

fn run(model: &mut LoadedModel, retirement: Retirement) -> Result<(Vec<u32>, Vec<u32>)> {
    let mut rows = wave(model, &[0, 1, 2, 3, 4], false)?;
    model.flush_decode_graphs()?;
    let expected = reference(model, rows[0])?;
    let mut observed = vec![rows[0].token];
    let mut inactive = Vec::new();
    for step in 0..16 {
        if step == 8 {
            model.flush_decode_graphs()?;
            inactive.extend(rows.drain(1..));
            if matches!(retirement, Retirement::Release) {
                for input in std::mem::take(&mut inactive) {
                    model.release_session(input.session)?;
                }
            }
        }
        let output = model.decode_grouped(&rows)?;
        for (input, (output, execution)) in rows.iter_mut().zip(output) {
            assert_eq!(
                execution,
                if step < 8 {
                    DecodeExecution::Packed { rows: 5 }
                } else {
                    DecodeExecution::Scalar
                }
            );
            input.token = token(model, output)?;
        }
        observed.push(rows[0].token);
        if step == 8 {
            model.flush_decode_graphs()?;
        }
    }
    let differences = observed
        .iter()
        .zip(&expected)
        .enumerate()
        .filter(|(_, (actual, expected))| actual != expected)
        .map(|(offset, (actual, expected))| (offset, *actual, *expected))
        .collect::<Vec<_>>();
    writeln!(
        std::io::stderr().lock(),
        "churn.retirement: mode={retirement:?}, scalar_differences={differences:?}, observed={observed:?}, scalar={expected:?}"
    )?;
    for input in rows.into_iter().chain(inactive) {
        model.release_session(input.session)?;
    }
    model.clear_prefix_cache();
    assert_eq!(model.stream().paged_arenas().resident_arenas()?, 0);
    Ok((observed, expected))
}
