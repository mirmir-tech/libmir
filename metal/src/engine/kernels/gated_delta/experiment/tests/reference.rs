use super::*;

// Independent f64 scalar recurrence; no SIMD-tree replication.
pub(super) fn check(
    stream: &Stream,
    case: Case,
    inputs: &[Array; 6],
    outputs: &[Array; 2],
) -> Result<()> {
    let values = inputs.iter().map(|x| read(stream, x)).collect::<Result<Vec<_>>>()?;
    let [q, k, v, g, beta, initial] = values.as_slice() else {
        unreachable!()
    };
    let mut state = initial.iter().map(|&x| f64::from(x)).collect::<Vec<_>>();
    let mut output = vec![0.0; v.len()];
    let Case {
        batch,
        steps,
        key_heads: hk,
        value_heads: hv,
        key_dim: dk,
        value_dim: dv,
        ..
    } = case;
    for b in 0..batch {
        for t in 0..steps {
            for h in 0..hv {
                let qk = ((b * steps + t) * hk + h / (hv / hk)) * dk;
                let gate = (b * steps + t) * hv + h;
                for row in 0..dv {
                    let start = ((b * hv + h) * dv + row) * dk;
                    let vi = ((b * steps + t) * hv + h) * dv + row;
                    let memory = &mut state[start..start + dk];
                    let mut projection = 0.0;
                    for (d, s) in memory.iter_mut().enumerate() {
                        *s *= f64::from(g[gate]);
                        projection = (*s).mul_add(f64::from(k[qk + d]), projection);
                    }
                    let delta = (f64::from(v[vi]) - projection) * f64::from(beta[gate]);
                    for (d, s) in memory.iter_mut().enumerate() {
                        *s = f64::from(k[qk + d]).mul_add(delta, *s);
                        output[vi] = (*s).mul_add(f64::from(q[qk + d]), output[vi]);
                    }
                }
            }
        }
    }
    for (index, expected) in [output, state].iter().enumerate() {
        let actual = read(stream, &outputs[index])?;
        let error = actual
            .iter()
            .zip(expected)
            .map(|(&a, &b)| (f64::from(a) - b).powi(2))
            .sum::<f64>()
            .sqrt();
        let norm = expected.iter().map(|v| v * v).sum::<f64>().sqrt();
        let relative = error / norm.max(1e-30);
        let limit = if index == 0 {
            0.004
        } else {
            2e-6
        };
        assert!(relative < limit, "f64 reference {case:?} output/state {index}: {relative}");
    }
    Ok(())
}
