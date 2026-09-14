/// Select the first maximum, matching device argmax for finite logits,
/// including ties between positive and negative zero.
pub fn argmax(values: &[f32]) -> Option<usize> {
    values
        .iter()
        .enumerate()
        .reduce(|best, candidate| {
            if candidate.1 > best.1 {
                candidate
            } else {
                best
            }
        })
        .map(|(index, _)| index)
}
