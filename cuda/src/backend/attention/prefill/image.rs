#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::backend) struct ImageAttentionSpan {
    pub start: usize,
    pub end: usize,
}
