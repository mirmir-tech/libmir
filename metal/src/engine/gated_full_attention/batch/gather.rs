use crate::engine::{
    Array, Error, KvCache, KvContext, PagedContextMode, Result, Stream,
    kernels::page_gather::Source,
};

pub(super) fn enabled(
    caches: &[&mut KvCache],
    mode: PagedContextMode,
    causal: bool,
    sequence: i32,
    stream: &Stream,
) -> bool {
    mode == PagedContextMode::View
        && !causal
        && sequence == 1
        && (2..=12).contains(&caches.len())
        && caches.iter().all(|cache| cache.has_native_page_views())
        && stream.config().diagnostics.history_batching == crate::config::HistoryBatching::Gathered
}

pub(super) fn execute(
    queries: &Array,
    contexts: &[KvContext],
    scale: f32,
    stream: &Stream,
) -> Result<Array> {
    let paged = contexts
        .iter()
        .map(|context| {
            context
                .paged
                .as_ref()
                .filter(|pages| {
                    context.mask.is_none()
                        && pages.key_scales.is_none()
                        && pages.value_scales.is_none()
                })
                .ok_or_else(|| {
                    Error::InvalidModel("page gather requires native unmasked K/V".into())
                })
        })
        .collect::<Result<Vec<_>>>()?;
    let first = paged.first().ok_or(Error::NullHandle("page gather batch"))?;
    if paged
        .iter()
        .any(|p| p.context_tokens != first.context_tokens || p.page_size != first.page_size)
    {
        return Err(Error::InvalidModel("page gather contexts differ".into()));
    }
    let sources = paged
        .iter()
        .map(|p| Source {
            keys: p.key_pages.native(),
            values: p.value_pages.native(),
            table: p.page_table.native(),
        })
        .collect::<Vec<_>>();
    let [keys, values] =
        stream.kernels().page_gather.execute(&sources, first.context_tokens, stream)?;
    let keys = Array::from_native(keys)?;
    let values = Array::from_native(values)?;
    crate::engine::probe::history::record_gather_batch();
    queries.scaled_dot_product_attention(&keys, &values, scale, false, stream)
}
