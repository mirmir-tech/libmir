use std::path::PathBuf;

use mircuda::{
    Compiler, Context, DeviceBuffer, DeviceElement, Driver, LaunchConfig, MemoryPool, Stream, bf16,
};

use super::*;

type KvArguments<'a> = (
    &'a DeviceBuffer<bf16>,
    &'a DeviceBuffer<bf16>,
    &'a mut DeviceBuffer<u8>,
    &'a mut DeviceBuffer<u8>,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
);

type AttentionArguments<'a> = (
    &'a DeviceBuffer<bf16>,
    &'a DeviceBuffer<u8>,
    &'a DeviceBuffer<u8>,
    &'a DeviceBuffer<u32>,
    &'a mut DeviceBuffer<bf16>,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    f32,
    u32,
);

struct Resources<'a> {
    kv: PagedKvStore,
    attention: PagedAttention,
    stream: &'a Stream,
    keys: &'a DeviceBuffer<bf16>,
    values: &'a DeviceBuffer<bf16>,
    key_pages: &'a mut DeviceBuffer<u8>,
    value_pages: &'a mut DeviceBuffer<u8>,
    query: &'a DeviceBuffer<bf16>,
    table: &'a DeviceBuffer<u32>,
    output: &'a mut DeviceBuffer<bf16>,
    local_start: u32,
    page_start: u32,
    total_tokens: u32,
}

#[test]
fn graph_rebinds_paged_store_and_attention() -> Result<()> {
    let runtime = Runtime::new()?;
    let kv_spec = PagedKvSpec {
        block_size: 2,
        block_count: 1,
        kv_heads: 1,
        key_head_dim: 2,
        value_head_dim: 2,
        dtype: KvCacheDType::BFloat16,
    };
    let attention_spec = PagedAttentionSpec {
        block_size: 2,
        max_blocks: 1,
        query_heads: 1,
        kv_heads: 1,
        head_dim: 2,
        value_head_dim: 2,
        dtype: KvCacheDType::BFloat16,
    };
    let kv = PagedKvStore::compile(&runtime.compiler, kv_spec)?;
    let attention = PagedAttention::compile(&runtime.compiler, attention_spec)?;
    let keys = runtime.copy(&bf16s(&[1.0, 0.0, 0.0, 1.0]))?;
    let values = runtime.copy(&bf16s(&[2.0, 4.0, 10.0, 20.0]))?;
    let query = runtime.copy(&bf16s(&[1.0, 0.0]))?;
    let table = runtime.copy(&[0_u32])?;
    let mut key_pages = runtime.pool.allocate_zeroed(&runtime.stream, kv.key_bytes()?)?;
    let mut value_pages = runtime.pool.allocate_zeroed(&runtime.stream, kv.value_bytes()?)?;
    let mut output = runtime.pool.allocate::<bf16>(&runtime.stream, 2)?;
    kv.execute(&runtime.stream, &keys, &values, &mut key_pages, &mut value_pages, 0, 1, 0, 0)?;
    runtime.stream.synchronize()?;
    let kv_kernel = kv.kernel.clone();
    let attention_kernel = attention.kernel.clone();
    let resources = Resources {
        kv,
        attention,
        stream: &runtime.stream,
        keys: &keys,
        values: &values,
        key_pages: &mut key_pages,
        value_pages: &mut value_pages,
        query: &query,
        table: &table,
        output: &mut output,
        local_start: 0,
        page_start: 0,
        total_tokens: 1,
    };
    {
        let mut graph = runtime.stream.capture(resources, capture)?;
        let kv_nodes = graph.kernel_nodes(&kv_kernel)?;
        let attention_nodes = graph.kernel_nodes(&attention_kernel)?;
        assert_eq!((kv_nodes.len(), attention_nodes.len()), (1, 1));
        graph.update_kernel(&kv_nodes[0], &kv_kernel, kv_config(), rebind_kv)?;
        graph.update_kernel(
            &attention_nodes[0],
            &attention_kernel,
            kv_config(),
            rebind_attention,
        )?;
        graph.launch(&runtime.stream)?;
    }
    let actual = runtime.read(&output)?;
    let first_weight = 1.0_f32.exp() / (1.0_f32.exp() + 1.0);
    let expected = [
        first_weight.mul_add(2.0, (1.0 - first_weight) * 10.0),
        first_weight.mul_add(4.0, (1.0 - first_weight) * 20.0),
    ];
    for (actual, expected) in actual.iter().zip(expected) {
        assert!((actual.to_f32() - expected).abs() < 0.1);
    }
    Ok(())
}
fn capture(resources: &mut Resources<'_>) -> Result<()> {
    resources.kv.execute(
        resources.stream,
        resources.keys,
        resources.values,
        resources.key_pages,
        resources.value_pages,
        0,
        1,
        0,
        0,
    )?;
    resources.attention.execute(
        resources.stream,
        resources.query,
        resources.key_pages,
        resources.value_pages,
        resources.table,
        resources.output,
        1,
        1,
        None,
        1.0,
    )
}
fn rebind_kv<'a>(resources: &'a mut Resources<'_>) -> KvArguments<'a> {
    resources.local_start = 1;
    resources.page_start = 1;
    (
        resources.keys,
        resources.values,
        resources.key_pages,
        resources.value_pages,
        resources.local_start,
        1,
        0,
        resources.page_start,
        2,
        1,
        2,
        2,
    )
}

fn rebind_attention<'a>(resources: &'a mut Resources<'_>) -> AttentionArguments<'a> {
    resources.total_tokens = 2;
    (
        resources.query,
        resources.key_pages,
        resources.value_pages,
        resources.table,
        resources.output,
        resources.total_tokens,
        1,
        2,
        1,
        1,
        2,
        2,
        0,
        1.0,
        0,
    )
}

const fn kv_config() -> LaunchConfig {
    LaunchConfig {
        grid: (1, 1, 1),
        block: (256, 1, 1),
        shared_memory_bytes: 0,
    }
}

fn bf16s(values: &[f32]) -> Vec<bf16> {
    values.iter().copied().map(bf16::from_f32).collect()
}

struct Runtime {
    context: Context,
    stream: Stream,
    pool: MemoryPool,
    compiler: Compiler,
}

impl Runtime {
    fn new() -> Result<Self> {
        let driver = Driver::initialize()?;
        let device = driver.devices()?.into_iter().next().ok_or(mircuda::Error::InvalidLaunch)?;
        let context = driver.create_context(device)?;
        let stream = context.create_stream()?;
        let pool = context.default_memory_pool()?;
        let compiler = Compiler::with_include_paths(
            context.clone(),
            [PathBuf::from("/usr/local/cuda/include")],
        )?;
        Ok(Self { context, stream, pool, compiler })
    }

    fn copy<T: DeviceElement>(&self, values: &[T]) -> Result<DeviceBuffer<T>> {
        let mut host = self.context.allocate_pinned::<T>(values.len())?;
        host.copy_from_slice(values)?;
        let mut device = self.pool.allocate::<T>(&self.stream, values.len())?;
        self.stream.copy_to_device(&mut host, &mut device)?;
        self.stream.synchronize()?;
        Ok(device)
    }

    fn read<T: DeviceElement>(&self, source: &DeviceBuffer<T>) -> Result<Vec<T>> {
        let mut host = self.context.allocate_pinned::<T>(source.len())?;
        self.stream.copy_to_host(source, &mut host)?;
        Ok(host.to_vec()?)
    }
}
