//! GGML conversion kernels: dequantization, embedding rows and the MXFP4
//! repack to the native expert layout.

mirtal::metal_kernel! {
    pub(super) fn ggml_dequantize {
        name: "mirmir_ggml_dequantize",
        templates: [T: dtype = bf16, TYPE: int = 12, INPUT: int = 4096],
        inputs: [weight: u8],
        outputs: [output: T],
        source: file "kernels/ggml/dequantize.metal",
        header: files ["kernels/ggml/blocks.h", "kernels/ggml/decode.h"],
        row_contiguous: true,
        atomic_outputs: false,
    }
}

mirtal::metal_kernel! {
    pub(super) fn ggml_dequantize_grids {
        name: "mirmir_ggml_dequantize_grids",
        templates: [T: dtype = bf16, TYPE: int = 12, INPUT: int = 4096],
        inputs: [weight: u8],
        outputs: [output: T],
        source: file "kernels/ggml/dequantize.metal",
        header: files [
            "kernels/ggml/blocks.h",
            "kernels/ggml/grids.h",
            "kernels/ggml/iq.h",
            "kernels/ggml/decode.h",
        ],
        row_contiguous: true,
        atomic_outputs: false,
    }
}

mirtal::metal_kernel! {
    pub(super) fn ggml_embedding {
        name: "mirmir_ggml_embedding",
        templates: [
            T: dtype = bf16, TYPE: int = 12, HIDDEN: int = 4096, VOCABULARY: int = 151936,
        ],
        inputs: [weight: u8, indices: u32],
        outputs: [output: T],
        source: file "kernels/ggml/embedding.metal",
        header: files ["kernels/ggml/blocks.h", "kernels/ggml/decode.h"],
        row_contiguous: true,
        atomic_outputs: false,
    }
}

mirtal::metal_kernel! {
    pub(super) fn ggml_embedding_grids {
        name: "mirmir_ggml_embedding_grids",
        templates: [
            T: dtype = bf16, TYPE: int = 12, HIDDEN: int = 4096, VOCABULARY: int = 151936,
        ],
        inputs: [weight: u8, indices: u32],
        outputs: [output: T],
        source: file "kernels/ggml/embedding.metal",
        header: files [
            "kernels/ggml/blocks.h",
            "kernels/ggml/grids.h",
            "kernels/ggml/iq.h",
            "kernels/ggml/decode.h",
        ],
        row_contiguous: true,
        atomic_outputs: false,
    }
}

mirtal::metal_kernel! {
    pub(super) fn ggml_mxfp4_repack {
        name: "mirmir_ggml_mxfp4_repack",
        templates: [BLOCKS: int = 90],
        inputs: [weight: u8],
        outputs: [blocks: u8, scales: u8],
        source: file "kernels/ggml/mxfp4_repack.metal",
        header: inline "",
        row_contiguous: true,
        atomic_outputs: false,
    }
}

mirtal::metal_kernel! {
    pub(super) fn ggml_affine_repack {
        name: "mirmir_ggml_affine_repack",
        templates: [T: dtype = bf16, TYPE: int = 12, INPUT: int = 4096, BITS: int = 4, OFFSET: int = 0],
        inputs: [weight: u8],
        outputs: [packed: u32, scales: T, biases: T],
        source: file "kernels/ggml/affine.metal",
        header: files ["kernels/ggml/blocks.h", "kernels/ggml/decode.h"],
        row_contiguous: true,
        atomic_outputs: false,
    }
}
