//! GGML product kernels: few-token GEMV and simdgroup-matrix products, and
//! their routed-expert variants.

mirtal::metal_kernel! {
    pub(super) fn ggml_gemv {
        name: "mirmir_ggml_gemv",
        templates: [
            T: dtype = bf16, TYPE: int = 12, INPUT: int = 4096, OUTPUT: int = 4096,
            TOKENS: int = 1, ROWS: int = 2,
        ],
        inputs: [input: T, weight: u8],
        outputs: [output: T],
        source: file "kernels/ggml/gemv.metal",
        header: files ["kernels/ggml/blocks.h", "kernels/ggml/decode.h"],
        row_contiguous: true,
        atomic_outputs: false,
    }
}

mirtal::metal_kernel! {
    pub(super) fn ggml_gemv_grids {
        name: "mirmir_ggml_gemv_grids",
        templates: [
            T: dtype = bf16, TYPE: int = 12, INPUT: int = 4096, OUTPUT: int = 4096,
            TOKENS: int = 1, ROWS: int = 2,
        ],
        inputs: [input: T, weight: u8],
        outputs: [output: T],
        source: file "kernels/ggml/gemv.metal",
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
    pub(super) fn ggml_gathered {
        name: "mirmir_ggml_gathered",
        templates: [
            T: dtype = bf16, TYPE: int = 12, INPUT: int = 4096, OUTPUT: int = 4096,
            EXPERTS: int = 128, GROUP: int = 8,
        ],
        inputs: [input: T, weight: u8, indices: u32],
        outputs: [output: T],
        source: file "kernels/ggml/gathered.metal",
        header: files ["kernels/ggml/blocks.h", "kernels/ggml/decode.h"],
        row_contiguous: true,
        atomic_outputs: false,
    }
}

mirtal::metal_kernel! {
    pub(super) fn ggml_grouped {
        name: "mirmir_ggml_grouped",
        templates: [
            T: dtype = bf16, TYPE: int = 12, INPUT: int = 4096, OUTPUT: int = 4096,
            EXPERTS: int = 128, ASSIGNMENTS: int = 16384,
        ],
        inputs: [input: T, weight: u8, indices: u32],
        outputs: [output: T],
        source: file "kernels/ggml/grouped.metal",
        header: files ["kernels/ggml/blocks.h", "kernels/ggml/decode.h"],
        row_contiguous: true,
        atomic_outputs: false,
    }
}

mirtal::metal_kernel! {
    pub(super) fn ggml_gathered_grids {
        name: "mirmir_ggml_gathered_grids",
        templates: [
            T: dtype = bf16, TYPE: int = 12, INPUT: int = 4096, OUTPUT: int = 4096,
            EXPERTS: int = 128, GROUP: int = 8,
        ],
        inputs: [input: T, weight: u8, indices: u32],
        outputs: [output: T],
        source: file "kernels/ggml/gathered.metal",
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
    pub(super) fn ggml_grouped_grids {
        name: "mirmir_ggml_grouped_grids",
        templates: [
            T: dtype = bf16, TYPE: int = 12, INPUT: int = 4096, OUTPUT: int = 4096,
            EXPERTS: int = 128, ASSIGNMENTS: int = 16384,
        ],
        inputs: [input: T, weight: u8, indices: u32],
        outputs: [output: T],
        source: file "kernels/ggml/grouped.metal",
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
    pub(super) fn ggml_mma {
        name: "mirmir_ggml_mma",
        templates: [
            T: dtype = bf16, TYPE: int = 12, INPUT: int = 4096, OUTPUT: int = 4096,
            TOKENS: int = 4,
        ],
        inputs: [input: T, weight: u8],
        outputs: [output: T],
        source: file "kernels/ggml/mma.metal",
        header: files ["kernels/ggml/blocks.h", "kernels/ggml/decode.h"],
        row_contiguous: true,
        atomic_outputs: false,
    }
}

mirtal::metal_kernel! {
    pub(super) fn ggml_mma_grids {
        name: "mirmir_ggml_mma_grids",
        templates: [
            T: dtype = bf16, TYPE: int = 12, INPUT: int = 4096, OUTPUT: int = 4096,
            TOKENS: int = 4,
        ],
        inputs: [input: T, weight: u8],
        outputs: [output: T],
        source: file "kernels/ggml/mma.metal",
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
