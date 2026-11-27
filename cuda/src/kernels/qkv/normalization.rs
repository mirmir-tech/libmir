use mircuda::CompileOptions;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QkvNormalization {
    pub query: bool,
    pub key: bool,
    pub value: bool,
    pub scope: QkNormScope,
    pub weights: QkNormWeights,
}

/// Where the query and key norms take their scale from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QkNormWeights {
    /// The checkpoint's learned weights.
    Learned,
    /// No learned scale (Llama 4's unit RMS norms): the kernels read unit
    /// weights.
    Unit,
}

/// What one query or key norm covers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QkNormScope {
    /// Each head separately, with a `head_dim` weight.
    Head,
    /// A token's whole projection across its heads (`OLMo` 2), with a weight
    /// spanning every head.
    Projection,
}

impl QkvNormalization {
    pub const ALL: Self = Self {
        query: true,
        key: true,
        value: true,
        scope: QkNormScope::Head,
        weights: QkNormWeights::Learned,
    };
    pub const NONE: Self = Self {
        query: false,
        key: false,
        value: false,
        scope: QkNormScope::Head,
        weights: QkNormWeights::Learned,
    };
    pub const QUERY_KEY: Self = Self {
        query: true,
        key: true,
        value: false,
        scope: QkNormScope::Head,
        weights: QkNormWeights::Learned,
    };
    pub const QUERY_KEY_ACROSS_HEADS: Self = Self {
        query: true,
        key: true,
        value: false,
        scope: QkNormScope::Projection,
        weights: QkNormWeights::Learned,
    };

    /// The norm weight lengths of the query and key projections.
    #[must_use]
    pub const fn weight_lengths(
        self,
        query_heads: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> [usize; 2] {
        match self.scope {
            QkNormScope::Head => [head_dim, head_dim],
            QkNormScope::Projection => [query_heads * head_dim, kv_heads * head_dim],
        }
    }

    /// Defines the macros of this variant of the postprocess kernels.
    pub(super) fn define_macros(self, options: &mut CompileOptions) {
        if self.scope == QkNormScope::Projection {
            options.extra_options.push("--define-macro=LIBMIR_QK_NORM_ACROSS_HEADS".into());
        }
    }
}
