// One token of the Mamba-2 recurrence per head.
//   next_state[h, d, s] = exp(A[h] * dt[h]) * state[h, d, s] + x[h, d] * dt[h] * B[g(h), s]
//   output[h, d] = sum_s next_state[h, d, s] * C[g(h), s] + x[h, d] * D[h]
// Heads run one threadgroup per head and channel row; each thread handles
// `Ds / 32` state entries, which `simd_sum` folds into one channel value.
auto n = thread_position_in_grid.z;
auto h_idx = n % H;
auto g_idx = n / G;
constexpr int n_per_t = Ds / 32;

auto x = X + n * Dh;
out += n * Dh;
auto i_state = state_in + n * Dh * Ds;
auto o_state = state_out + n * Dh * Ds;

auto C_ = C + g_idx * Ds;
auto B_ = B + g_idx * Ds;

auto ds_idx = thread_position_in_threadgroup.x;
auto d_idx = thread_position_in_grid.y;

auto dt_ = static_cast<float>(dt[n]);
auto A = -fast::exp(static_cast<float>(A_log[h_idx]));
auto dA = fast::exp(A * dt_);

float acc = 0.0;
auto x_ = static_cast<float>(x[d_idx]);

for (int i = 0; i < n_per_t; ++i) {
    auto s_idx = n_per_t * ds_idx + i;
    auto idx = d_idx * Ds + s_idx;
    auto dB_by_x = x_ * dt_ * static_cast<float>(B_[s_idx]);
    auto state = dA * static_cast<float>(i_state[idx]) + dB_by_x;
    o_state[idx] = static_cast<U>(state);
    acc += state * static_cast<float>(C_[s_idx]);
}
acc = simd_sum(acc);
if (thread_index_in_simdgroup == 0) {
    out[d_idx] = static_cast<T>(acc + x_ * static_cast<float>(D[h_idx]));
}
