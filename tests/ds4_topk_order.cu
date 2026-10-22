// Exact top-k sets AND the indexer's output order: values above the k-th in position order,
// then the lowest-position ties in position order. Exercises the old kernel, segmented merge,
// repeatability, row strides, mapped candidates, and staging-cap boundaries.
// nvcc -O3 -std=c++17 -arch=native -I include -I src/ds4 tests/ds4_topk_order.cu src/ds4/ds4_kernels.cu \
//      src/ds4/ds4_host_ar.cpp -Xcompiler -mavx2,-mfma,-mf16c -o build/ds4_topk_order
#include "ds4_kernels.cuh"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <vector>

using namespace strata::ds4;

namespace {
constexpr int k = 512, rows = 3, output_stride = k + 17, guard = -123456789;

void check_cuda(cudaError_t status, const char* call, int line) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "GPU failure at line %d: %s: %s\n", line, call, cudaGetErrorString(status));
        std::exit(2);
    }
}
#define GPU(call) check_cuda((call), #call, __LINE__)

uint32_t key(float f) {
    uint32_t b;
    std::memcpy(&b, &f, sizeof(b));
    return b & 0x80000000u ? ~b : b | 0x80000000u;
}

std::vector<int> reference(const float* v, int n) {
    std::vector<float> sorted(v, v + n);
    std::nth_element(sorted.begin(), sorted.begin() + k - 1, sorted.end(), std::greater<float>());
    const float kth = sorted[k - 1];
    std::vector<int> want;
    for (int i = 0; i < n; ++i)
        if (v[i] > kth) want.push_back(i);
    for (int i = 0; i < n && (int) want.size() < k; ++i)
        if (v[i] == kth) want.push_back(i);
    return want;
}

bool compare(const std::vector<int>& got, const std::vector<std::vector<int>>& want, const char* path, int n,
             int kind, const std::vector<int>* candidates = nullptr, int candidate_stride = 0) {
    for (int t = 0; t < rows; ++t) {
        for (int j = 0; j < output_stride; ++j) {
            int expected = guard;
            if (j < k) {
                const int pos = want[t][j];
                expected = candidates ? (*candidates)[(size_t) t * candidate_stride + pos / 4] * 4 + pos % 4 : pos;
            }
            const int actual = got[1 + (size_t) t * output_stride + j];
            if (actual != expected) {
                std::fprintf(stderr, "%s n=%d kind=%d row=%d slot=%d: got %d, expected %d\n", path, n, kind, t, j,
                             actual, expected);
                return false;
            }
        }
    }
    if (got.front() != guard || got.back() != guard) {
        std::fprintf(stderr, "%s n=%d kind=%d: output boundary overwritten\n", path, n, kind);
        return false;
    }
    return true;
}
}   // namespace

int main() {
    std::mt19937 rng(7);
    int fails = 0, cases = 0;
    const char* names[] = {"random", "heavy ties", "one value", "overflow plateau"};
    // 131072: eight segments, 4096 candidates exactly; 131073: nine, 4608 (> HIP cap).
    // 524288: 32 segments, 16384 candidates (> both HIP 4096 and CUDA 8192 caps).
    for (int n : {k + 1, 600, 4096, kTopkSegMin - 1, kTopkSegMin, kTopkSegMin + 1, 65536, 131072, 131073,
                  262144, 524288}) {
        const int P = std::min(kTopkSegMax, std::max(2, (n + kTopkSeg - 1) / kTopkSeg));
        const int S = (n + P - 1) / P;
        for (int kind = 0; kind < 4; ++kind) {
            // Padding and different rows catch incorrect input/output/candidate row strides.
            const int input_stride = n + 11, nb = (n + 3) / 4, candidate_stride = nb + 5;
            std::vector<float> v((size_t) rows * input_stride, -999.0f);
            std::vector<std::vector<int>> want;
            std::normal_distribution<float> nd;
            for (int t = 0; t < rows; ++t) {
                float* vr = v.data() + (size_t) t * input_stride;
                for (int i = 0; i < n; ++i) {
                    vr[i] = kind == 0 ? nd(rng) : kind == 1 ? (float) ((rng() % 21) / 4) : 1.5f;
                    if (kind == 3) {
                        // Each segment has >512 ties at 2 and seven later 3s. Its local 512th is 2,
                        // so ALL P*512 segment candidates pass the merge floor. The global output
                        // starts with later 3s and then earlier 2s: both output runs matter.
                        const int local = i % S;
                        vr[i] = local < 1024 + t ? 2.0f : 0.0f;
                        if (local >= 2048 + t && local < 2055 + t) vr[i] = 3.0f;
                    }
                }
                want.push_back(reference(vr, n));
            }
            std::vector<int> candidates((size_t) rows * candidate_stride, guard);
            for (int t = 0; t < rows; ++t)
                for (int b = 0; b < nb; ++b)
                    candidates[(size_t) t * candidate_stride + b] = 7 * (nb - 1 - b) + 3 + t * (7 * nb + 19);
            float* dv = nullptr;
            int* dout = nullptr;
            int* dc = nullptr;
            GPU(cudaMalloc(&dv, v.size() * sizeof(float)));
            GPU(cudaMalloc(&dout, (rows * output_stride + 2) * sizeof(int)));
            GPU(cudaMalloc(&dc, candidates.size() * sizeof(int)));
            GPU(cudaMemcpy(dv, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice));
            GPU(cudaMemcpy(dc, candidates.data(), candidates.size() * sizeof(int), cudaMemcpyHostToDevice));
            TopkScratch sc;
            sc.T = rows, sc.P = kTopkSegMax, sc.K = k;
            GPU(cudaMalloc(&sc.val, (size_t) sc.T * sc.P * sc.K * sizeof(float)));
            GPU(cudaMalloc(&sc.idx, (size_t) sc.T * sc.P * sc.K * sizeof(int)));
            GPU(cudaMalloc(&sc.cnt, (size_t) sc.T * sc.P * sizeof(int)));
            GPU(cudaMalloc(&sc.kth, (size_t) sc.T * sc.P * sizeof(uint32_t)));
            Rows r;
            for (int t = 0; t < rows; ++t) r.n[t] = n, r.k[t] = k;
            bool ok = true;
            for (int mapped = 0; mapped < 2; ++mapped) {
                std::vector<int> baseline;
                for (int segmented = 0; segmented < 2; ++segmented) {
                    std::vector<int> previous;
                    for (int rep = 0; rep < 2; ++rep) {
                        std::vector<int> got(rows * output_stride + 2, guard);
                        GPU(cudaMemcpy(dout, got.data(), got.size() * sizeof(int), cudaMemcpyHostToDevice));
                        topk(dv, input_stride, rows, r, mapped ? dc : nullptr, candidate_stride, 4, dout + 1,
                             output_stride, nullptr, segmented ? &sc : nullptr);
                        GPU(cudaGetLastError());
                        GPU(cudaDeviceSynchronize());
                        GPU(cudaMemcpy(got.data(), dout, got.size() * sizeof(int), cudaMemcpyDeviceToHost));
                        const char* path = segmented ? (mapped ? "segmented mapped" : "segmented")
                                                     : (mapped ? "old mapped" : "old");
                        ok = compare(got, want, path, n, kind, mapped ? &candidates : nullptr, candidate_stride) && ok;
                        if (rep && got != previous) {
                            std::fprintf(stderr, "%s n=%d kind=%d: repeated output differs\n", path, n, kind);
                            ok = false;
                        }
                        if (segmented && got != baseline) {
                            std::fprintf(stderr, "%s n=%d kind=%d: differs from old kernel\n", path, n, kind);
                            ok = false;
                        }
                        if (!segmented) baseline = got;
                        previous = got;
                    }
                }
            }
            int passing_min = -1, passing_max = -1;
            if (n >= kTopkSegMin) {
                std::vector<int> counts(rows * P);
                std::vector<uint32_t> floors(rows * P);
                std::vector<float> values((size_t) rows * P * k);
                GPU(cudaMemcpy(counts.data(), sc.cnt, counts.size() * sizeof(int), cudaMemcpyDeviceToHost));
                GPU(cudaMemcpy(floors.data(), sc.kth, floors.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
                GPU(cudaMemcpy(values.data(), sc.val, values.size() * sizeof(float), cudaMemcpyDeviceToHost));
                for (int t = 0; t < rows; ++t) {
                    const uint32_t floor = *std::max_element(floors.begin() + t * P, floors.begin() + (t + 1) * P);
                    int passing = 0;
                    for (int p = 0; p < P; ++p) {
                        const int count = counts[t * P + p];
                        if (count != k) {
                            std::fprintf(stderr, "n=%d kind=%d row=%d segment=%d: count %d, expected %d\n", n, kind,
                                         t, p, count, k);
                            ok = false;
                        }
                        for (int j = 0; j < std::max(0, std::min(count, k)); ++j)
                            passing += key(values[((size_t) t * P + p) * k + j]) >= floor;
                    }
                    if (kind >= 2 && passing != P * k) {
                        std::fprintf(stderr, "n=%d kind=%d row=%d: passing %d, expected %d (fallback proof failed)\n",
                                     n, kind, t, passing, P * k);
                        ok = false;
                    }
                    passing_min = passing_min < 0 ? passing : std::min(passing_min, passing);
                    passing_max = std::max(passing_max, passing);
                }
            }
            std::printf("n %6d, %-16s: %s; segments=%d, floor-passing=%d..%d\n", n, names[kind],
                        ok ? "PASS (host/order/repeat/old/mapped/guards)" : "FAIL", n >= kTopkSegMin ? P : 0,
                        passing_min, passing_max);
            ++cases;
            fails += !ok;
            GPU(cudaFree(dc));
            GPU(cudaFree(sc.val));
            GPU(cudaFree(sc.idx));
            GPU(cudaFree(sc.cnt));
            GPU(cudaFree(sc.kth));
            GPU(cudaFree(dv));
            GPU(cudaFree(dout));
        }
    }
    std::printf("%s: %d cases, %d failed\n", fails == 0 ? "PASS" : "FAIL", cases, fails);
    return fails == 0 ? 0 : 1;
}
