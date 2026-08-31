// tb_matmul_int4.cpp — testbench for matmul_int4
#include <hls_half.h>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <vector>

#define DIM     1024
#define TILE    32
#define NTILES  (DIM / TILE)

typedef half   data_t;
typedef half   acc_t;
typedef half   scale_t;

extern "C" void matmul_int4(
    const uint8_t* __restrict__ A_packed,
    const scale_t* __restrict__ scales,
    const data_t*  __restrict__ x,
    acc_t*         __restrict__ y
);

int main() {
    srand(42);

    std::vector<float>   A(DIM * DIM);
    std::vector<float>   x(DIM);
    std::vector<int8_t>  A_q(DIM * DIM);      // one int4 value per int8 slot (pre-pack)
    std::vector<uint8_t> A_packed(DIM * DIM / 2);
    std::vector<scale_t> scales(DIM * NTILES);
    std::vector<data_t>  x_buf(DIM);
    std::vector<acc_t>   y_hw(DIM);
    std::vector<float>   y_ref(DIM);

    // ---- generate random fp32 weight matrix and input vector ----
    for (int i = 0; i < DIM * DIM; i++)
        A[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f; // [-1, 1]
    for (int i = 0; i < DIM; i++) {
        x[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
        x_buf[i] = (data_t)x[i];
    }

    // ---- groupwise symmetric int4 quantization (per row, per tile of 32) ----
    for (int i = 0; i < DIM; i++) {
        for (int k = 0; k < NTILES; k++) {
            float max_abs = 1e-8f;
            for (int t = 0; t < TILE; t++) {
                float v = fabsf(A[i * DIM + k * TILE + t]);
                if (v > max_abs) max_abs = v;
            }
            float scale = max_abs / 7.0f;   // int4 signed range [-8,7], use 7 for symmetry
            scales[i * NTILES + k] = (scale_t)scale;

            for (int t = 0; t < TILE; t++) {
                float w = A[i * DIM + k * TILE + t];
                int q = (int)roundf(w / scale);
                if (q > 7)  q = 7;
                if (q < -8) q = -8;
                A_q[i * DIM + k * TILE + t] = (int8_t)q;
            }
        }
    }

    // ---- pack two int4 nibbles per byte (matches kernel: lo = even idx, hi = odd idx) ----
    for (int idx = 0; idx < DIM * DIM; idx += 2) {
        uint8_t lo = (uint8_t)(A_q[idx]     & 0x0F);
        uint8_t hi = (uint8_t)(A_q[idx + 1] & 0x0F);
        A_packed[idx / 2] = (hi << 4) | lo;
    }

    // ---- software reference: dequant + matvec in float ----
    for (int i = 0; i < DIM; i++) {
        float acc = 0.0f;
        for (int k = 0; k < NTILES; k++) {
            float scale = (float)scales[i * NTILES + k];
            for (int t = 0; t < TILE; t++) {
                float w = (float)A_q[i * DIM + k * TILE + t] * scale;
                acc += w * x[k * TILE + t];
            }
        }
        y_ref[i] = acc;
    }

    // ---- run hardware kernel ----
    matmul_int4(A_packed.data(), scales.data(), x_buf.data(), y_hw.data());

    // ---- compare ----
    double max_err = 0.0, mean_err = 0.0;
    bool has_nan = false;
    for (int i = 0; i < DIM; i++) {
        float hw = (float)y_hw[i];
        float err = fabsf(hw - y_ref[i]);
        if (std::isnan(hw)) has_nan = true;
        max_err = std::max(max_err, (double)err);
        mean_err += err;
    }
    mean_err /= DIM;

    std::cout << "max_err  = " << max_err  << "\n";
    std::cout << "mean_err = " << mean_err << "\n";
    std::cout << "has_nan  = " << (has_nan ? "true" : "false") << "\n";

    // tolerance: int4 quant noise is coarse, allow relative margin
    float tol = 0.5f; // absolute, tune based on data range
    bool pass = (max_err < tol) && !has_nan;

    std::cout << (pass ? "TEST PASSED" : "TEST FAILED") << std::endl;
    return pass ? 0 : 1;
}
