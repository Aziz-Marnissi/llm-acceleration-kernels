// matmul_fp16_tb.cpp
#include "hls_half.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

#define DIM 1024

typedef half data_t;
typedef half acc_t;

#include "matmul_weights_fp16_N1024.h"   // provides A_fp16[1024][1024], x_fp16[1024]

extern "C" void matmul_fp16(const data_t* A, const data_t* x, acc_t* y);

int main() {
    static acc_t y_hw[DIM];
    static float y_ref[DIM];   // reference kept in fp32, error measured vs fp16 output

    matmul_fp16(&A_fp16[0][0], x_fp16, y_hw);

    for (int i = 0; i < DIM; i++) {
        float acc = 0.0f;
        for (int k = 0; k < DIM; k++) {
            acc += (float)A_fp16[i][k] * (float)x_fp16[k];
        }
        y_ref[i] = acc;
    }

    double max_abs_err = 0.0, max_rel_err = 0.0;
    double sum_abs_err = 0.0;
    int fail_cnt = 0;
    const double ABS_TOL = 5e-1;   // loosened: fp16 accumulation over 1024 terms
    const double REL_TOL = 1.5e-1;

    for (int i = 0; i < DIM; i++) {
        double abs_err = std::fabs((double)y_hw[i] - (double)y_ref[i]);
        double denom   = std::fabs((double)y_ref[i]) > 1e-6 ? std::fabs((double)y_ref[i]) : 1e-6;
        double rel_err = abs_err / denom;

        max_abs_err = std::max(max_abs_err, abs_err);
        max_rel_err = std::max(max_rel_err, rel_err);
        sum_abs_err += abs_err;

        if (abs_err > ABS_TOL && rel_err > REL_TOL) {
            fail_cnt++;
            if (fail_cnt <= 10) {
                printf("MISMATCH[%d]: hw=%f ref=%f abs_err=%f rel_err=%f\n",
                       i, (double)y_hw[i], (double)y_ref[i], abs_err, rel_err);
            }
        }
    }

    printf("=== matmul_fp16 (full fp16 accum) testbench (DIM=%d) ===\n", DIM);
    printf("max_abs_err = %f\n", max_abs_err);
    printf("max_rel_err = %f\n", max_rel_err);
    printf("mean_abs_err = %f\n", sum_abs_err / DIM);
    printf("fail_cnt = %d / %d\n", fail_cnt, DIM);

    if (fail_cnt == 0) {
        printf("TEST PASSED\n");
        return 0;
    } else {
        printf("TEST FAILED\n");
        return 1;
    }
}
