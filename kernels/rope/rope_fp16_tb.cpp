// rope_fp16_tb.cpp
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <ap_int.h>
#include "hls_half.h"

typedef half         data_t;
typedef ap_uint<64>  wide_t;

#define HEAD_DIM   128
#define HALF_DIM   64
#define MAX_SEQLEN 512

#include "rope_weights_fp16_N16.h"
#include "rope_weights_fp16_N32.h"
#include "rope_weights_fp16_N64.h"
#include "rope_weights_fp16_N128.h"
#include "rope_weights_fp16_N256.h"
#include "rope_weights_fp16_N512.h"

extern "C" void rope(const wide_t *vec_in, wide_t *vec_out, uint32_t N);

static wide_t pack4(half a, half b, half c, half d) {
    ap_uint<16> w0 = *reinterpret_cast<ap_uint<16>*>(&a);
    ap_uint<16> w1 = *reinterpret_cast<ap_uint<16>*>(&b);
    ap_uint<16> w2 = *reinterpret_cast<ap_uint<16>*>(&c);
    ap_uint<16> w3 = *reinterpret_cast<ap_uint<16>*>(&d);
    wide_t w;
    w(15, 0) = w0; w(31, 16) = w1; w(47, 32) = w2; w(63, 48) = w3;
    return w;
}

static void unpack4(wide_t w, half &a, half &b, half &c, half &d) {
    ap_uint<16> w0 = w(15, 0), w1 = w(31, 16), w2 = w(47, 32), w3 = w(63, 48);
    a = *reinterpret_cast<half*>(&w0); b = *reinterpret_cast<half*>(&w1);
    c = *reinterpret_cast<half*>(&w2); d = *reinterpret_cast<half*>(&w3);
}

static long long get_cycle_count() {
#ifdef __x86_64__
    unsigned int lo, hi;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return ((long long)hi << 32) | lo;
#else
    return 0;
#endif
}

// note: cos_arr/sin_arr toujours passés au tb pour rien casser côté headers,
// mais ne sont plus utilisés pour piloter le kernel (le kernel utilise sa LUT interne).
// Ils servent seulement si tu veux comparer avec ta propre référence recalculée.
static int run_test(uint32_t N, const half (*x_arr)[HEAD_DIM], const half (*cos_arr)[HEAD_DIM],
                    const half (*sin_arr)[HEAD_DIM], const half (*expected_arr)[HEAD_DIM]) {
    static wide_t hw_in[MAX_SEQLEN * (HALF_DIM / 2)];
    static wide_t hw_out[MAX_SEQLEN * (HALF_DIM / 2)];

    for (uint32_t p = 0; p < N; p++) {
        for (int i = 0; i < HALF_DIM / 2; i++) {
            hw_in[p * (HALF_DIM / 2) + i] = pack4(x_arr[p][2*i], x_arr[p][2*i+1], x_arr[p][HALF_DIM+2*i], x_arr[p][HALF_DIM+2*i+1]);
        }
    }

    long long t0 = get_cycle_count();
    rope(hw_in, hw_out, N);
    long long t1 = get_cycle_count();

    const double THRESH = 5e-2;
    double global_max_err = 0.0;
    int fail_count = 0;

    uint32_t check_pos[] = {0, 1, N/4, N/2, (N>1?N-1:0)};
    for (uint32_t p : check_pos) {
        if (p >= N) continue;
        double max_err = 0.0;
        int worst_i = 0;
        for (int i = 0; i < HALF_DIM / 2; i++) {
            half a, b, c, d;
            unpack4(hw_out[p * (HALF_DIM/2) + i], a, b, c, d);
            half hw_vals[4] = {a, b, c, d};
            int cols[4] = {2*i, 2*i+1, HALF_DIM+2*i, HALF_DIM+2*i+1};
            for (int k = 0; k < 4; k++) {
                double diff = std::fabs((double)hw_vals[k] - (double)expected_arr[p][cols[k]]);
                if (diff > max_err) { max_err = diff; worst_i = cols[k]; }
            }
        }
        if (max_err >= THRESH) fail_count++;
        if (max_err > global_max_err) global_max_err = max_err;
        printf("  pos=%4u | abs=%.2e | worst_col=%2d | %s\n", p, max_err, worst_i, (max_err < THRESH) ? "PASS" : "FAIL");
    }
    printf("  N=%-4u | cycles=%lld | max_err=%.2e | %s\n", N, t1 - t0, global_max_err, fail_count == 0 ? "ALL PASS" : "FAIL");
    return fail_count;
}

int main() {
    int total_fail = 0;

    printf("=== SEQ_LEN 16 ===\n");
    total_fail += run_test(16, x_fp16_N16, cos_fp16_N16, sin_fp16_N16, expected_fp16_N16);

    printf("\n=== SEQ_LEN 32 ===\n");
    total_fail += run_test(32, x_fp16_N32, cos_fp16_N32, sin_fp16_N32, expected_fp16_N32);

    printf("\n=== SEQ_LEN 64 ===\n");
    total_fail += run_test(64, x_fp16_N64, cos_fp16_N64, sin_fp16_N64, expected_fp16_N64);

    printf("\n=== SEQ_LEN 128 ===\n");
    total_fail += run_test(128, x_fp16_N128, cos_fp16_N128, sin_fp16_N128, expected_fp16_N128);

    printf("\n=== SEQ_LEN 256 ===\n");
    total_fail += run_test(256, x_fp16_N256, cos_fp16_N256, sin_fp16_N256, expected_fp16_N256);

    printf("\n=== SEQ_LEN 512 ===\n");
    total_fail += run_test(512, x_fp16_N512, cos_fp16_N512, sin_fp16_N512, expected_fp16_N512);

    printf("\n=== GLOBAL SUMMARY ===\n");
    printf("Total tests failed : %d\n", total_fail);
    return (total_fail == 0) ? 0 : 1;
}
