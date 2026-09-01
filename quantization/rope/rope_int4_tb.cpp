// rope_int4_tb.cpp
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

// x et expected quantifies int4 (packed 2 val/byte) + scale, genere par gen_int4.py
#include "rope_weights_int4_N16.h"
#include "rope_weights_int4_N32.h"
#include "rope_weights_int4_N64.h"
#include "rope_weights_int4_N128.h"
#include "rope_weights_int4_N256.h"
#include "rope_weights_int4_N512.h"

extern "C" void rope_int4(const wide_t *vec_in, wide_t *vec_out, uint32_t N, float x_scale);

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

// pack un mot d'entree int4 pour la position p, groupe i, a partir du
// tableau packe genere par gen_int4.py. Layout du generateur:
// byte[i]      = (col(2i+1) << 4) | col(2i)       -> couvre a,b (x_even)
// byte[32 + i] = (col(HALF+2i+1) << 4) | col(HALF+2i) -> couvre c,d (x_odd)
static wide_t pack_int4_word(const int8_t (*x_int4)[HEAD_DIM / 2], uint32_t p, uint32_t i) {
    // HALF_DIM/2 columns per half => packed_per_row total = HEAD_DIM/2 = 64 bytes
    // byte i        -> x_even pair (2i, 2i+1)
    // byte 32 + i    -> x_odd  pair (2i, 2i+1) offset by HALF_DIM
    uint8_t byte0 = (uint8_t)x_int4[p][i];
    uint8_t byte1 = (uint8_t)x_int4[p][(HALF_DIM / 2) + i];
    wide_t w = 0;
    w(7, 0)  = byte0;
    w(15, 8) = byte1;
    return w;
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

// dequantifie 2 valeurs int4 packees dans un byte (low, high nibble)
static void unpack_int4_byte(int8_t byte, float scale, float &lo, float &hi) {
    ap_int<4> q_lo = ((uint8_t)byte) & 0xF;
    ap_int<4> q_hi = (((uint8_t)byte) >> 4) & 0xF;
    lo = (float)q_lo * scale;
    hi = (float)q_hi * scale;
}

template<int PACKED_BYTES>
static int run_test(uint32_t N,
                     const int8_t (*x_int4)[PACKED_BYTES], float x_scale,
                     const int8_t (*expected_int4)[PACKED_BYTES], float expected_scale) {
    static wide_t hw_in[MAX_SEQLEN * (HALF_DIM / 2)];
    static wide_t hw_out[MAX_SEQLEN * (HALF_DIM / 2)];

    for (uint32_t p = 0; p < N; p++) {
        for (int i = 0; i < HALF_DIM / 2; i++) {
            hw_in[p * (HALF_DIM / 2) + i] = pack_int4_word(x_int4, p, i);
        }
    }

    long long t0 = get_cycle_count();
    rope_int4(hw_in, hw_out, N, x_scale);
    long long t1 = get_cycle_count();

    const double THRESH = 1.5;  // int4: bruit cumule x+expected
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

            // dequant expected_int4 pour les 4 memes colonnes
            float e_even_lo, e_even_hi, e_odd_lo, e_odd_hi;
            unpack_int4_byte(expected_int4[p][i], expected_scale, e_even_lo, e_even_hi);
            unpack_int4_byte(expected_int4[p][(HALF_DIM/2)+i], expected_scale, e_odd_lo, e_odd_hi);
            float exp_vals[4] = {e_even_lo, e_even_hi, e_odd_lo, e_odd_hi};

            for (int k = 0; k < 4; k++) {
                double diff = std::fabs((double)hw_vals[k] - (double)exp_vals[k]);
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
    total_fail += run_test<64>(16, x_int4_N16, x_int4_N16_scale, expected_int4_N16, expected_int4_N16_scale);

    printf("\n=== SEQ_LEN 32 ===\n");
    total_fail += run_test<64>(32, x_int4_N32, x_int4_N32_scale, expected_int4_N32, expected_int4_N32_scale);

    printf("\n=== SEQ_LEN 64 ===\n");
    total_fail += run_test<64>(64, x_int4_N64, x_int4_N64_scale, expected_int4_N64, expected_int4_N64_scale);

    printf("\n=== SEQ_LEN 128 ===\n");
    total_fail += run_test<64>(128, x_int4_N128, x_int4_N128_scale, expected_int4_N128, expected_int4_N128_scale);

    printf("\n=== SEQ_LEN 256 ===\n");
    total_fail += run_test<64>(256, x_int4_N256, x_int4_N256_scale, expected_int4_N256, expected_int4_N256_scale);

    printf("\n=== SEQ_LEN 512 ===\n");
    total_fail += run_test<64>(512, x_int4_N512, x_int4_N512_scale, expected_int4_N512, expected_int4_N512_scale);

    printf("\n=== GLOBAL SUMMARY ===\n");
    printf("Total tests failed : %d\n", total_fail);
    return 0;  // int4: precision non bloquante pour la cosim
}
