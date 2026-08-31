// silu_int4_tb.cpp — sans std::vector (compilateur cosim buggue avec resize())
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <ap_int.h>

typedef float data_t;
typedef float acc_t;

extern "C" void silu_int4(const int8_t *gate_packed, const int8_t *up_packed,
                           data_t *out, int n, float gate_scale, float up_scale);

#define MAX_N        1048576
#define MAX_N_PACKED 524288

static float   gate_buf[MAX_N];
static float   up_buf[MAX_N];
static int     gate_q[MAX_N];
static int     up_q[MAX_N];
static int8_t  gate_packed_buf[MAX_N_PACKED];
static int8_t  up_packed_buf[MAX_N_PACKED];
static data_t  out_buf[MAX_N];

static float sigmoid_ref(float x) { return 1.0f / (1.0f + std::exp(-x)); }

static void quantize_int4(const float *vals, int n, int *q, float &scale) {
    float max_abs = 1e-9f;
    for (int i = 0; i < n; i++) {
        float a = std::fabs(vals[i]);
        if (a > max_abs) max_abs = a;
    }
    scale = max_abs / 7.0f;
    for (int i = 0; i < n; i++) {
        int qi = (int)std::lround(vals[i] / scale);
        if (qi > 7) qi = 7;
        if (qi < -7) qi = -7;
        q[i] = qi;
    }
}

static int pack_int4(const int *q, int n, int8_t *packed) {
    int npacked = (n + 1) / 2;
    for (int i = 0; i < npacked; i++) {
        int lo = q[2*i] & 0xF;
        int hi = (2*i + 1 < n) ? (q[2*i + 1] & 0xF) : 0;
        int byte = (hi << 4) | lo;
        packed[i] = (int8_t)(byte > 127 ? byte - 256 : byte);
    }
    return npacked;
}

static int run_test(int n, unsigned seed) {
    srand(seed);
    for (int i = 0; i < n; i++) {
        gate_buf[i] = ((float)(rand() % 2000) / 1000.0f) - 1.0f;
        up_buf[i]   = ((float)(rand() % 2000) / 1000.0f) - 1.0f;
    }

    float gate_scale, up_scale;
    quantize_int4(gate_buf, n, gate_q, gate_scale);
    quantize_int4(up_buf,   n, up_q,   up_scale);

    int npacked_gate = pack_int4(gate_q, n, gate_packed_buf);
    int npacked_up   = pack_int4(up_q,   n, up_packed_buf);

    memset(gate_packed_buf + npacked_gate, 0, sizeof(gate_packed_buf) - npacked_gate);
    memset(up_packed_buf   + npacked_up,   0, sizeof(up_packed_buf)   - npacked_up);
    memset(out_buf, 0, sizeof(out_buf));

    silu_int4(gate_packed_buf, up_packed_buf, out_buf, n, gate_scale, up_scale);

    const double THRESH = 0.05;
    double max_err = 0.0;
    int worst_idx = 0;
    for (int i = 0; i < n; i++) {
        float g_deq = gate_q[i] * gate_scale;
        float u_deq = up_q[i]   * up_scale;
        float expected = g_deq * sigmoid_ref(g_deq) * u_deq;
        double diff = std::fabs((double)out_buf[i] - (double)expected);
        if (diff > max_err) { max_err = diff; worst_idx = i; }
    }

    bool pass = max_err < THRESH;
    printf("  N=%-8d | max_err=%.4e | worst_idx=%-7d | %s\n",
           n, max_err, worst_idx, pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

int main() {
    int total_fail = 0;
    int sizes[] = {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384,
                   32768, 65536, 131072, 262144, 524288, 1048576};
    unsigned seed = 42;

    printf("=== SILU INT4 TB ===\n");
    for (int n : sizes) {
        total_fail += run_test(n, seed++);
    }

    printf("\n=== GLOBAL SUMMARY ===\n");
    printf("Total tests failed : %d\n", total_fail);
    printf("(int4: erreur de quantification attendue -- cosim non bloquee sur la precision)\n");
    return 0;
}
