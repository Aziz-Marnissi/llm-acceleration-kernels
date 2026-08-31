#include <ap_int.h>
typedef float fp;
typedef unsigned int idx_t;
#define NUM_LANES 16
#define VOCAB_SIZE 32768
#define HALF (VOCAB_SIZE / 2)

void greedy_sampler(const fp *logits0, const fp *logits1, idx_t *best_idx, int size) {
#pragma HLS INTERFACE m_axi port=logits0 offset=slave bundle=gmem0 \
                      depth=16384 max_read_burst_length=8 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=logits1 offset=slave bundle=gmem1 \
                      depth=16384 max_read_burst_length=8 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=best_idx offset=slave bundle=gmem0 depth=1
#pragma HLS INTERFACE s_axilite port=logits0 bundle=CTRL
#pragma HLS INTERFACE s_axilite port=logits1 bundle=CTRL
#pragma HLS INTERFACE s_axilite port=best_idx bundle=CTRL
#pragma HLS INTERFACE s_axilite port=size bundle=CTRL
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL

#pragma HLS DEPENDENCE variable=logits0 inter false
#pragma HLS DEPENDENCE variable=logits1 inter false

    fp best_val0[NUM_LANES], best_val1[NUM_LANES];
    idx_t best_idx0[NUM_LANES], best_idx1[NUM_LANES];
#pragma HLS ARRAY_PARTITION variable=best_val0 complete
#pragma HLS ARRAY_PARTITION variable=best_val1 complete
#pragma HLS ARRAY_PARTITION variable=best_idx0 complete
#pragma HLS ARRAY_PARTITION variable=best_idx1 complete
#pragma HLS DEPENDENCE variable=best_val0 inter false
#pragma HLS DEPENDENCE variable=best_val1 inter false
#pragma HLS DEPENDENCE variable=best_idx0 inter false
#pragma HLS DEPENDENCE variable=best_idx1 inter false

INIT:
    for (int i = 0; i < NUM_LANES; i++) {
#pragma HLS UNROLL
        best_val0[i] = -3.4028235e38f;
        best_val1[i] = -3.4028235e38f;
        best_idx0[i] = 0;
        best_idx1[i] = 0;
    }

    const int half = size / 2;
    const int num_reads = half / NUM_LANES;

MAIN_LOOP:
    for (int i = 0; i < num_reads; i++) {
#pragma HLS PIPELINE II=20
#pragma HLS LOOP_TRIPCOUNT min=8 max=1024
        int base = i * NUM_LANES;
LANES:
        for (int j = 0; j < NUM_LANES; j++) {
#pragma HLS UNROLL
            fp v0 = logits0[base + j];
            fp v1 = logits1[base + j];
            if (v0 > best_val0[j]) {
                best_val0[j] = v0;
                best_idx0[j] = (idx_t)(base + j);
            }
            if (v1 > best_val1[j]) {
                best_val1[j] = v1;
                best_idx1[j] = (idx_t)(HALF + base + j);
            }
        }
    }

    fp final_best = best_val0[0];
    idx_t final_idx = best_idx0[0];
REDUCE:
    for (int i = 1; i < NUM_LANES; i++) {
#pragma HLS UNROLL
        if (best_val0[i] > final_best) { final_best = best_val0[i]; final_idx = best_idx0[i]; }
    }
    for (int i = 0; i < NUM_LANES; i++) {
#pragma HLS UNROLL
        if (best_val1[i] > final_best) { final_best = best_val1[i]; final_idx = best_idx1[i]; }
    }

    *best_idx = final_idx;
}
