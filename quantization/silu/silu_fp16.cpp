#include <hls_math.h>
#include <hls_half.h>
#include <string.h>
typedef half data_t;
typedef half acc_t;
#define LUT_SIZE 256
#define LUT_MIN  -8.0f
#define LUT_MAX   8.0f
#define TILE     1024
#define NTILES_MAX 1024   // borne haute pour TOTAL_LEN/TILE (ajuste si besoin)
// =====================================================
// LUT (BRAM safe)
// =====================================================
data_t sigmoid_lut[LUT_SIZE] = {
#include "sigmoid_lut.h"
};
inline data_t sigmoid_lookup(data_t x) {
#pragma HLS INLINE
    float xf = (float)x;
    int idx = (int)((xf - LUT_MIN) * (float)(LUT_SIZE / (LUT_MAX - LUT_MIN)));
    if (idx < 0)
        idx = 0;
    else if (idx > LUT_SIZE - 1)
        idx = LUT_SIZE - 1;
    return sigmoid_lut[idx];
}
// =====================================================
// Sous-fonctions séparées pour DATAFLOW (load / compute / store)
// =====================================================
void load_tile(const data_t *gate, const data_t *up, int base, int size,
               data_t gate_buf[TILE], data_t up_buf[TILE]) {
#pragma HLS INLINE off
    memcpy(gate_buf, &gate[base], size * sizeof(data_t));
    memcpy(up_buf,   &up[base],   size * sizeof(data_t));
}
void compute_tile(data_t gate_buf[TILE], data_t up_buf[TILE],
                   data_t out_buf[TILE], int size) {
#pragma HLS INLINE off
COMPUTE:
    for (int i = 0; i < size; i++) {
#pragma HLS PIPELINE II=1
        data_t g = gate_buf[i];
        data_t u = up_buf[i];
        data_t s = sigmoid_lookup(g);
        acc_t tmp = g * s * u;
        out_buf[i] = (data_t)tmp;
    }
}
void store_tile(data_t *out, int base, int size, data_t out_buf[TILE]) {
#pragma HLS INLINE off
    memcpy(&out[base], out_buf, size * sizeof(data_t));
}
// =====================================================
// SILU KERNEL — double buffered, DATAFLOW across tiles
// =====================================================
void silu(const data_t *gate,
          const data_t *up,
          data_t *out,
          int n)
{
#pragma HLS INTERFACE m_axi port=gate offset=slave bundle=gmem0 depth=1024
#pragma HLS INTERFACE m_axi port=up   offset=slave bundle=gmem1 depth=1024
#pragma HLS INTERFACE m_axi port=out  offset=slave bundle=gmem2 depth=1024
#pragma HLS INTERFACE s_axilite port=gate bundle=CTRL
#pragma HLS INTERFACE s_axilite port=up   bundle=CTRL
#pragma HLS INTERFACE s_axilite port=out  bundle=CTRL
#pragma HLS INTERFACE s_axilite port=n    bundle=CTRL
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL
#pragma HLS ARRAY_PARTITION variable=sigmoid_lut cyclic factor=4 dim=1
    // Buffers simples — DATAFLOW pipeline automatiquement les itérations successives
    static data_t gate_buf[TILE], up_buf[TILE], out_buf[TILE];
    int ntiles = (n + TILE - 1) / TILE;
TILE_LOOP:
    for (int t = 0; t < ntiles; t++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=NTILES_MAX
#pragma HLS DATAFLOW
        int base = t * TILE;
        int remaining = n - base;
        int clip = (remaining < TILE) ? remaining : TILE;
        int size = (clip > 0) ? clip : 0;
        load_tile(gate, up, base, size, gate_buf, up_buf);
        compute_tile(gate_buf, up_buf, out_buf, size);
        store_tile(out, base, size, out_buf);
    }
}
