#include <hls_math.h>
#include <ap_int.h>
#include <ap_fixed.h>
#include <string.h>
#include <cstdint>

typedef ap_fixed<16, 5, AP_TRN, AP_SAT>  fixed_t;
typedef ap_fixed<24, 8, AP_TRN, AP_SAT>  acc_t;
typedef float data_t;

#define LUT_SIZE 256
#define LUT_MIN  -8
#define LUT_MAX   8
#define TILE      1024
#define TILE_PACKED (TILE / 2)
#define NTILES_MAX 64

static const fixed_t sigmoid_lut[LUT_SIZE] = {
#include "sigmoid_lut.h"
};

inline fixed_t sigmoid_lookup(fixed_t x) {
#pragma HLS INLINE
    ap_fixed<16,5> shifted = x - (fixed_t)LUT_MIN;
    int idx = (int)(shifted * (fixed_t)((float)LUT_SIZE / (LUT_MAX - LUT_MIN)));
    if (idx < 0) idx = 0;
    else if (idx > LUT_SIZE - 1) idx = LUT_SIZE - 1;
    return sigmoid_lut[idx];
}

inline void unpack_int4_pair(int8_t byte, fixed_t scale, fixed_t &lo, fixed_t &hi) {
#pragma HLS INLINE
    ap_int<4> q_lo = ((uint8_t)byte) & 0xF;
    ap_int<4> q_hi = (((uint8_t)byte) >> 4) & 0xF;
    lo = (fixed_t)q_lo * scale;
    hi = (fixed_t)q_hi * scale;
}

// =====================================================
// Sous-fonctions separees pour DATAFLOW (load / compute / store)
// meme structure que la reference fp32
// =====================================================
void load_tile(const int8_t *gate_packed, const int8_t *up_packed, int base_packed, int size_packed,
               int8_t gate_buf[TILE_PACKED], int8_t up_buf[TILE_PACKED]) {
#pragma HLS INLINE off
#pragma HLS LOOP_TRIPCOUNT min=1 max=512
    memcpy(gate_buf, &gate_packed[base_packed], size_packed * sizeof(int8_t));
    memcpy(up_buf,   &up_packed[base_packed],   size_packed * sizeof(int8_t));
}

void compute_tile(int8_t gate_buf[TILE_PACKED], int8_t up_buf[TILE_PACKED],
                   data_t out_buf[TILE], int size_packed,
                   float gate_scale, float up_scale) {
#pragma HLS INLINE off
    fixed_t gate_scale_fx = (fixed_t)gate_scale;
    fixed_t up_scale_fx   = (fixed_t)up_scale;

COMPUTE:
    for (int i = 0; i < size_packed; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=512
#pragma HLS PIPELINE II=1
        fixed_t g_lo, g_hi, u_lo, u_hi;
        unpack_int4_pair(gate_buf[i], gate_scale_fx, g_lo, g_hi);
        unpack_int4_pair(up_buf[i],   up_scale_fx,   u_lo, u_hi);

        fixed_t s_lo = sigmoid_lookup(g_lo);
        fixed_t s_hi = sigmoid_lookup(g_hi);

        // Etage intermediaire — casse le chemin combinatoire en 2 registres
        acc_t partial_lo = (acc_t)g_lo * (acc_t)s_lo;
        acc_t partial_hi = (acc_t)g_hi * (acc_t)s_hi;

        // Aucun BIND_OP — HLS choisit librement DSP ou fabric pour tenir le timing
        acc_t tmp_lo = partial_lo * (acc_t)u_lo;
        acc_t tmp_hi = partial_hi * (acc_t)u_hi;

        out_buf[2*i]   = (data_t)tmp_lo.to_float();
        out_buf[2*i+1] = (data_t)tmp_hi.to_float();
    }
}

void store_tile(data_t *out, int base, int size, data_t out_buf[TILE]) {
#pragma HLS INLINE off
STORE_LOOP:
    for (int i = 0; i < size; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=1024
#pragma HLS PIPELINE II=1
        out[base + i] = out_buf[i];
    }
}

// =====================================================
// SILU KERNEL INT4 — meme structure double-buffered / DATAFLOW que la reference fp32
// gate/up entrent quantifies int4 (packes 2/byte) ; sortie reste en float
// =====================================================
extern "C" void silu_int4(const int8_t *gate_packed,
               const int8_t *up_packed,
               data_t       *out,
               int           n,
               float         gate_scale,
               float         up_scale)
{
#pragma HLS INTERFACE m_axi port=gate_packed offset=slave bundle=gmem0 depth=524288
#pragma HLS INTERFACE m_axi port=up_packed   offset=slave bundle=gmem1 depth=524288
#pragma HLS INTERFACE m_axi port=out         offset=slave bundle=gmem2 depth=1048576
#pragma HLS INTERFACE s_axilite port=gate_packed bundle=CTRL
#pragma HLS INTERFACE s_axilite port=up_packed   bundle=CTRL
#pragma HLS INTERFACE s_axilite port=out         bundle=CTRL
#pragma HLS INTERFACE s_axilite port=n           bundle=CTRL
#pragma HLS INTERFACE s_axilite port=gate_scale  bundle=CTRL
#pragma HLS INTERFACE s_axilite port=up_scale    bundle=CTRL
#pragma HLS INTERFACE s_axilite port=return      bundle=CTRL
#pragma HLS BIND_STORAGE variable=sigmoid_lut type=ROM_2P impl=bram

    static int8_t gate_buf[TILE_PACKED], up_buf[TILE_PACKED];
    static data_t out_buf[TILE];

    int ntiles = (n + TILE - 1) / TILE;
TILE_LOOP:
    for (int t = 0; t < ntiles; t++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=NTILES_MAX
#pragma HLS DATAFLOW
        int base        = t * TILE;
        int base_packed = t * TILE_PACKED;
        int remaining   = n - base;
        int clip        = (remaining < TILE) ? remaining : TILE;
        int size        = (clip > 0) ? clip : 0;
        int size_packed = (size + 1) / 2;

        load_tile(gate_packed, up_packed, base_packed, size_packed, gate_buf, up_buf);
        compute_tile(gate_buf, up_buf, out_buf, size_packed, gate_scale, up_scale);
        store_tile(out, base, size, out_buf);
    }
}
