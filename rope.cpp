// rope.cpp
#include <cstdint>
#include <hls_stream.h>
#include <ap_int.h>

typedef float    data_t;
typedef float    trig_t;
typedef ap_uint<64> wide_t;  // 2 floats packés

#define HEAD_DIM   64
#define HALF_DIM   32
#define MAX_SEQLEN 512

#include "rope_tables.h"

// Unpack 2 floats depuis un mot 64 bits
static inline void unpack(wide_t w, data_t &a, data_t &b) {
#pragma HLS INLINE
    ap_uint<32> lo = w(31,  0);
    ap_uint<32> hi = w(63, 32);
    a = *reinterpret_cast<float*>(&lo);
    b = *reinterpret_cast<float*>(&hi);
}

// Pack 2 floats dans un mot 64 bits
static inline wide_t pack(data_t a, data_t b) {
#pragma HLS INLINE
    ap_uint<32> lo = *reinterpret_cast<ap_uint<32>*>(&a);
    ap_uint<32> hi = *reinterpret_cast<ap_uint<32>*>(&b);
    wide_t w;
    w(31,  0) = lo;
    w(63, 32) = hi;
    return w;
}

// ─── STAGE 1 : AXI 64 bits → 2 streams (32 cy/pos) ──────────────────────────
void burst_load(const wide_t * __restrict__ vec_in,
                uint32_t n_pos,
                hls::stream<data_t> &s_even,
                hls::stream<data_t> &s_odd) {
#pragma HLS INLINE off
    BURST_P:
    for (uint32_t p = 0; p < n_pos; p++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=512
        BURST_I:
        for (int i = 0; i < HALF_DIM; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=vec_in type=inter direction=RAW false
            data_t a, b;
            unpack(vec_in[p * HALF_DIM + i], a, b);
            s_even.write(a);
            s_odd .write(b);
        }
    }
}

// ─── STAGE 2 : RoPE compute (32 cy/pos) ──────────────────────────────────────
void compute_stream(hls::stream<data_t> &s_even_in,
                    hls::stream<data_t> &s_odd_in,
                    hls::stream<data_t> &s_even_out,
                    hls::stream<data_t> &s_odd_out,
                    uint32_t n_pos) {
#pragma HLS INLINE off
    COMPUTE_P:
    for (uint32_t p = 0; p < n_pos; p++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=512
        FUSED:
        for (int i = 0; i < HALF_DIM; i++) {
#pragma HLS PIPELINE II=1
            data_t x0 = s_even_in.read();
            data_t x1 = s_odd_in .read();
            trig_t c  = cos_table[p][i];
            trig_t s  = sin_table[p][i];
            s_even_out.write(x0 * c - x1 * s);
            s_odd_out .write(x0 * s + x1 * c);
        }
    }
}

// ─── STAGE 3 : 2 streams → AXI 64 bits (32 cy/pos) ──────────────────────────
void burst_store(hls::stream<data_t> &s_even,
                 hls::stream<data_t> &s_odd,
                 wide_t * __restrict__ vec_out,
                 uint32_t n_pos) {
#pragma HLS INLINE off
    STORE_P:
    for (uint32_t p = 0; p < n_pos; p++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=512
        STORE_I:
        for (int i = 0; i < HALF_DIM; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=vec_out type=inter direction=WAW false
            vec_out[p * HALF_DIM + i] = pack(s_even.read(), s_odd.read());
        }
    }
}

// ─── TOP LEVEL ────────────────────────────────────────────────────────────────
extern "C" void rope(
    const wide_t * __restrict__ vec_in,   // AXI 64 bits
    wide_t       * __restrict__ vec_out,
    uint32_t                    N
) {
#pragma HLS INTERFACE m_axi port=vec_in  offset=slave bundle=gmem0 \
    depth=16384 max_read_burst_length=64  latency=4
#pragma HLS INTERFACE m_axi port=vec_out offset=slave bundle=gmem1 \
    depth=16384 max_write_burst_length=64 latency=4
#pragma HLS INTERFACE s_axilite port=vec_in   bundle=control
#pragma HLS INTERFACE s_axilite port=vec_out  bundle=control
#pragma HLS INTERFACE s_axilite port=N        bundle=control
#pragma HLS INTERFACE s_axilite port=return   bundle=control

#pragma HLS ARRAY_PARTITION variable=cos_table cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=sin_table cyclic factor=8 dim=2
#pragma HLS BIND_STORAGE    variable=cos_table type=ROM_2P impl=BRAM
#pragma HLS BIND_STORAGE    variable=sin_table type=ROM_2P impl=BRAM

#pragma HLS DATAFLOW

     hls::stream<data_t> s_even_in,  s_odd_in;
     hls::stream<data_t> s_even_out, s_odd_out;
#pragma HLS STREAM variable=s_even_in  depth=HALF_DIM
#pragma HLS STREAM variable=s_odd_in   depth=HALF_DIM
#pragma HLS STREAM variable=s_even_out depth=HALF_DIM
#pragma HLS STREAM variable=s_odd_out  depth=HALF_DIM

    uint32_t n_pos = (N <= MAX_SEQLEN) ? N : MAX_SEQLEN;

    burst_load    (vec_in,  n_pos, s_even_in, s_odd_in);
    compute_stream(s_even_in, s_odd_in, s_even_out, s_odd_out, n_pos);
    burst_store   (s_even_out, s_odd_out, vec_out, n_pos);
}
