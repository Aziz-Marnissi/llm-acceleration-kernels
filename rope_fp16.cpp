#include <cstdint>
#include <ap_int.h>
#include "hls_half.h"
#include "/home/aziz/llm-acceleration/kernels/rope/fp16/rope_fp16/rope_tables.h"
typedef half         data_t;
typedef ap_uint<64>  wide_t;

#define HEAD_DIM   128
#define HALF_DIM   64
#define MAX_SEQLEN 512
#define WORDS_PER_POS (HALF_DIM / 2)   // 32

static half get16(wide_t w, int lane) {
#pragma HLS INLINE
    ap_uint<16> v;
    switch (lane) {
        case 0: v = w(15, 0);  break;
        case 1: v = w(31, 16); break;
        case 2: v = w(47, 32); break;
        default: v = w(63, 48); break;
    }
    return *reinterpret_cast<half*>(&v);
}

static wide_t pack4(half a, half b, half c, half d) {
#pragma HLS INLINE
    ap_uint<16> w0 = *reinterpret_cast<ap_uint<16>*>(&a);
    ap_uint<16> w1 = *reinterpret_cast<ap_uint<16>*>(&b);
    ap_uint<16> w2 = *reinterpret_cast<ap_uint<16>*>(&c);
    ap_uint<16> w3 = *reinterpret_cast<ap_uint<16>*>(&d);
    wide_t w;
    w(15, 0)  = w0;
    w(31, 16) = w1;
    w(47, 32) = w2;
    w(63, 48) = w3;
    return w;
}

extern "C" void rope(
    const wide_t* __restrict__ vec_in,
    wide_t*       __restrict__ vec_out,
    uint32_t N
) {
#pragma HLS INTERFACE m_axi port=vec_in  offset=slave bundle=gmem0 depth=(MAX_SEQLEN*WORDS_PER_POS) max_read_burst_length=64
#pragma HLS INTERFACE m_axi port=vec_out offset=slave bundle=gmem1 depth=(MAX_SEQLEN*WORDS_PER_POS) max_write_burst_length=64
#pragma HLS INTERFACE s_axilite port=vec_in  bundle=control
#pragma HLS INTERFACE s_axilite port=vec_out bundle=control
#pragma HLS INTERFACE s_axilite port=N       bundle=control
#pragma HLS INTERFACE s_axilite port=return  bundle=control

#pragma HLS BIND_STORAGE variable=cos_table type=ROM_2P impl=bram
#pragma HLS BIND_STORAGE variable=sin_table type=ROM_2P impl=bram

    uint32_t total_words = N * WORDS_PER_POS;
    ROPE_MAIN:
    for (uint32_t idx = 0; idx < total_words; idx++) {
#pragma HLS LOOP_TRIPCOUNT min=32 max=(MAX_SEQLEN*WORDS_PER_POS)
#pragma HLS PIPELINE II=1
        uint32_t pos = idx / WORDS_PER_POS;
        uint32_t i   = idx % WORDS_PER_POS;   // 0..31
        uint32_t i0  = 2 * i;
        uint32_t i1  = i0 + 1;

        wide_t in_w  = vec_in[idx];
        half a = get16(in_w, 0);  // x_even[2i]
        half b = get16(in_w, 1);  // x_even[2i+1]
        half c = get16(in_w, 2);  // x_odd[2i]
        half d = get16(in_w, 3);  // x_odd[2i+1]

        half cos0 = cos_table[pos][i0];
        half cos1 = cos_table[pos][i1];
        half sin0 = sin_table[pos][i0];
        half sin1 = sin_table[pos][i1];

        half y_even0 = (half)((float)a * (float)cos0 - (float)c * (float)sin0);
        half y_even1 = (half)((float)b * (float)cos1 - (float)d * (float)sin1);
        half y_odd0  = (half)((float)a * (float)sin0 + (float)c * (float)cos0);
        half y_odd1  = (half)((float)b * (float)sin1 + (float)d * (float)cos1);

        vec_out[idx] = pack4(y_even0, y_even1, y_odd0, y_odd1);
    }
}
