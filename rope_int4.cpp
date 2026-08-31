// rope_int4.cpp — RoPE int4 symetrique, layout rotate-half (comme gen_int4.py)
// v2 : burst_load_int4 simplifie — plus de buffers even/odd ni de branchement runtime.
// Les 4 premiers mots (64 nibbles) = x_even, les 4 derniers = x_odd (rotate-half) :
// on ecrit donc directement dans le bon stream selon l'indice de mot (connu a la compilation),
// au lieu de bufferiser 128 valeurs et de brancher sur elem<HALF_DIM a l'execution.
#include <cstdint>
#include <hls_stream.h>
#include <ap_int.h>
#include "hls_half.h"
#include "/home/aziz/llm-acceleration/kernels/rope/int4/rope_tables.h"

typedef half         data_t;
typedef half         trig_t;
typedef ap_uint<64>  wide_t;

#define HEAD_DIM      128
#define HALF_DIM      64
#define MAX_SEQLEN    512
#define WORDS_IN_POS  8     // 64 bytes/position / 8 bytes-par-mot = 8 mots
#define WORDS_HALF    4     // 4 mots = 64 nibbles = HALF_DIM
#define WORDS_OUT_POS (HALF_DIM / 2)  // sortie fp16 pack4 (4 half/mot), 32 mots/position

// ─── Pack 4 half dans un mot 64 bits (sortie fp16, comme kernel fp16) ──────
static inline wide_t pack4(half a, half b, half c, half d) {
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

// ─── Dequant sequentiel des 128 nibbles d'une position → 2 streams ─────────
// Une seule boucle pipelinee (au lieu de 2 boucles even/odd separees) : une seule
// ecriture stream par cycle (vers s_even si elem<64, sinon s_odd) -> pas de
// contention FIFO, et plus de drain de pipeline entre deux appels de fonction.
static void dequant_pos_to_streams(const wide_t * __restrict__ vec_in,
                                    uint32_t word_base,   // premier mot de la position
                                    float x_scale,
                                    hls::stream<data_t> &s_even,
                                    hls::stream<data_t> &s_odd) {
#pragma HLS INLINE
    wide_t cur_w = 0;
    ELEM:
    for (int elem = 0; elem < HEAD_DIM; elem++) {
#pragma HLS PIPELINE II=1
        if ((elem & 0xF) == 0) {
#pragma HLS DEPENDENCE variable=vec_in type=inter direction=RAW false
            cur_w = vec_in[word_base + (elem >> 4)];
        }
        int nib = (elem & 0xF);
        ap_uint<8> byte_k = cur_w(8 * (nib / 2) + 7, 8 * (nib / 2));
        ap_int<4> q = (nib & 1) ? byte_k(7, 4) : byte_k(3, 0);
        half val = (half)((float)q * x_scale);
        if (elem < HALF_DIM) s_even.write(val);
        else                 s_odd .write(val);
    }
}

// ─── STAGE 1 : AXI int4 (rotate-half, scale globale) → dequant → 2 streams ─
static void burst_load_int4(const wide_t * __restrict__ vec_in,
                             float x_scale,
                             uint32_t n_pos,
                             hls::stream<data_t> &s_even,
                             hls::stream<data_t> &s_odd) {
#pragma HLS INLINE off
    BURST_P:
    for (uint32_t p = 0; p < n_pos; p++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=512
        dequant_pos_to_streams(vec_in, p * WORDS_IN_POS, x_scale, s_even, s_odd);
    }
}

// ─── STAGE 2 : RoPE compute (identique a rope.cpp, data_t = half) ──────────
static void compute_stream(hls::stream<data_t> &s_even_in,
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
            s_even_out.write((half)((float)x0 * (float)c - (float)x1 * (float)s));
            s_odd_out .write((half)((float)x0 * (float)s + (float)x1 * (float)c));
        }
    }
}

// ─── STAGE 3 : 2 streams → AXI fp16 pack4 ──────────────────────────────────
// 1 lecture par stream par cycle (comme compute_stream) : on bufferise sur
// 2 cycles (e0/o0 puis e1/o1) avant d'ecrire le mot 64 bits, au lieu de lire
// 4 elements/cycle (2x s_even + 2x s_odd) qui sature le port unique du FIFO.
static void burst_store_half(hls::stream<data_t> &s_even,
                              hls::stream<data_t> &s_odd,
                              wide_t * __restrict__ vec_out,
                              uint32_t n_pos) {
#pragma HLS INLINE off
    STORE_P:
    for (uint32_t p = 0; p < n_pos; p++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=512
        half e_buf = 0, o_buf = 0;
        STORE_I:
        for (int i = 0; i < HALF_DIM; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=vec_out type=inter direction=WAW false
            half e = s_even.read();
            half o = s_odd .read();
            if (i & 1) {
                vec_out[p * WORDS_OUT_POS + (i >> 1)] = pack4(e_buf, e, o_buf, o);
            } else {
                e_buf = e;
                o_buf = o;
            }
        }
    }
}

// ─── TOP LEVEL ───────────────────────────────────────────────────────────────
extern "C" void rope_int4(
    const wide_t* __restrict__ vec_in,   // packed int4 rotate-half, 2 nibbles/byte, 8 mots/position
    wide_t*       __restrict__ vec_out,  // fp16 (half), pack4, comme kernel fp16
    uint32_t N,
    float x_scale                        // scale globale symetrique (gen_int4.py *_scale)
) {
#pragma HLS INTERFACE m_axi port=vec_in  offset=slave bundle=gmem0 depth=(MAX_SEQLEN*WORDS_IN_POS)  max_read_burst_length=64  latency=4
#pragma HLS INTERFACE m_axi port=vec_out offset=slave bundle=gmem1 depth=(MAX_SEQLEN*WORDS_OUT_POS) max_write_burst_length=64 latency=4
#pragma HLS INTERFACE s_axilite port=vec_in   bundle=control
#pragma HLS INTERFACE s_axilite port=vec_out  bundle=control
#pragma HLS INTERFACE s_axilite port=N        bundle=control
#pragma HLS INTERFACE s_axilite port=x_scale  bundle=control
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

    burst_load_int4 (vec_in, x_scale, n_pos, s_even_in, s_odd_in);
    compute_stream   (s_even_in, s_odd_in, s_even_out, s_odd_out, n_pos);
    burst_store_half (s_even_out, s_odd_out, vec_out, n_pos);
}
