#include <hls_stream.h>
#include <hls_half.h>
#include <ap_fixed.h>
#include <cstdint>

#define DIM     1024
#define TILE    32
#define NTILES  (DIM / TILE)   // 16, fixed at compile time
#define ROW_WORDS (DIM / 8)    // 128 32-bit words per row (8 nibbles/word)
#define TOTAL_SCALES (DIM * NTILES)

typedef half     data_t;   // activations, as stored in DRAM
typedef half     acc_t;    // final output type
typedef half     scale_t;
typedef int8_t   nib_t;    // holds decoded int4 value [-8,7]
typedef uint32_t pack_t;   // 8 nibbles packed per 32-bit AXI word

// fixed-point activation: converted once from half to fixed at kernel entry.
typedef ap_fixed<16,2>  xfix_t;
// fixed-point product: nib_t[-8,7] * xfix_t[-2,2) -> fits in 8 int bits
typedef ap_fixed<20,8>  prod_t;
// fixed-point accumulator for the reduction tree
typedef ap_fixed<24,10> iacc_t;

// row_chunk_t carries the RAW decoded int4 values (no scale applied yet)
struct row_chunk_t {
    nib_t v[TILE];
};

void load_rows(const pack_t* __restrict__ A_packed,
               hls::stream<row_chunk_t> &s_row) {
#pragma HLS INLINE off
    ROWS_LOAD:
    for (int i = 0; i < DIM; i++) {

        row_chunk_t chunk;
#pragma HLS ARRAY_PARTITION variable=chunk complete

        int t = 0;

        DECODE_WORDS:
        for (int w = 0; w < ROW_WORDS; w++) {
#pragma HLS PIPELINE II=1
            // one 32-bit AXI beat = 8 packed nibbles, native bus width
            pack_t word = A_packed[i * ROW_WORDS + w];

            UNPACK_NIBBLES:
            for (int n = 0; n < 8; n++) {
#pragma HLS UNROLL
                uint8_t nib_u = (word >> (n * 4)) & 0xF;
                nib_t nib = (nib_t)(nib_u << 4) >> 4;  // sign-extend 4->8 bit
                chunk.v[t + n] = nib;
            }

            t += 8;
            if (t == TILE) {
                s_row.write(chunk);
                t = 0;
            }
        }
    }
}

// prod_chunk_t carries fixed-point products: nib_t (int4) * xfix_t.
struct prod_chunk_t {
    prod_t v[TILE];
};

void mul_rows(hls::stream<row_chunk_t> &s_row, const xfix_t x_buf[DIM],
              hls::stream<prod_chunk_t> &s_mul) {
#pragma HLS INLINE off
    ROWS_MUL:
    for (int i = 0; i < DIM; i++) {
        MUL:
        for (int k = 0; k < NTILES; k++) {
#pragma HLS PIPELINE II=1
            row_chunk_t chunk = s_row.read();
            prod_chunk_t prod;
#pragma HLS ARRAY_PARTITION variable=prod complete
            MUL_TILE:
            for (int t = 0; t < TILE; t++) {
#pragma HLS UNROLL
                prod.v[t] = (prod_t)chunk.v[t] * x_buf[k * TILE + t];
            }
            s_mul.write(prod);
        }
    }
}

void reduce_rows(hls::stream<prod_chunk_t> &s_mul,
                  const scale_t scales_buf[TOTAL_SCALES],
                  hls::stream<acc_t> &s_total) {
#pragma HLS INLINE off
    ROWS_REDUCE:
    for (int i = 0; i < DIM; i++) {
        acc_t partials[NTILES];
#pragma HLS ARRAY_PARTITION variable=partials complete

        REDUCE:
        for (int k = 0; k < NTILES; k++) {
#pragma HLS PIPELINE II=1
            prod_chunk_t prod = s_mul.read();

            iacc_t stage1[TILE/2];
#pragma HLS ARRAY_PARTITION variable=stage1 complete
            RED1: for (int t = 0; t < TILE/2; t++) {
#pragma HLS UNROLL
                stage1[t] = (iacc_t)prod.v[t] + (iacc_t)prod.v[t + TILE/2];
            }

            iacc_t stage2[TILE/4];
#pragma HLS ARRAY_PARTITION variable=stage2 complete
            RED2: for (int t = 0; t < TILE/4; t++) {
#pragma HLS UNROLL
                stage2[t] = stage1[t] + stage1[t + TILE/4];
            }

            iacc_t total2 = 0;
            for (int q = 0; q < TILE/4; q++) {
#pragma HLS UNROLL
                total2 += stage2[q];
            }
            partials[k] = (acc_t)total2 * scales_buf[i * NTILES + k];
        }

        for (int stride = NTILES / 2; stride >= 1; stride >>= 1) {
#pragma HLS PIPELINE off
            for (int t = 0; t < NTILES; t++) {
#pragma HLS UNROLL
                if (t < stride && (t + stride) < NTILES) {
                    partials[t] += partials[t + stride];
                }
            }
        }
        s_total.write(partials[0]);
    }
}

void store_rows(hls::stream<acc_t> &s_total, acc_t* __restrict__ y) {
#pragma HLS INLINE off
    STORE:
    for (int i = 0; i < DIM; i++) {
#pragma HLS PIPELINE II=1
        y[i] = s_total.read();
    }
}

extern "C" void matmul_int4(
    const pack_t*  __restrict__ A_packed,   // packed 8 nibbles per 32-bit word
    const scale_t* __restrict__ scales,
    const data_t*  __restrict__ x,
    acc_t*         __restrict__ y
) {
#pragma HLS INTERFACE m_axi port=A_packed offset=slave bundle=gmem0 depth=(DIM*ROW_WORDS) max_read_burst_length=16
#pragma HLS INTERFACE m_axi port=scales   offset=slave bundle=gmem1 depth=(DIM*NTILES) max_read_burst_length=16
#pragma HLS INTERFACE m_axi port=x        offset=slave bundle=gmem2 depth=DIM max_read_burst_length=16
#pragma HLS INTERFACE m_axi port=y        offset=slave bundle=gmem3 depth=DIM max_write_burst_length=16
#pragma HLS INTERFACE s_axilite port=A_packed bundle=control
#pragma HLS INTERFACE s_axilite port=scales   bundle=control
#pragma HLS INTERFACE s_axilite port=x        bundle=control
#pragma HLS INTERFACE s_axilite port=y        bundle=control
#pragma HLS INTERFACE s_axilite port=return   bundle=control

    xfix_t x_buf[DIM];
#pragma HLS ARRAY_PARTITION variable=x_buf cyclic factor=TILE dim=1
    LOAD_X:
    for (int k = 0; k < DIM; k++) {
#pragma HLS PIPELINE II=1
        x_buf[k] = (xfix_t)x[k];
    }

    static scale_t scales_buf[TOTAL_SCALES];
    LOAD_SCALES:
    for (int s = 0; s < TOTAL_SCALES; s++) {
#pragma HLS PIPELINE II=1
        scales_buf[s] = scales[s];
    }

#pragma HLS DATAFLOW
    hls::stream<row_chunk_t>  s_row("s_row");
    hls::stream<prod_chunk_t> s_mul("s_mul");
    hls::stream<acc_t>        s_total("s_total");
#pragma HLS STREAM variable=s_row   depth=64
#pragma HLS STREAM variable=s_mul   depth=64
#pragma HLS STREAM variable=s_total depth=64

    load_rows(A_packed, s_row);
    mul_rows(s_row, x_buf, s_mul);
    reduce_rows(s_mul, scales_buf, s_total);
    store_rows(s_total, y);
}
