#include <hls_stream.h>
#include <cstdint>

#define DIM     1024
#define TILE    32
#define NTILES  (DIM / TILE)   // 16, fixed at compile time

typedef float data_t;
typedef float acc_t;

struct row_chunk_t {
    data_t v[TILE];
};

void load_rows(const data_t* __restrict__ A, hls::stream<row_chunk_t> &s_row) {
#pragma HLS INLINE off
    ROWS_LOAD:
    for (int i = 0; i < DIM; i++) {
        PACK_TILES:
        for (int k = 0; k < NTILES; k++) {
            row_chunk_t chunk;
#pragma HLS ARRAY_PARTITION variable=chunk complete
            PACK_TILE:
            for (int t = 0; t < TILE; t++) {
#pragma HLS PIPELINE II=1
                chunk.v[t] = A[i * DIM + k * TILE + t];
            }
            s_row.write(chunk);
        }
    }
}

void mul_rows(hls::stream<row_chunk_t> &s_row, const data_t x_buf[DIM],
              hls::stream<row_chunk_t> &s_mul) {
#pragma HLS INLINE off
    ROWS_MUL:
    for (int i = 0; i < DIM; i++) {
        MUL:
        for (int k = 0; k < NTILES; k++) {
#pragma HLS PIPELINE II=1
            row_chunk_t chunk = s_row.read();
            row_chunk_t prod;
#pragma HLS ARRAY_PARTITION variable=prod complete
            MUL_TILE:
            for (int t = 0; t < TILE; t++) {
#pragma HLS UNROLL
                prod.v[t] = chunk.v[t] * x_buf[k * TILE + t];
            }
            s_mul.write(prod);
        }
    }
}

void reduce_rows(hls::stream<row_chunk_t> &s_mul, hls::stream<acc_t> &s_total) {
#pragma HLS INLINE off
    ROWS_REDUCE:
    for (int i = 0; i < DIM; i++) {
        acc_t partials[NTILES];
#pragma HLS ARRAY_PARTITION variable=partials complete

        REDUCE:
        for (int k = 0; k < NTILES; k++) {
#pragma HLS PIPELINE II=1
            row_chunk_t prod = s_mul.read();

            acc_t stage1[TILE/2];
#pragma HLS ARRAY_PARTITION variable=stage1 complete
            RED1: for (int t = 0; t < TILE/2; t++) {
#pragma HLS UNROLL
                stage1[t] = prod.v[t] + prod.v[t + TILE/2];
            }

            acc_t stage2[TILE/4];
#pragma HLS ARRAY_PARTITION variable=stage2 complete
            RED2: for (int t = 0; t < TILE/4; t++) {
#pragma HLS UNROLL
                stage2[t] = stage1[t] + stage1[t + TILE/4];
            }

            acc_t total2 = 0;
            for (int q = 0; q < TILE/4; q++) {
#pragma HLS UNROLL
                total2 += stage2[q];
            }
            partials[k] = total2;
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

extern "C" void matmul(
    const data_t* __restrict__ A,
    const data_t* __restrict__ x,
    acc_t*        __restrict__ y
) {
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 depth=(DIM*DIM) max_read_burst_length=8
#pragma HLS INTERFACE m_axi port=x offset=slave bundle=gmem1 depth=DIM max_read_burst_length=8
#pragma HLS INTERFACE m_axi port=y offset=slave bundle=gmem2 depth=DIM max_write_burst_length=8
#pragma HLS INTERFACE s_axilite port=A      bundle=control
#pragma HLS INTERFACE s_axilite port=x      bundle=control
#pragma HLS INTERFACE s_axilite port=y      bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    data_t x_buf[DIM];
#pragma HLS ARRAY_PARTITION variable=x_buf cyclic factor=TILE dim=1
    LOAD_X:
    for (int k = 0; k < DIM; k++) {
#pragma HLS PIPELINE II=1
        x_buf[k] = x[k];
    }

#pragma HLS DATAFLOW
    hls::stream<row_chunk_t> s_row("s_row");
    hls::stream<row_chunk_t> s_mul("s_mul");
    hls::stream<acc_t>       s_total("s_total");
#pragma HLS STREAM variable=s_row   depth=64
#pragma HLS STREAM variable=s_mul   depth=64
#pragma HLS STREAM variable=s_total depth=64

    load_rows(A, s_row);
    mul_rows(s_row, x_buf, s_mul);
    reduce_rows(s_mul, s_total);
    store_rows(s_total, y);
}
