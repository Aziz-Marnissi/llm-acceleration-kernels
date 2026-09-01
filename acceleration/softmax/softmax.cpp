#include "ap_fixed.h"
#include "hls_math.h"

#define SEQ_LEN  128
#define TILE      16
#define TILE_EXP   8

typedef ap_fixed<16, 8>  data_t;
typedef ap_fixed<32, 16> acc_t;
typedef ap_ufixed<16, 1> prob_t;

void softmax(data_t* in, prob_t* out)
{
#pragma HLS INTERFACE m_axi port=in  offset=slave bundle=gmem0 depth=16384 max_read_burst_length=128
#pragma HLS INTERFACE m_axi port=out offset=slave bundle=gmem1 depth=16384 max_write_burst_length=128
#pragma HLS INTERFACE s_axilite port=return

    acc_t row_buf[SEQ_LEN];
#pragma HLS BIND_STORAGE variable=row_buf type=RAM_1P impl=BRAM
#pragma HLS ARRAY_PARTITION variable=row_buf cyclic factor=TILE dim=1

    acc_t exp_buf[SEQ_LEN];
#pragma HLS BIND_STORAGE variable=exp_buf type=RAM_2P impl=BRAM
#pragma HLS ARRAY_PARTITION variable=exp_buf cyclic factor=TILE_EXP dim=1

    Rows:
    for (int i = 0; i < SEQ_LEN; i++) {

        // Passe 1 : Load
        Load:
        for (int j = 0; j < SEQ_LEN; j++) {
#pragma HLS PIPELINE II=1
            row_buf[j] = (acc_t)in[i * SEQ_LEN + j];
        }

        // Passe 2 : Max_Find
        acc_t partial_max[TILE];
#pragma HLS ARRAY_PARTITION variable=partial_max complete

        Init_Max:
        for (int t = 0; t < TILE; t++) {
#pragma HLS UNROLL
            partial_max[t] = (acc_t)-128;
        }

        Max_Find:
        for (int j = 0; j < SEQ_LEN / TILE; j++) {
#pragma HLS PIPELINE II=1
            for (int t = 0; t < TILE; t++) {
#pragma HLS UNROLL
                acc_t val = row_buf[j * TILE + t];
                if (val > partial_max[t]) partial_max[t] = val;
            }
        }

        // Tree-reduction max
        acc_t mx0 = (partial_max[0]  > partial_max[1])  ? partial_max[0]  : partial_max[1];
        acc_t mx1 = (partial_max[2]  > partial_max[3])  ? partial_max[2]  : partial_max[3];
        acc_t mx2 = (partial_max[4]  > partial_max[5])  ? partial_max[4]  : partial_max[5];
        acc_t mx3 = (partial_max[6]  > partial_max[7])  ? partial_max[6]  : partial_max[7];
        acc_t mx4 = (partial_max[8]  > partial_max[9])  ? partial_max[8]  : partial_max[9];
        acc_t mx5 = (partial_max[10] > partial_max[11]) ? partial_max[10] : partial_max[11];
        acc_t mx6 = (partial_max[12] > partial_max[13]) ? partial_max[12] : partial_max[13];
        acc_t mx7 = (partial_max[14] > partial_max[15]) ? partial_max[14] : partial_max[15];
        acc_t ma0 = (mx0 > mx1) ? mx0 : mx1;
        acc_t ma1 = (mx2 > mx3) ? mx2 : mx3;
        acc_t ma2 = (mx4 > mx5) ? mx4 : mx5;
        acc_t ma3 = (mx6 > mx7) ? mx6 : mx7;
        acc_t mb0 = (ma0 > ma1) ? ma0 : ma1;
        acc_t mb1 = (ma2 > ma3) ? ma2 : ma3;
        acc_t row_max = (mb0 > mb1) ? mb0 : mb1;

        // Passe 3 : Exp_Sum
        acc_t partial_sum[TILE_EXP];
#pragma HLS ARRAY_PARTITION variable=partial_sum complete

        Init_Sum:
        for (int t = 0; t < TILE_EXP; t++) {
#pragma HLS UNROLL
            partial_sum[t] = (acc_t)0;
        }

        Exp_Sum:
        for (int j = 0; j < SEQ_LEN / TILE_EXP; j++) {
#pragma HLS PIPELINE II=1
            for (int t = 0; t < TILE_EXP; t++) {
#pragma HLS UNROLL
                acc_t shifted = row_buf[j * TILE_EXP + t] - row_max;
                if (shifted < (acc_t)-20) shifted = (acc_t)-20;
                acc_t e = (acc_t)hls::exp((float)shifted);
                exp_buf[j * TILE_EXP + t] = e;
                partial_sum[t] += e;
            }
        }

        // Tree-reduction sum
        acc_t s0 = partial_sum[0] + partial_sum[1];
        acc_t s1 = partial_sum[2] + partial_sum[3];
        acc_t s2 = partial_sum[4] + partial_sum[5];
        acc_t s3 = partial_sum[6] + partial_sum[7];
        acc_t t0 = s0 + s1;
        acc_t t1 = s2 + s3;
        acc_t row_sum = t0 + t1;
        acc_t inv_sum = (acc_t)(1.0f / (float)row_sum);

        // Passe 4 : Normalize
        Normalize:
        for (int j = 0; j < SEQ_LEN; j++) {
#pragma HLS PIPELINE II=1
            out[i * SEQ_LEN + j] = (prob_t)(exp_buf[j] * inv_sum);
        }
    }
}
