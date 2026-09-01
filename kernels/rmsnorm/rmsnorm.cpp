#include <hls_math.h>
#include <cstring>
typedef float fp;
#define VEC 8
#define BLOCK 128
#define MAX_SIZE 4096

void rmsnorm(fp *out, fp *x, fp *weight, int size, fp eps)
{
#pragma HLS INTERFACE m_axi port=out    depth=256 bundle=g0
#pragma HLS INTERFACE m_axi port=x      depth=256 bundle=g1
#pragma HLS INTERFACE m_axi port=weight depth=256 bundle=g2
#pragma HLS INTERFACE s_axilite port=return
#pragma HLS INTERFACE s_axilite port=size
#pragma HLS INTERFACE s_axilite port=eps

    static fp local_buf[MAX_SIZE];
    static fp w_buf[MAX_SIZE];
#pragma HLS ARRAY_RESHAPE variable=local_buf cyclic factor=VEC dim=1
#pragma HLS DEPENDENCE variable=local_buf false
#pragma HLS ARRAY_RESHAPE variable=w_buf cyclic factor=VEC dim=1

    static fp x_buf[MAX_SIZE];
#pragma HLS ARRAY_RESHAPE variable=x_buf cyclic factor=VEC dim=1
#pragma HLS DEPENDENCE variable=x_buf false

    memcpy(x_buf, x, size * sizeof(fp));

    fp global_sum = 0;

SUM_BLOCK:
    for(int b = 0; b < size; b += BLOCK)
    {
        fp acc0=0, acc1=0, acc2=0, acc3=0;
        fp acc4=0, acc5=0, acc6=0, acc7=0;
    SUM_LOCAL:
        for(int i = 0; i < BLOCK && (b+i) < size; i += VEC)
        {
#pragma HLS PIPELINE II=1
            int idx = b + i;
            fp x0=x_buf[idx], x1=x_buf[idx+1], x2=x_buf[idx+2], x3=x_buf[idx+3];
            fp x4=x_buf[idx+4], x5=x_buf[idx+5], x6=x_buf[idx+6], x7=x_buf[idx+7];
            local_buf[idx]   = x0;
            local_buf[idx+1] = x1;
            local_buf[idx+2] = x2;
            local_buf[idx+3] = x3;
            local_buf[idx+4] = x4;
            local_buf[idx+5] = x5;
            local_buf[idx+6] = x6;
            local_buf[idx+7] = x7;
            acc0 += x0*x0; acc1 += x1*x1; acc2 += x2*x2; acc3 += x3*x3;
            acc4 += x4*x4; acc5 += x5*x5; acc6 += x6*x6; acc7 += x7*x7;
        }
        fp block_sum = ((acc0+acc1)+(acc2+acc3)) + ((acc4+acc5)+(acc6+acc7));
        global_sum += block_sum;
    }

    fp rms = hls::sqrtf(global_sum / size + eps);
    fp inv = 1.0f / rms;

LOAD_W:
    memcpy(w_buf, weight, size * sizeof(fp));

    fp out_buf[MAX_SIZE];
#pragma HLS ARRAY_RESHAPE variable=out_buf cyclic factor=VEC dim=1

SCALE:
    for(int i = 0; i < size; i += VEC)
    {
#pragma HLS PIPELINE II=1
        out_buf[i]   = local_buf[i]   * w_buf[i]   * inv;
        out_buf[i+1] = local_buf[i+1] * w_buf[i+1] * inv;
        out_buf[i+2] = local_buf[i+2] * w_buf[i+2] * inv;
        out_buf[i+3] = local_buf[i+3] * w_buf[i+3] * inv;
        out_buf[i+4] = local_buf[i+4] * w_buf[i+4] * inv;
        out_buf[i+5] = local_buf[i+5] * w_buf[i+5] * inv;
        out_buf[i+6] = local_buf[i+6] * w_buf[i+6] * inv;
        out_buf[i+7] = local_buf[i+7] * w_buf[i+7] * inv;
    }

STORE:
    memcpy(out, out_buf, size * sizeof(fp));
}
