#include <cstdio>
#include <cmath>
#include <cstdlib>
#include "ap_fixed.h"

#define SEQ_LEN 128

typedef ap_fixed<16, 8>  data_t;
typedef ap_fixed<32, 16> acc_t;
typedef ap_ufixed<16, 1> prob_t;

void softmax(data_t* in, prob_t* out);

int main() {
    static data_t in[SEQ_LEN * SEQ_LEN];
    static prob_t out[SEQ_LEN * SEQ_LEN];

    // Initialisation : valeurs aléatoires dans [-4, 4]
    srand(42);
    for (int i = 0; i < SEQ_LEN * SEQ_LEN; i++)
        in[i] = (data_t)((float)(rand() % 801 - 400) / 100.0f);

    softmax(in, out);

    // Vérification : chaque row doit sommer à ~1.0
    int pass = 1;
    for (int i = 0; i < SEQ_LEN; i++) {
        float sum = 0.0f;
        float row_max = -1e9f;

        for (int j = 0; j < SEQ_LEN; j++) {
            float v = (float)in[i * SEQ_LEN + j];
            if (v > row_max) row_max = v;
        }

        float ref_sum = 0.0f;
        for (int j = 0; j < SEQ_LEN; j++)
            ref_sum += expf((float)in[i * SEQ_LEN + j] - row_max);

        for (int j = 0; j < SEQ_LEN; j++)
            sum += (float)out[i * SEQ_LEN + j];

        if (fabsf(sum - 1.0f) > 0.02f) {
            printf("FAIL row %d : sum = %.6f\n", i, sum);
            pass = 0;
        }
    }

    printf("\nRow 0 (premiers 8 éléments) :\n");
    for (int j = 0; j < 8; j++)
        printf("  out[0][%d] = %.6f\n", j, (float)out[j]);

    printf("\n%s\n", pass ? "=== TEST PASSED ===" : "=== TEST FAILED ===");
    return pass ? 0 : 1;
}
