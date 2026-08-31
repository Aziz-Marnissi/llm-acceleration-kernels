#include <iostream>
#include <cmath>

typedef float data_t;

#define VEC_LEN     4864
#define BATCH_SIZE  8
#define TOTAL_LEN   (VEC_LEN * BATCH_SIZE)

// Kernel prototype (must match silu_float32.cpp exactly)
void silu(const data_t *gate, const data_t *up, data_t *out, int n);

// =====================================================
// Reference model
// =====================================================
float silu_ref(float g, float u) {
    float s = 1.0f / (1.0f + std::exp(-g));
    return g * s * u;
}

int main() {
    static data_t gate[TOTAL_LEN], up[TOTAL_LEN], out[TOTAL_LEN];

    // =====================================================
    // INPUT
    // =====================================================
    for (int i = 0; i < TOTAL_LEN; i++) {
        float g = ((i % VEC_LEN) - VEC_LEN / 2) * 8.0f / VEC_LEN;
        float u = 1.0f + (i % 5) * 0.1f;
        gate[i] = g;
        up[i]   = u;
    }

    // =====================================================
    // FPGA CALL
    // =====================================================
    silu(gate, up, out, TOTAL_LEN);

    // =====================================================
    // VALIDATION
    // =====================================================
    int errors = 0;
    for (int i = 0; i < TOTAL_LEN; i++) {
        float g = gate[i];
        float u = up[i];
        float ref = silu_ref(g, u);
        float got = out[i];
        float err = std::fabs(got - ref);
        if (err > 0.02f) {  // LUT-based sigmoid, tighter tolerance than fixed-point
            std::cout
                << "MISMATCH i=" << i
                << " ref=" << ref
                << " got=" << got
                << " err=" << err
                << "\n";
            errors++;
        }
    }
    std::cout << (errors == 0 ? "PASS\n" : "FAIL\n");
    return errors;
}
