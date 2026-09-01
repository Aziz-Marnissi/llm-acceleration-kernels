#include "silu.h"
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <vector>

double sigmoid_ref(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

int main() {
    int n = 256;

    std::vector<data_t> gate(n), up(n), out(n);
    std::vector<double> gate_ref(n), up_ref(n);

    srand(42);
    for (int i = 0; i < n; i++) {
        double g = ((double)rand()/RAND_MAX)*16.0 - 8.0;
        double u = ((double)rand()/RAND_MAX)*2.0 - 1.0;
        gate[i] = (data_t)g;
        up[i]   = (data_t)u;
        gate_ref[i] = g;
        up_ref[i]   = u;
    }

    silu(gate.data(), up.data(), out.data(), n);

    double max_err = 0.0;
    for (int i = 0; i < n; i++) {
        double ref = gate_ref[i] * sigmoid_ref(gate_ref[i]) * up_ref[i];
        double got = (double)out[i];
        double err = std::fabs(ref - got);
        if (err > max_err) max_err = err;
    }

    printf("N=%d max_err=%f -> %s\n", n, max_err, (max_err < 0.05) ? "PASS" : "FAIL");
    return (max_err < 0.05) ? 0 : 1;
}
