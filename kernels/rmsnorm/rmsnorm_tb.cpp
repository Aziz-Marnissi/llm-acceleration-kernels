#include <iostream>
#include <cmath>
#include <cstdlib>
typedef float fp;
void rmsnorm(fp *out, fp *x, fp *weight, int size, fp eps);

int main() {
    const int size = 256;
    fp x[size];
    fp weight[size];
    fp out[size];

    for (int i = 0; i < size; i++) {
        x[i]      = (float)(std::sin(i * 0.01f) + 0.5f * std::cos(i * 0.03f));
        weight[i] = 1.0f;
    }
    fp eps = 1e-5f;

    rmsnorm(out, x, weight, size, eps);

    float sum = 0.0f;
    for (int i = 0; i < size; i++) sum += x[i] * x[i];
    float rms = std::sqrt(sum / (float)size + eps);

    int errors = 0;
    for (int i = 0; i < size; i++) {
        float ref = (x[i] * weight[i]) / rms;
        float diff = std::fabs(out[i] - ref);
        if (diff > 1e-3f) {
            std::cout << "ERROR i=" << i << " out=" << out[i]
                      << " ref=" << ref << " diff=" << diff << std::endl;
            errors++;
        }
    }
    std::cout << (errors == 0 ? "TEST PASSED" : "TEST FAILED: " + std::to_string(errors)) << std::endl;
    return errors ? 1 : 0;
}
