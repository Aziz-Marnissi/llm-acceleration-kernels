#include <iostream>
#include <cmath>
#define M   1024
#define K   1024
#define TOL 1e-3f   // was 1.0f, way too loose for float32 accumulation
typedef float data_t;
typedef float acc_t;
void matmul(const data_t* A, const data_t* x, acc_t* y);
void matvec_cpu(data_t A[M][K], data_t x[K], acc_t y[M])
{
    for (int i = 0; i < M; i++) {
        acc_t sum = 0;
        for (int k = 0; k < K; k++)
            sum += A[i][k] * x[k];
        y[i] = sum;
    }
}
void init_matrix(data_t A[M][K], int mode)
{
    for (int i = 0; i < M; i++)
        for (int k = 0; k < K; k++) {
            if (mode == 0) A[i][k] = (i == k) ? 1.0f : 0.0f;
            else if (mode == 1) A[i][k] = 0.0f;
            else A[i][k] = (float)((i * 7 + k * 3) % 13);
        }
}
void init_vector(data_t x[K], int mode)
{
    for (int k = 0; k < K; k++) {
        if (mode == 0) x[k] = 1.0f;
        else if (mode == 1) x[k] = 0.0f;
        else x[k] = (float)((k * 5 + 3) % 11);
    }
}
// Functional check only. This is csim (software execution of the HLS
// C++ source) — it validates correctness, NOT hardware timing. Real
// FPGA latency can only be measured on-board via PYNQ (ap_done polling).
int check(acc_t y_fpga[M], acc_t y_ref[M], const char* name)
{
    int errors = 0;
    acc_t max_err = 0;
    for (int i = 0; i < M; i++) {
        acc_t diff = std::fabs(y_fpga[i] - y_ref[i]);
        if (diff > TOL) errors++;
        if (diff > max_err) max_err = diff;
    }
    std::cout << "[" << name << "] errors=" << errors
              << " max_err=" << max_err << "\n";
    return errors;
}
int main()
{
    static data_t A[M][K], x[K];
    static acc_t y_fpga[M], y_cpu[M];
    int total_errors = 0;
    init_matrix(A, 0);
    init_vector(x, 0);
    matmul(&A[0][0], x, y_fpga);
    matvec_cpu(A, x, y_cpu);
    total_errors += check(y_fpga, y_cpu, "Identity x ones");
    init_matrix(A, 1);
    init_vector(x, 2);
    matmul(&A[0][0], x, y_fpga);
    matvec_cpu(A, x, y_cpu);
    total_errors += check(y_fpga, y_cpu, "Zero x random");
    init_matrix(A, 2);
    init_vector(x, 2);
    matmul(&A[0][0], x, y_fpga);
    matvec_cpu(A, x, y_cpu);
    total_errors += check(y_fpga, y_cpu, "Random");
    std::cout << "\n==============================\n";
    std::cout << "Total errors: " << total_errors << "\n";
    std::cout << (total_errors == 0 ? "PASS" : "FAIL") << "\n";
    std::cout << "==============================\n";
    std::cout << "Note: this is csim functional validation only.\n";
    std::cout << "Real hardware timing must come from PYNQ on-board runs.\n";
    return total_errors;
}
