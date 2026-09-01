// rope_tb.cpp — Testbench RoPE float32 dataflow
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

typedef float data_t;

#define HEAD_DIM   64
#define HALF_DIM   (HEAD_DIM / 2)
#define MAX_SEQLEN 512

extern "C" void rope(const data_t *vec_in, data_t *vec_out, uint32_t N);

// ─── Référence CPU double précision ──────────────────────────────────────────
static void rope_reference(const double *in, double *out, uint32_t pos) {
    const double base = 10000.0;
    for (int i = 0; i < HALF_DIM; i++) {
        double theta = std::pow(base, -2.0 * i / HEAD_DIM);
        double angle = pos * theta;
        double c = std::cos(angle);
        double s = std::sin(angle);
        double x0 = in[2*i];
        double x1 = in[2*i + 1];
        out[2*i]     = x0 * c - x1 * s;
        out[2*i + 1] = x0 * s + x1 * c;
    }
}

// ─── Mesure perf cycles (C sim uniquement) ───────────────────────────────────
static long long get_cycle_count() {
#ifdef __x86_64__
    unsigned int lo, hi;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return ((long long)hi << 32) | lo;
#else
    return 0;
#endif
}

int main() {
    const uint32_t N = MAX_SEQLEN;

    static data_t hw_in [MAX_SEQLEN * HEAD_DIM];
    static data_t hw_out[MAX_SEQLEN * HEAD_DIM];
    static double ref_in [HEAD_DIM];
    static double ref_out[HEAD_DIM];

    // ── Génération entrées ────────────────────────────────────────────────────
    for (uint32_t p = 0; p < N; p++)
        for (int i = 0; i < HEAD_DIM; i++) {
            double v = std::sin((double)(i+1) * 0.37 + p * 0.07) * 3.0;
            hw_in[p * HEAD_DIM + i] = (data_t)v;
        }

    // ── Appel kernel ──────────────────────────────────────────────────────────
    long long t0 = get_cycle_count();
    rope(hw_in, hw_out, N);
    long long t1 = get_cycle_count();
    printf("C-sim host cycles (rdtsc) : %lld\n\n", t1 - t0);

    // ── Vérification précision ────────────────────────────────────────────────
    const double THRESH = 1e-4;
    double global_max_err = 0.0;
    int    fail_count     = 0;

    uint32_t check_pos[] = {0, 1, 7, 31, 63, 64, 96, 127, 255, 384, 511};
    int num_checks = sizeof(check_pos) / sizeof(check_pos[0]);

    printf("─── Precision checks ───\n");
    for (int t = 0; t < num_checks; t++) {
        uint32_t p = check_pos[t];

        for (int i = 0; i < HEAD_DIM; i++)
            ref_in[i] = std::sin((double)(i+1) * 0.37 + p * 0.07) * 3.0;
        rope_reference(ref_in, ref_out, p);

        double max_err = 0.0, max_rel = 0.0;
        int worst_i = 0;
        for (int i = 0; i < HEAD_DIM; i++) {
            double diff = std::fabs((double)hw_out[p*HEAD_DIM+i] - ref_out[i]);
            double rel  = (std::fabs(ref_out[i]) > 1e-9)
                          ? diff / std::fabs(ref_out[i]) : diff;
            if (diff > max_err) { max_err = diff; worst_i = i; }
            if (rel  > max_rel)   max_rel = rel;
        }
        bool pass = (max_err < THRESH);
        if (!pass) fail_count++;
        if (max_err > global_max_err) global_max_err = max_err;
        printf("pos=%4u | abs=%.2e | rel=%.2e | worst=%2d | %s\n",
               p, max_err, max_rel, worst_i, pass ? "PASS" : "FAIL");
    }

    // ── Cohérence dataflow : positions consécutives ───────────────────────────
    printf("\n─── Dataflow coherence (pos 0..7) ───\n");
    for (uint32_t p = 0; p < 8; p++) {
        for (int i = 0; i < HEAD_DIM; i++)
            ref_in[i] = std::sin((double)(i+1) * 0.37 + p * 0.07) * 3.0;
        rope_reference(ref_in, ref_out, p);
        double err = 0.0;
        for (int i = 0; i < HEAD_DIM; i++)
            err = std::fmax(err, std::fabs((double)hw_out[p*HEAD_DIM+i] - ref_out[i]));
        printf("  pos=%u err=%.2e %s\n", p, err, err < THRESH ? "OK" : "DRIFT");
        if (err >= THRESH) fail_count++;
    }

    // ── Test N partiel ────────────────────────────────────────────────────────
    printf("\n─── Partial N=10 ───\n");
    {
        static data_t tmp[10 * HEAD_DIM] = {};
        rope(hw_in, tmp, 10);
        for (int i = 0; i < HEAD_DIM; i++)
            ref_in[i] = std::sin((double)(i+1) * 0.37 + 9 * 0.07) * 3.0;
        rope_reference(ref_in, ref_out, 9);
        double err = 0.0;
        for (int i = 0; i < HEAD_DIM; i++)
            err = std::fmax(err, std::fabs((double)tmp[9*HEAD_DIM+i] - ref_out[i]));
        printf("  pos=9 err=%.2e %s\n", err, err < THRESH ? "PASS" : "FAIL");
        if (err >= THRESH) fail_count++;
    }

    // ── Test stream : vérifier que les positions ne se mélangent pas ──────────
    printf("\n─── Stream ordering test ───\n");
    {
        // Entrée avec valeur unique par position : hw_in[p][i] = p+1
        static data_t order_in [MAX_SEQLEN * HEAD_DIM];
        static data_t order_out[MAX_SEQLEN * HEAD_DIM];
        for (uint32_t p = 0; p < N; p++)
            for (int i = 0; i < HEAD_DIM; i++)
                order_in[p*HEAD_DIM+i] = (data_t)(p + 1);
        rope(order_in, order_out, N);

        int order_fail = 0;
        for (uint32_t p = 0; p < N; p++) {
        	double angle    = (double)p; // theta_i0 = base^0 = 1
        	double expected = (double)(p + 1) * (std::cos(angle) - std::sin(angle));
        	double got      = (double)order_out[p * HEAD_DIM + 0];
            if (std::fabs(got - expected) > 0.01) {
                printf("  ORDERING FAIL pos=%u expected=%.2f got=%.2f\n",
                       p, expected, got);
                order_fail++;
                if (order_fail > 5) break;
            }
        }
        if (order_fail == 0)
            printf("  All %u positions correctly ordered\n", N);
        fail_count += order_fail;
    }

    // ── N=0 ───────────────────────────────────────────────────────────────────
    printf("\n─── Edge N=0 ───\n");
    {
        static data_t dummy[HEAD_DIM] = {};
        rope(hw_in, dummy, 0);
        printf("  N=0 : no crash OK\n");
    }

    // ── N > MAX_SEQLEN (clamp) ────────────────────────────────────────────────
    printf("\n─── Edge N=600 (clamp to 512) ───\n");
    {
        static data_t big_out[MAX_SEQLEN * HEAD_DIM] = {};
        rope(hw_in, big_out, 600);
        // Vérifie pos=511 (dernière valide)
        for (int i = 0; i < HEAD_DIM; i++)
            ref_in[i] = std::sin((double)(i+1) * 0.37 + 511 * 0.07) * 3.0;
        rope_reference(ref_in, ref_out, 511);
        double err = 0.0;
        for (int i = 0; i < HEAD_DIM; i++)
            err = std::fmax(err, std::fabs((double)big_out[511*HEAD_DIM+i] - ref_out[i]));
        printf("  pos=511 err=%.2e %s\n", err, err < THRESH ? "PASS" : "FAIL");
        if (err >= THRESH) fail_count++;
    }

    // ── Résumé ────────────────────────────────────────────────────────────────
    printf("\n=== SUMMARY ===\n");
    printf("Threshold    : %.1e\n", THRESH);
    printf("Max abs err  : %.6e\n", global_max_err);
    printf("Tests failed : %d\n",   fail_count);
    return (fail_count == 0) ? 0 : 1;
}
