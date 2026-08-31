#include <iostream>
#include <cstdlib>

typedef float fp;
typedef unsigned int idx_t;

#define VOCAB_SIZE 32768
#define HALF (VOCAB_SIZE / 2)

void greedy_sampler(const fp *logits0, const fp *logits1, idx_t *best_idx, int size);

static void set_value(fp *l0, fp *l1, int idx, fp val) {
    if (idx < HALF) l0[idx] = val;
    else l1[idx - HALF] = val;
}
static void reset_array(fp *l0, fp *l1, fp val) {
    for (int i = 0; i < HALF; i++) { l0[i] = val; l1[i] = val; }
}

bool run_test(const char *name, fp *l0, fp *l1, idx_t expected) {
    idx_t result = 0;
    greedy_sampler(l0, l1, &result, VOCAB_SIZE);
    bool ok = (result == expected);
    std::cout << (ok ? "[PASS] " : "[FAIL] ") << name;
    if (!ok) std::cout << " attendu=" << expected << " obtenu=" << result;
    std::cout << "\n";
    return ok;
}

int main() {
    static fp logits0[HALF], logits1[HALF];
    int pass = 0, total = 0;
    srand(42);

    for (int i = 0; i < HALF; i++) {
        logits0[i] = (float)(rand() % 1000) / 100.0f;
        logits1[i] = (float)(rand() % 1000) / 100.0f;
    }
    set_value(logits0, logits1, 12345, 999.0f);
    pass += run_test("T1 max milieu", logits0, logits1, 12345); total++;

    reset_array(logits0, logits1, 1.0f);
    set_value(logits0, logits1, 0, 50.0f);
    pass += run_test("T2 max debut", logits0, logits1, 0); total++;

    reset_array(logits0, logits1, 1.0f);
    set_value(logits0, logits1, VOCAB_SIZE - 1, 50.0f);
    pass += run_test("T3 max fin", logits0, logits1, VOCAB_SIZE - 1); total++;

    reset_array(logits0, logits1, 3.14f);
    pass += run_test("T4 tous egaux", logits0, logits1, 0); total++;

    reset_array(logits0, logits1, 1.0f);
    set_value(logits0, logits1, 255, 77.0f);
    pass += run_test("T5 frontiere", logits0, logits1, 255); total++;

    std::cout << "\n=== " << pass << "/" << total << " tests passes ===\n";
    return (pass == total) ? 0 : 1;
}
