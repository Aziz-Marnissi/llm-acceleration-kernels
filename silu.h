#ifndef SILU_H
#define SILU_H
#include <hls_half.h>
typedef half data_t;
typedef half acc_t;

void silu(const data_t *gate, const data_t *up, data_t *out, int n);

#endif
