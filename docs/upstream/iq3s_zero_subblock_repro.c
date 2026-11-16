/* Minimal repro: quantize_iq3_s mis-places the later sub-blocks of a block that has an exactly-zero 32-element sub-block.
   Build against a llama.cpp checkout (from its root):
     gcc -O2 -DNDEBUG -w -Iggml/src -Iggml/include iq3s_zero_subblock_repro.c ggml/src/ggml-quants.c -lm -o repro
   Same stubs as tools/golden/*_reference.c (ggml_abort and the type-traits helpers the file references). */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "ggml-quants.h"
#include "ggml-impl.h"
void ggml_abort(const char*f,int l,const char*fmt,...){fprintf(stderr,"abort %s:%d\n",f,l);exit(2);}
size_t ggml_type_size(enum ggml_type t){return 0;} size_t ggml_row_size(enum ggml_type t,int64_t n){return 0;} const char*ggml_type_name(enum ggml_type t){return "";}
int main(void){
  iq3xs_init_impl(512);
  float x[256], y[256];
  for (int i = 0; i < 256; ++i) x[i] = (float)(((i*37) % 64) - 32) / 16.0f + 0.03125f*(i%5);   /* deterministic, non-zero */
  for (int i = 32; i < 64; ++i) x[i] = 0.0f;                                                   /* sub-block 1 exactly zero */
  block_iq3_s b;
  quantize_iq3_s(x, &b, 1, 256, NULL);
  dequantize_row_iq3_s(&b, y, 256);
  for (int ib = 0; ib < 8; ++ib) {
    double e = 0, n = 0;
    for (int i = 32*ib; i < 32*ib+32; ++i) { e += fabs(y[i]-x[i]); n += fabs(x[i]); }
    printf("sub-block %d: mean |x| %.3f  mean |decoded - x| %.3f\n", ib, n/32, e/32);
  }
  return 0;
}
