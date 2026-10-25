#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "ggml-quants.h"
#include "ggml-impl.h"
void ggml_abort(const char*f,int l,const char*fmt,...){fprintf(stderr,"abort %s:%d\n",f,l);exit(2);}
int main(int argc,char**argv){
  FILE*f=fopen(argv[1],"rb");fseek(f,0,SEEK_END);long n=ftell(f)/4;fseek(f,0,SEEK_SET);
  float*x=malloc(n*4);fread(x,4,n,f);fclose(f);
  long nb=n/32;block_q8_0*y=malloc(nb*sizeof(block_q8_0));float*d=malloc(n*4);
  quantize_row_q8_0_ref(x,y,n);dequantize_row_q8_0(y,d,n);
  f=fopen(argv[2],"wb");fwrite(y,sizeof(block_q8_0),nb,f);fclose(f);
  f=fopen(argv[3],"wb");fwrite(d,4,n,f);fclose(f);
  printf("sizeof=%d n=%ld\n",(int)sizeof(block_q8_0),n);return 0;}
size_t ggml_type_size(enum ggml_type t){return 0;} size_t ggml_row_size(enum ggml_type t,int64_t n){return 0;} const char*ggml_type_name(enum ggml_type t){return "";}
