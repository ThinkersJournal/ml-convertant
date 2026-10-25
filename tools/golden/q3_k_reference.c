/* Harness: reads raw f32, writes ggml's Q3_K encoding and its decoding. Built against ggml-quants.c
   from the pinned llama.cpp commit (see THIRD-PARTY-NOTICES.md); build with -DNDEBUG as llama.cpp does. */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "ggml-quants.h"
#include "ggml-impl.h"
#include <fcntl.h>
#include <sys/stat.h>
#ifdef _WIN32
#include <io.h>
#define OPEN _open
#define MODE (_S_IREAD | _S_IWRITE)
#define BIN _O_BINARY
#else
#include <unistd.h>
#define OPEN open
#define MODE (S_IRUSR | S_IWUSR)
#define BIN 0
#endif
/* Output files are created owner-only (0600), never world-writable. */
static FILE*create(const char*path){
  int fd=OPEN(path,O_WRONLY|O_CREAT|O_TRUNC|BIN,MODE);
  return fd<0?NULL:fdopen(fd,"wb");}
void ggml_abort(const char*f,int l,const char*fmt,...){fprintf(stderr,"abort %s:%d\n",f,l);exit(2);}
int main(int argc,char**argv){
  FILE*f=fopen(argv[1],"rb");fseek(f,0,SEEK_END);long n=ftell(f)/4;fseek(f,0,SEEK_SET);
  float*x=malloc(n*4);fread(x,4,n,f);fclose(f);
  long nb=n/256;block_q3_K*y=malloc(nb*sizeof(block_q3_K));float*d=malloc(n*4);
  quantize_row_q3_K_ref(x,y,n);dequantize_row_q3_K(y,d,n);
  f=create(argv[2]);fwrite(y,sizeof(block_q3_K),nb,f);fclose(f);
  f=create(argv[3]);fwrite(d,4,n,f);fclose(f);
  printf("sizeof=%d n=%ld\n",(int)sizeof(block_q3_K),n);return 0;}
size_t ggml_type_size(enum ggml_type t){return 0;} size_t ggml_row_size(enum ggml_type t,int64_t n){return 0;} const char*ggml_type_name(enum ggml_type t){return "";}
