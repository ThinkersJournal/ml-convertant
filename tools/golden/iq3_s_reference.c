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
/* usage: iq3_s_reference in.f32 out.bin out.deq.f32 [importance.f32]
   Calls ggml's quantize_iq3_s, the function llama-quantize uses. With an importance file, its length
   is n_per_row and the input is rows of that length; without one the input is a single row. */
static float*readf(const char*path,long*n){
  FILE*f=fopen(path,"rb");fseek(f,0,SEEK_END);*n=ftell(f)/4;fseek(f,0,SEEK_SET);
  float*x=malloc((*n)*4);fread(x,4,*n,f);fclose(f);return x;}
int main(int argc,char**argv){
  iq3xs_init_impl(512);
  long n,nw=0;float*x=readf(argv[1],&n);float*w=NULL;
  if(argc>4)w=readf(argv[4],&nw);
  long per_row=w?nw:n;long nrow=n/per_row;
  long nb=n/256;block_iq3_s*y=malloc(nb*sizeof(block_iq3_s));float*d=malloc(n*4);
  quantize_iq3_s(x,y,nrow,per_row,w);dequantize_row_iq3_s(y,d,n);
  FILE*f=create(argv[2]);fwrite(y,sizeof(block_iq3_s),nb,f);fclose(f);
  f=create(argv[3]);fwrite(d,4,n,f);fclose(f);
  printf("sizeof=%d n=%ld rows=%ld\n",(int)sizeof(block_iq3_s),n,nrow);return 0;}
size_t ggml_type_size(enum ggml_type t){return 0;} size_t ggml_row_size(enum ggml_type t,int64_t n){return 0;} const char*ggml_type_name(enum ggml_type t){return "";}
