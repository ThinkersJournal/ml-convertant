#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
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
#include "ggml-quants.h"
#include "ggml-impl.h"
/* Output files are created owner-only (0600), never world-writable. */
static FILE*create(const char*path){
  int fd=OPEN(path,O_WRONLY|O_CREAT|O_TRUNC|BIN,MODE);
  return fd<0?NULL:fdopen(fd,"wb");}
void ggml_abort(const char*f,int l,const char*fmt,...){fprintf(stderr,"abort %s:%d\n",f,l);exit(2);}
/* usage: iq_dequant_reference <format> blocks.bin out.f32
   Calls ggml's own dequantize_row_<format> on raw blocks (any bytes with a finite f16 scale are a valid block).
   format: iq3_xxs, iq2_xxs, iq2_xs, iq2_s, iq1_s, iq1_m. */
int main(int argc,char**argv){
  if(argc<4)return 2;
  FILE*f=fopen(argv[2],"rb");if(!f){fprintf(stderr,"open failed\n");return 2;}
  fseek(f,0,SEEK_END);long nb=ftell(f);fseek(f,0,SEEK_SET);
  unsigned char*b=malloc(nb);if(fread(b,1,nb,f)!=(size_t)nb)return 2;fclose(f);
  long n;float*d;
  if(!strcmp(argv[1],"iq3_xxs")){
    if(nb%sizeof(block_iq3_xxs))return 2;n=nb/sizeof(block_iq3_xxs)*256;d=malloc(n*4);
    dequantize_row_iq3_xxs((const block_iq3_xxs*)b,d,n);
  }else if(!strcmp(argv[1],"iq2_xxs")){
    if(nb%sizeof(block_iq2_xxs))return 2;n=nb/sizeof(block_iq2_xxs)*256;d=malloc(n*4);
    dequantize_row_iq2_xxs((const block_iq2_xxs*)b,d,n);
  }else if(!strcmp(argv[1],"iq2_xs")){
    if(nb%sizeof(block_iq2_xs))return 2;n=nb/sizeof(block_iq2_xs)*256;d=malloc(n*4);
    dequantize_row_iq2_xs((const block_iq2_xs*)b,d,n);
  }else if(!strcmp(argv[1],"iq2_s")){
    if(nb%sizeof(block_iq2_s))return 2;n=nb/sizeof(block_iq2_s)*256;d=malloc(n*4);
    dequantize_row_iq2_s((const block_iq2_s*)b,d,n);
  }else if(!strcmp(argv[1],"iq1_s")){
    if(nb%sizeof(block_iq1_s))return 2;n=nb/sizeof(block_iq1_s)*256;d=malloc(n*4);
    dequantize_row_iq1_s((const block_iq1_s*)b,d,n);
  }else if(!strcmp(argv[1],"iq1_m")){
    if(nb%sizeof(block_iq1_m))return 2;n=nb/sizeof(block_iq1_m)*256;d=malloc(n*4);
    dequantize_row_iq1_m((const block_iq1_m*)b,d,n);
  }else{fprintf(stderr,"unknown format %s\n",argv[1]);return 2;}
  f=create(argv[3]);fwrite(d,4,n,f);fclose(f);
  printf("%s blocks_bytes=%ld n=%ld\n",argv[1],nb,n);return 0;}
size_t ggml_type_size(enum ggml_type t){return 0;} size_t ggml_row_size(enum ggml_type t,int64_t n){return 0;} const char*ggml_type_name(enum ggml_type t){return "";}
