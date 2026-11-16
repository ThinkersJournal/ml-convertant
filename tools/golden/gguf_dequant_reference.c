#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include "ggml-quants.h"
#include "ggml-impl.h"
/* Output files are created owner-only (0600), never world-writable. */
static FILE*create(const char*path){
  int fd=open(path,O_WRONLY|O_CREAT|O_TRUNC,S_IRUSR|S_IWUSR);
  return fd<0?NULL:fdopen(fd,"wb");}
void ggml_abort(const char*f,int l,const char*fmt,...){fprintf(stderr,"abort %s:%d\n",f,l);exit(2);}
/* usage: gguf_dequant_reference model.gguf spec.txt outdir
   spec.txt has one line per tensor: "<idx> <ggml_type> <abs_offset> <nbytes> <nelem>". Reads the blocks
   straight from the GGUF and writes outdir/<idx>.f32 with ggml's own dequantize_row_* output.
   ggml_type: 20 = IQ4_NL, 21 = IQ3_S, 23 = IQ4_XS. */
int main(int argc,char**argv){
  if(argc<4)return 2;
  FILE*g=fopen(argv[1],"rb"),*s=fopen(argv[2],"r");
  if(!g||!s){fprintf(stderr,"open failed\n");return 2;}
  long idx,ty,off,nb,n;
  while(fscanf(s,"%ld %ld %ld %ld %ld",&idx,&ty,&off,&nb,&n)==5){
    unsigned char*b=malloc(nb);float*d=malloc(n*4);
    fseek(g,off,SEEK_SET);if(fread(b,1,nb,g)!=(size_t)nb){fprintf(stderr,"short read %ld\n",idx);return 2;}
    if(ty==20)dequantize_row_iq4_nl((const block_iq4_nl*)b,d,n);
    else if(ty==21)dequantize_row_iq3_s((const block_iq3_s*)b,d,n);
    else if(ty==23)dequantize_row_iq4_xs((const block_iq4_xs*)b,d,n);
    else{fprintf(stderr,"type %ld unsupported\n",ty);return 2;}
    char p[4096];snprintf(p,sizeof p,"%s/%ld.f32",argv[3],idx);
    FILE*o=create(p);fwrite(d,4,n,o);fclose(o);free(b);free(d);}
  return 0;}
size_t ggml_type_size(enum ggml_type t){return 0;} size_t ggml_row_size(enum ggml_type t,int64_t n){return 0;} const char*ggml_type_name(enum ggml_type t){return "";}
