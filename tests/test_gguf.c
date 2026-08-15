#include "fox/fox.h"
#include "check.h"

#include <stdio.h>
#include <stdint.h>
#include <string.h>

static void put32(FILE *f, uint32_t v) { uint8_t b[4]={(uint8_t)v,(uint8_t)(v>>8),(uint8_t)(v>>16),(uint8_t)(v>>24)}; fwrite(b,1,4,f); }
static void put64(FILE *f, uint64_t v) { int i; for(i=0;i<8;i++) fputc((int)(v>>(8*i)),f); }
static void str(FILE *f, const char *s) { size_t n=strlen(s); put64(f,n); fwrite(s,1,n,f); }

int main(void)
{
    const char *path = "rustfox-test.gguf";
    FILE *f = fopen(path, "wb");
    fox_gguf *g = NULL;
    fox_gguf_tensor t;
    size_t idx;
    const char *s;
    fox_status st;

    put32(f, 0x46554747u); put32(f, 3); put64(f, 2); put64(f, 2);
    str(f, "general.alignment"); put32(f, FOX_GGUF_UINT32); put32(f, 32);
    str(f, "general.name"); put32(f, FOX_GGUF_STRING); str(f, "tiny-ternary");
    str(f, "input"); put32(f, 1); put64(f, 256); put32(f, FOX_GGML_TQ1_0); put64(f, 0);
    str(f, "output"); put32(f, 1); put64(f, 256); put32(f, FOX_GGML_TQ2_0); put64(f, 54);
    while (ftell(f) % 32) fputc(0, f);
    for (idx=0; idx<120; idx++) fputc(0, f);
    fclose(f);

    st=fox_gguf_open(path,&g); CHECK(st==FOX_OK,"valid GGUF opens");
    CHECK(fox_gguf_version(g)==3,"version is exposed");
    CHECK(fox_gguf_metadata_count(g)==2,"metadata count is exposed");
    idx=fox_gguf_find(g,"general.name"); CHECK(idx!=SIZE_MAX,"metadata lookup works");
    s=NULL; CHECK(fox_gguf_get_string(g,idx,&s)==FOX_OK&&strcmp(s,"tiny-ternary")==0,"string metadata works");
    CHECK(fox_gguf_tensor_count(g)==2,"tensor count is exposed");
    CHECK(fox_gguf_tensor_find(g,"output")==1,"tensor lookup works");
    CHECK(fox_gguf_tensor_at(g,0,&t)==FOX_OK&&t.type==FOX_GGML_TQ1_0&&t.size_bytes==54,"TQ1 tensor size is known");
    CHECK(fox_gguf_tensor_at(g,1,&t)==FOX_OK&&t.type==FOX_GGML_TQ2_0&&t.offset==54,"TQ2 tensor descriptor works");
    fox_gguf_close(g);
    remove(path);
    CHECK(fox_gguf_open("missing.gguf",&g)==FOX_ERR_IO,"missing files report IO");
    CHECK(fox_gguf_open(NULL,&g)==FOX_ERR_ARG,"null path reports argument error");
    f=fopen(path,"wb"); fputc('G',f); fputc('G',f); fputc('U',f); fputc('F',f); fclose(f);
    CHECK(fox_gguf_open(path,&g)==FOX_ERR_FORMAT,"truncated files report format error");
    remove(path);
    CHECK_DONE();
}
