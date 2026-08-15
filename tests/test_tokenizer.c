#include "fox/fox.h"
#include "check.h"

#include <stdio.h>
#include <stdint.h>
#include <string.h>

static void put32(FILE *f, uint32_t v) { uint8_t b[4]={(uint8_t)v,(uint8_t)(v>>8),(uint8_t)(v>>16),(uint8_t)(v>>24)}; fwrite(b,1,4,f); }
static void put64(FILE *f, uint64_t v) { int i; for(i=0;i<8;i++) fputc((int)(v>>(8*i)),f); }
static void putstr(FILE *f, const char *s) { size_t n=strlen(s); put64(f,n); fwrite(s,1,n,f); }
static void putf32(FILE *f, float value) { union { uint32_t u; float f; } v; v.f=value; put32(f,v.u); }

static void make_model(const char *path)
{
    static const char *tokens[] = { "<s>", "</s>", "▁", "▁Rust", "▁Fox", "Rust", "Fox", "<0x21>" };
    static const float scores[] = { 0, 0, -5, 2, 2, -2, -2, -100 };
    FILE *f=fopen(path,"wb"); size_t i;
    put32(f,0x46554747u); put32(f,3); put64(f,0); put64(f,6);
    putstr(f,"tokenizer.ggml.model"); put32(f,FOX_GGUF_STRING); putstr(f,"llama");
    putstr(f,"tokenizer.ggml.tokens"); put32(f,FOX_GGUF_ARRAY); put32(f,FOX_GGUF_STRING); put64(f,8); for(i=0;i<8;i++)putstr(f,tokens[i]);
    putstr(f,"tokenizer.ggml.scores"); put32(f,FOX_GGUF_ARRAY); put32(f,FOX_GGUF_FLOAT32); put64(f,8); for(i=0;i<8;i++)putf32(f,scores[i]);
    putstr(f,"tokenizer.ggml.bos_token_id"); put32(f,FOX_GGUF_UINT32); put32(f,0);
    putstr(f,"tokenizer.ggml.eos_token_id"); put32(f,FOX_GGUF_UINT32); put32(f,1);
    putstr(f,"tokenizer.ggml.add_bos_token"); put32(f,FOX_GGUF_BOOL); fputc(1,f);
    while(ftell(f)%32)fputc(0,f); fclose(f);
}

int main(void)
{
    const char *path="rustfox-tokenizer.gguf"; fox_gguf *g=NULL; fox_tokenizer *t=NULL; uint32_t ids[8]; char text[64]; size_t n=0;
    make_model(path);
    CHECK(fox_gguf_open(path,&g)==FOX_OK,"realistic tokenizer GGUF opens");
    CHECK(fox_tokenizer_open(g,&t)==FOX_OK,"Llama tokenizer metadata is accepted");
    CHECK(fox_tokenizer_vocab_size(t)==8&&fox_tokenizer_bos(t)==0&&fox_tokenizer_eos(t)==1,"tokenizer IDs come from GGUF");
    CHECK(fox_tokenizer_encode(t,"Rust Fox!",ids,8,&n)==FOX_OK&&n==4&&ids[0]==0&&ids[1]==3&&ids[2]==4&&ids[3]==7,"prompt becomes model vocabulary IDs");
    CHECK(fox_tokenizer_decode(t,ids,n,text,sizeof(text))==FOX_OK&&strcmp(text,"Rust Fox!")==0,"generated IDs decode to text");
    CHECK(fox_tokenizer_encode(t,"Rust Fox!",ids,2,&n)==FOX_ERR_ARG&&n==4,"encoding reports required token capacity");
    fox_tokenizer_close(t); fox_gguf_close(g); remove(path);
    CHECK(fox_tokenizer_open(NULL,&t)==FOX_ERR_ARG,"null tokenizer model is rejected");
    CHECK_DONE();
}
