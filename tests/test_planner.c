#include "fox/fox.h"
#include "check.h"

#include <stdio.h>
#include <stdint.h>
#include <string.h>

static void put32(FILE *f, uint32_t v) { uint8_t b[4]={(uint8_t)v,(uint8_t)(v>>8),(uint8_t)(v>>16),(uint8_t)(v>>24)}; fwrite(b,1,4,f); }
static void put64(FILE *f, uint64_t v) { int i; for(i=0;i<8;i++) fputc((int)(v>>(8*i)),f); }
static void putstr(FILE *f, const char *s) { size_t n=strlen(s); put64(f,n); fwrite(s,1,n,f); }

static void make_model(const char *path)
{
    FILE *f=fopen(path,"wb"); size_t i;
    put32(f,0x46554747u); put32(f,3); put64(f,4); put64(f,1);
    putstr(f,"general.alignment"); put32(f,FOX_GGUF_UINT32); put32(f,32);
    putstr(f,"blk.0.attn.q_proj"); put32(f,1); put64(f,256); put32(f,FOX_GGML_TQ2_0); put64(f,0);
    putstr(f,"blk.0.ffn.down_proj"); put32(f,1); put64(f,256); put32(f,FOX_GGML_TQ2_0); put64(f,66);
    putstr(f,"token_embd.weight"); put32(f,1); put64(f,256); put32(f,FOX_GGML_TQ2_0); put64(f,132);
    putstr(f,"mystery"); put32(f,1); put64(f,256); put32(f,FOX_GGML_TQ2_0); put64(f,198);
    while(ftell(f)%32)fputc(0,f); for(i=0;i<264;i++)fputc(0,f); fclose(f);
}

int main(void)
{
    const char *path="rustfox-planner.gguf"; fox_gguf *g=NULL; fox_plan *p=NULL; fox_plan_config cfg; fox_plan_item item; fox_plan_summary s;
    make_model(path);
    CHECK(fox_gguf_open(path,&g)==FOX_OK,"planner fixture opens");
    cfg.residency_budget_bytes=66; cfg.seq_read_mbps=100; cfg.compute_tok_s=10;
    CHECK(fox_plan_build(g,&cfg,&p)==FOX_OK,"planner builds");
    CHECK(fox_plan_count(p)==4,"planner has one item per tensor");
    CHECK(fox_plan_item_at(p,0,&item)==FOX_OK&&item.placement==FOX_PLAN_PIN,"hot attention tensor is pinned");
    CHECK(fox_plan_item_at(p,2,&item)==FOX_OK&&item.placement==FOX_PLAN_MMAP,"embedding uses mmap placement");
    CHECK(fox_plan_item_at(p,1,&item)==FOX_OK&&item.placement==FOX_PLAN_STREAM,"FFN projection streams");
    CHECK(fox_plan_get_summary(p,&s)==FOX_OK&&s.pinned_bytes<=66,"pinned bytes respect budget");
    CHECK(s.streamed_bytes==132&&s.mmap_bytes==66,"summary separates stream and mmap bytes");
    CHECK(s.predicted_tok_s>0.0&&s.predicted_tok_s<=10.0,"prediction is bounded by compute");
    fox_plan_destroy(p); p=NULL;
    cfg.seq_read_mbps=0.0;
    CHECK(fox_plan_build(g,&cfg,&p)==FOX_OK,"planner accepts unmeasured storage");
    CHECK(fox_plan_get_summary(p,&s)==FOX_OK&&s.predicted_tok_s==0.0,"unmeasured storage yields no tok/s claim");
    fox_plan_destroy(p); fox_gguf_close(g); remove(path);
    CHECK(fox_plan_build(NULL,&cfg,&p)==FOX_ERR_ARG,"null model is rejected");
    CHECK_DONE();
}
