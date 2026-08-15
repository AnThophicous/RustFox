#include "fox/fox.h"
#include "fox_internal.h"
#include "gguf_internal.h"
#include "platform.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#define GGUF_MAX_ITEMS 1000000u
#define GGUF_MAX_META 65536u
#define GGUF_MAX_STRING (16u * 1024u * 1024u)

typedef struct { fox_file *f; uint64_t pos, size; } reader;

static int add_ok(uint64_t a, uint64_t b, uint64_t *r) { if (b > UINT64_MAX - a) return 0; *r = a + b; return 1; }
static int mul_ok(uint64_t a, uint64_t b, uint64_t *r) { if (a && b > UINT64_MAX / a) return 0; *r = a * b; return 1; }
static int read_bytes(reader *r, void *p, size_t n) { if ((uint64_t)n > r->size - r->pos || fox_file_pread(r->f, p, n, r->pos) != (int64_t)n) return 0; r->pos += n; return 1; }
static int u32(reader *r, uint32_t *v) { uint8_t b[4]; if (!read_bytes(r,b,4)) return 0; *v=(uint32_t)b[0]|((uint32_t)b[1]<<8)|((uint32_t)b[2]<<16)|((uint32_t)b[3]<<24); return 1; }
static int u64(reader *r, uint64_t *v) { uint8_t b[8]; int i; if (!read_bytes(r,b,8)) return 0; *v=0; for(i=0;i<8;i++) *v |= (uint64_t)b[i] << (8*i); return 1; }
static int i32(reader *r, int32_t *v) { uint32_t x; if (!u32(r,&x)) return 0; *v=(int32_t)x; return 1; }
static int str_read(reader *r, char **out) { uint64_t n; char *s; if (!u64(r,&n) || n > GGUF_MAX_STRING || n > r->size-r->pos || n > SIZE_MAX-1) return 0; s=(char *)malloc((size_t)n+1); if (!s) return -1; if (!read_bytes(r,s,(size_t)n)) { free(s); return 0; } s[n]='\0'; *out=s; return 1; }
static int skip(reader *r, uint64_t n) { if (n > r->size - r->pos) return 0; r->pos += n; return 1; }

static int value_size(fox_gguf_value_type t, uint64_t *n) { static const uint8_t sizes[] = {1,1,2,2,4,4,4,1,0,0,8,8,8}; if (t >= 13 || t == FOX_GGUF_STRING || t == FOX_GGUF_ARRAY) return 0; *n=sizes[t]; return 1; }

static int tensor_layout(fox_ggml_type t, uint32_t *block, uint32_t *bytes) {
    switch (t) {
    case FOX_GGML_F32: *block=1; *bytes=4; return 1; case FOX_GGML_F16: case FOX_GGML_BF16: *block=1; *bytes=2; return 1;
    case FOX_GGML_I8: *block=1; *bytes=1; return 1; case FOX_GGML_I16: *block=1; *bytes=2; return 1;
    case FOX_GGML_I32: *block=1; *bytes=4; return 1; case FOX_GGML_I64: case FOX_GGML_F64: *block=1; *bytes=8; return 1;
    case FOX_GGML_Q4_0: *block=32; *bytes=18; return 1; case FOX_GGML_Q4_1: *block=32; *bytes=20; return 1;
    case FOX_GGML_Q5_0: *block=32; *bytes=22; return 1; case FOX_GGML_Q5_1: *block=32; *bytes=24; return 1;
    case FOX_GGML_Q8_0: *block=32; *bytes=34; return 1; case FOX_GGML_Q2_K: *block=256; *bytes=84; return 1;
    case FOX_GGML_Q3_K: *block=256; *bytes=110; return 1; case FOX_GGML_Q4_K: *block=256; *bytes=144; return 1;
    case FOX_GGML_Q5_K: *block=256; *bytes=176; return 1; case FOX_GGML_Q6_K: *block=256; *bytes=210; return 1;
    case FOX_GGML_TQ1_0: *block=256; *bytes=54; return 1; case FOX_GGML_TQ2_0: *block=256; *bytes=66; return 1;
    default: return 0;
    }
}

static void dispose(fox_gguf *g) { size_t i; if (!g) return; for(i=0;i<g->n_meta;i++){free(g->meta[i].key);free(g->meta[i].string);} free(g->meta); for(i=0;i<g->n_tensors;i++) free((char *)g->tensors[i].name); free(g->tensors); free(g); }

fox_status fox_gguf_open(const char *path, fox_gguf **out) {
    reader r; char magic[4]; uint64_t nmeta, ntensor, n, count, data_start, total, elements; uint32_t v, type, dims, align=32; size_t i,j; fox_gguf *g; int32_t it; uint64_t sz; int64_t fsz;
    if (!path || !out) return FOX_ERR_ARG; *out=NULL; r.f=fox_file_open_read(path,FOX_OPEN_SEQ,NULL); if(!r.f) return FOX_ERR_IO; fsz=fox_file_size(r.f); if(fsz<0){fox_file_close(r.f);return FOX_ERR_IO;} r.size=(uint64_t)fsz; r.pos=0;
    g=(fox_gguf *)calloc(1,sizeof(*g)); if(!g){fox_file_close(r.f);return FOX_ERR_NOMEM;} if(!read_bytes(&r,magic,4)||memcmp(magic,"GGUF",4)!=0||!u32(&r,&v)||!u64(&r,&ntensor)||!u64(&r,&nmeta)||v<2||v>3||ntensor>GGUF_MAX_ITEMS||nmeta>GGUF_MAX_META){dispose(g);fox_file_close(r.f);return FOX_ERR_FORMAT;} g->version=v; g->n_meta=(size_t)nmeta; g->n_tensors=(size_t)ntensor; g->meta=calloc(g->n_meta,sizeof(*g->meta)); g->tensors=calloc(g->n_tensors,sizeof(*g->tensors)); if((g->n_meta&&!g->meta)||(g->n_tensors&&!g->tensors)){dispose(g);fox_file_close(r.f);return FOX_ERR_NOMEM;}
    for(i=0;i<g->n_meta;i++){if(str_read(&r,&g->meta[i].key)!=1||!i32(&r,&it)||it<0||it>12){dispose(g);fox_file_close(r.f);return FOX_ERR_FORMAT;} for(j=0;j<i;j++)if(strcmp(g->meta[i].key,g->meta[j].key)==0){dispose(g);fox_file_close(r.f);return FOX_ERR_FORMAT;} g->meta[i].type=(fox_gguf_value_type)it; if(value_size(g->meta[i].type,&sz)){uint8_t raw[8]={0}; if(!read_bytes(&r,raw,(size_t)sz)){dispose(g);fox_file_close(r.f);return FOX_ERR_FORMAT;} for(j=0;j<sz;j++)g->meta[i].u64|=(uint64_t)raw[j]<<(8*j);} else if(g->meta[i].type==FOX_GGUF_STRING){if(str_read(&r,&g->meta[i].string)!=1){dispose(g);fox_file_close(r.f);return FOX_ERR_FORMAT;}} else if(g->meta[i].type==FOX_GGUF_ARRAY){if(!i32(&r,&it)||it<0||it>12||!u64(&r,&count)||count>GGUF_MAX_ITEMS){dispose(g);fox_file_close(r.f);return FOX_ERR_FORMAT;} if((fox_gguf_value_type)it==FOX_GGUF_STRING){for(j=0;j<count;j++){char *item=NULL;if(str_read(&r,&item)!=1){free(item);dispose(g);fox_file_close(r.f);return FOX_ERR_FORMAT;}free(item);}} else if(value_size((fox_gguf_value_type)it,&sz)){if(!mul_ok(count,sz,&total)||!skip(&r,total)){dispose(g);fox_file_close(r.f);return FOX_ERR_FORMAT;}} else {dispose(g);fox_file_close(r.f);return FOX_ERR_FORMAT;}} else {dispose(g);fox_file_close(r.f);return FOX_ERR_FORMAT;} if(strcmp(g->meta[i].key,"general.alignment")==0&&g->meta[i].type==FOX_GGUF_UINT32) align=(uint32_t)g->meta[i].u64;}
    if(align==0||(align&(align-1))!=0||align>4096){dispose(g);fox_file_close(r.f);return FOX_ERR_FORMAT;} g->alignment=align;
    for(i=0;i<g->n_tensors;i++){char *name=NULL; if(str_read(&r,&name)!=1||!u32(&r,&dims)||dims==0||dims>FOX_GGUF_MAX_DIMS){free(name);dispose(g);fox_file_close(r.f);return FOX_ERR_FORMAT;} g->tensors[i].name=name;g->tensors[i].n_dims=dims; elements=1; for(j=0;j<dims;j++){uint64_t d;if(!u64(&r,&d)||d==0||!mul_ok(elements,d,&elements)){dispose(g);fox_file_close(r.f);return FOX_ERR_FORMAT;}g->tensors[i].ne[j]=d;} if(!u32(&r,&type)||type>35||!u64(&r,&n)||!tensor_layout((fox_ggml_type)type,&v,&dims)){dispose(g);fox_file_close(r.f);return FOX_ERR_UNSUPPORTED;} if(elements%v!=0||!mul_ok(elements/v,dims,&sz)){dispose(g);fox_file_close(r.f);return FOX_ERR_FORMAT;} g->tensors[i].type=(fox_ggml_type)type;g->tensors[i].offset=n;g->tensors[i].size_bytes=sz; }
    if(!add_ok(r.pos,align-1,&total)){dispose(g);fox_file_close(r.f);return FOX_ERR_FORMAT;} data_start=total&~((uint64_t)align-1); if(data_start<r.pos||data_start>r.size){dispose(g);fox_file_close(r.f);return FOX_ERR_FORMAT;} g->data_offset=data_start; for(i=0;i<g->n_tensors;i++){if(!add_ok(data_start,g->tensors[i].offset,&total)||!add_ok(total,g->tensors[i].size_bytes,&total)||total>r.size){dispose(g);fox_file_close(r.f);return FOX_ERR_FORMAT;}} fox_file_close(r.f); *out=g; return FOX_OK;
}

void fox_gguf_close(fox_gguf *g){dispose(g);} uint32_t fox_gguf_version(const fox_gguf*g){return g?g->version:0;} size_t fox_gguf_metadata_count(const fox_gguf*g){return g?g->n_meta:0;} size_t fox_gguf_tensor_count(const fox_gguf*g){return g?g->n_tensors:0;}
size_t fox_gguf_find(const fox_gguf*g,const char*k){size_t i;if(!g||!k)return SIZE_MAX;for(i=0;i<g->n_meta;i++)if(strcmp(g->meta[i].key,k)==0)return i;return SIZE_MAX;}
fox_gguf_value_type fox_gguf_metadata_type(const fox_gguf*g,size_t i){return g&&i<g->n_meta?g->meta[i].type:FOX_GGUF_ARRAY;}
fox_status fox_gguf_get_u32(const fox_gguf*g,size_t i,uint32_t*out){if(!g||!out)return FOX_ERR_ARG;if(i>=g->n_meta)return FOX_ERR_NOTFOUND;if(g->meta[i].type!=FOX_GGUF_UINT32)return FOX_ERR_FORMAT;*out=(uint32_t)g->meta[i].u64;return FOX_OK;}
fox_status fox_gguf_get_string(const fox_gguf*g,size_t i,const char**out){if(!g||!out)return FOX_ERR_ARG;if(i>=g->n_meta)return FOX_ERR_NOTFOUND;if(g->meta[i].type!=FOX_GGUF_STRING)return FOX_ERR_FORMAT;*out=g->meta[i].string;return FOX_OK;}
fox_status fox_gguf_tensor_at(const fox_gguf*g,size_t i,fox_gguf_tensor*out){if(!g||!out)return FOX_ERR_ARG;if(i>=g->n_tensors)return FOX_ERR_NOTFOUND;*out=g->tensors[i];return FOX_OK;}
size_t fox_gguf_tensor_find(const fox_gguf*g,const char*n){size_t i;if(!g||!n)return SIZE_MAX;for(i=0;i<g->n_tensors;i++)if(strcmp(g->tensors[i].name,n)==0)return i;return SIZE_MAX;}
const char *fox_ggml_type_name(fox_ggml_type t){switch(t){case FOX_GGML_F32:return "F32";case FOX_GGML_F16:return "F16";case FOX_GGML_TQ1_0:return "TQ1_0";case FOX_GGML_TQ2_0:return "TQ2_0";default:return "unknown";}}
uint32_t fox_ggml_type_block_size(fox_ggml_type t){uint32_t b,n;return tensor_layout(t,&b,&n)?b:0;} uint32_t fox_ggml_type_block_bytes(fox_ggml_type t){uint32_t b,n;return tensor_layout(t,&b,&n)?n:0;}
