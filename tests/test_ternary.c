#include "fox/fox.h"
#include "check.h"

#include <stdint.h>

int main(void)
{
    int8_t x[512]; uint8_t tq1[54*2]={0}; uint8_t tq2[66*2]={0}; float a=0.0f,b=0.0f; size_t i;
    for(i=0;i<512;i++) x[i]=1;
    for(i=0;i<48;i++) tq1[i]=242;
    for(i=0;i<4;i++) tq1[48+i]=0xaa;
    tq1[52]=0x00; tq1[53]=0x3c; tq1[106]=0x00; tq1[107]=0x3c;
    for(i=0;i<64;i++) tq2[i]=0xaa;
    tq2[64]=0x00; tq2[65]=0x3c; tq2[130]=0x00; tq2[131]=0x3c;
    CHECK(fox_tq1_dot_i8(x,tq1,512,&a)==FOX_OK&&a==512.0f,"TQ1 scalar dot matches all-positive block");
    CHECK(fox_tq2_dot_i8_scalar(x,tq2,512,&a)==FOX_OK&&a==512.0f,"TQ2 scalar dot matches all-positive block");
    CHECK(fox_tq2_dot_i8(x,tq2,512,&b)==FOX_OK&&b==a,"TQ2 dispatch matches scalar reference");
    CHECK(fox_tq1_dot_i8(x,tq1,255,&a)==FOX_ERR_ARG,"partial TQ1 blocks are rejected");
    CHECK(fox_tq2_dot_i8(NULL,tq2,256,&a)==FOX_ERR_ARG,"null activation is rejected");
    CHECK_DONE();
}
