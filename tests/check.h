#ifndef FOX_TEST_CHECK_H
#define FOX_TEST_CHECK_H

#include <stdio.h>

static int fox_test_failures = 0;
static int fox_test_count = 0;

#define CHECK(cond, msg)                                                      \
    do {                                                                      \
        fox_test_count++;                                                     \
        if (cond) {                                                           \
            printf("ok %d - %s\n", fox_test_count, (msg));                    \
        } else {                                                              \
            printf("not ok %d - %s (%s:%d)\n", fox_test_count, (msg),         \
                   __FILE__, __LINE__);                                       \
            fox_test_failures++;                                              \
        }                                                                     \
    } while (0)

#define CHECK_DONE()                                                          \
    do {                                                                      \
        printf("1..%d\n", fox_test_count);                                    \
        if (fox_test_failures) {                                              \
            printf("# %d of %d checks failed\n", fox_test_failures,           \
                   fox_test_count);                                           \
            return 1;                                                         \
        }                                                                     \
        return 0;                                                             \
    } while (0)

#endif
