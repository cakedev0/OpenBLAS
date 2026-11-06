#include "openblas_utest.h"
#include <cblas.h>
#include <stdlib.h>
#include <string.h>

#ifndef NAN
#define NAN 0.0/0.0
#endif

/* Threaded syrk with fewer columns than threads * switch_ratio, where the
 * driver picks its thread count from the shape. Small integer inputs keep
 * every sum exact, so results must equal the reference bit for bit. */

#define NTHREADS 16
#define PAD 3

#define DEFINE_SYRK_CHECK(NAME, TYPE, CBLAS)                                  \
    static int NAME(enum CBLAS_UPLO uplo, enum CBLAS_TRANSPOSE trans, int n,  \
                    int k, TYPE beta)                                         \
    {                                                                         \
        int ldc = n + PAD, ldk = trans == CblasNoTrans ? n : k;               \
        int ncol = trans == CblasNoTrans ? k : n;                             \
        TYPE *a = malloc(sizeof(TYPE) * ldk * ncol);                          \
        TYPE *c = malloc(sizeof(TYPE) * ldc * n);                             \
        TYPE *ref = malloc(sizeof(TYPE) * ldc * n);                           \
        int i, j, l, bad = 0, threads = openblas_get_num_threads();           \
        for (i = 0; i < ldk * ncol; i++) a[i] = (i * 7 + i / 5) % 5 - 2;      \
        for (i = 0; i < ldc * n; i++) c[i] = i % 11;                          \
        memcpy(ref, c, sizeof(TYPE) * ldc * n);                               \
        for (j = 0; j < n; j++) {                                             \
            int i0 = uplo == CblasUpper ? 0 : j;                              \
            int i1 = uplo == CblasUpper ? j + 1 : n;                          \
            for (i = i0; i < i1; i++) {                                       \
                TYPE s = 0;                                                   \
                for (l = 0; l < k; l++)                                       \
                    s += trans == CblasNoTrans                                \
                             ? a[i + l * ldk] * a[j + l * ldk]                \
                             : a[l + i * ldk] * a[l + j * ldk];               \
                ref[i + j * ldc] = s * -2 + beta * ref[i + j * ldc];          \
            }                                                                 \
        }                                                                     \
        if (beta == 0)                                                        \
            for (j = 0; j < n; j++)                                           \
                for (i = uplo == CblasUpper ? 0 : j;                          \
                     i < (uplo == CblasUpper ? j + 1 : n); i++)               \
                    c[i + j * ldc] = NAN;                                   \
        openblas_set_num_threads(NTHREADS);                                   \
        CBLAS(CblasColMajor, uplo, trans, n, k, -2, a, ldk, beta, c, ldc);    \
        openblas_set_num_threads(threads);                                    \
        for (i = 0; i < ldc * n; i++) bad += c[i] != ref[i];                  \
        free(a);                                                              \
        free(c);                                                              \
        free(ref);                                                            \
        return bad;                                                           \
    }

#ifdef BUILD_DOUBLE
DEFINE_SYRK_CHECK(check_dsyrk, double, cblas_dsyrk)

CTEST(syrk_threads, dsyrk)
{
    enum CBLAS_UPLO uplos[] = {CblasUpper, CblasLower};
    enum CBLAS_TRANSPOSE transs[] = {CblasNoTrans, CblasTrans};
    int u, t;
    for (u = 0; u < 2; u++)
        for (t = 0; t < 2; t++) {
            ASSERT_EQUAL(0, check_dsyrk(uplos[u], transs[t], 100, 2000, 0));
            ASSERT_EQUAL(0, check_dsyrk(uplos[u], transs[t], 100, 2000, 0.5));
        }
}
#endif

#ifdef BUILD_SINGLE
DEFINE_SYRK_CHECK(check_ssyrk, float, cblas_ssyrk)

CTEST(syrk_threads, ssyrk)
{
    enum CBLAS_UPLO uplos[] = {CblasUpper, CblasLower};
    enum CBLAS_TRANSPOSE transs[] = {CblasNoTrans, CblasTrans};
    int u, t;
    for (u = 0; u < 2; u++)
        for (t = 0; t < 2; t++) {
            ASSERT_EQUAL(0, check_ssyrk(uplos[u], transs[t], 200, 1000, 0));
            ASSERT_EQUAL(0, check_ssyrk(uplos[u], transs[t], 200, 1000, 0.5));
        }
}
#endif
