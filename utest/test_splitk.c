#include "openblas_utest.h"
#include <cblas.h>
#include <stdlib.h>
#include <string.h>

#ifndef NAN
#define NAN 0.0/0.0
#endif

/* gemm and syrk with a small C and a long k, where k is split between threads
 * and the partial products are added at the end. Small integer inputs keep
 * every sum exact, so results must equal the reference exactly, including the
 * padding between columns and the triangle syrk must not touch. */

#define NTHREADS 8
#define PAD 3
#define M 37
#define N 45
#define SYRK_N 64
#define K 2048

#define IS_NT(T) ((T) == CblasNoTrans)

#define DEFINE_CHECKS(PREFIX, TYPE, CBLAS_GEMM, CBLAS_SYRK)                    \
    static TYPE PREFIX##_entry(const TYPE *a, int ld, int trans, int major,    \
                               int i, int l)                                   \
    {                                                                          \
        /* Element (i, l) of op(A), whatever the transpose and storage. */     \
        int r = trans ? l : i, c = trans ? i : l;                              \
        return major == CblasColMajor ? a[r + c * ld] : a[c + r * ld];         \
    }                                                                          \
    static int PREFIX##_gemm(enum CBLAS_ORDER major, int ta, int tb, TYPE beta)\
    {                                                                          \
        enum CBLAS_TRANSPOSE ctrans[2] = {CblasNoTrans, CblasTrans};           \
        int nouter = major == CblasColMajor ? N : M;                           \
        int ninner = major == CblasColMajor ? M : N;                           \
        int lda = (major == CblasColMajor) == !ta ? M + PAD : K + PAD;         \
        int ldb = (major == CblasColMajor) == !tb ? K + PAD : N + PAD;         \
        int na = (major == CblasColMajor) == !ta ? K : M;                      \
        int nb = (major == CblasColMajor) == !tb ? N : K;                      \
        int ldc = ninner + PAD, threads = openblas_get_num_threads();          \
        int i, j, l, bad = 0;                                                  \
        TYPE *a = malloc(sizeof(TYPE) * lda * na);                             \
        TYPE *b = malloc(sizeof(TYPE) * ldb * nb);                             \
        TYPE *c = malloc(sizeof(TYPE) * ldc * nouter);                         \
        TYPE *ref = malloc(sizeof(TYPE) * ldc * nouter);                       \
        for (i = 0; i < lda * na; i++) a[i] = (i * 7 + i / 5) % 5 - 2;         \
        for (i = 0; i < ldb * nb; i++) b[i] = (i * 3 + i / 7) % 5 - 2;         \
        for (i = 0; i < ldc * nouter; i++) c[i] = i % 11;                      \
        memcpy(ref, c, sizeof(TYPE) * ldc * nouter);                           \
        for (j = 0; j < N; j++)                                                \
            for (i = 0; i < M; i++) {                                          \
                TYPE s = 0;                                                    \
                TYPE *r = major == CblasColMajor ? &ref[i + j * ldc]           \
                                                 : &ref[j + i * ldc];          \
                for (l = 0; l < K; l++)                                        \
                    s += PREFIX##_entry(a, lda, ta, major, i, l) *             \
                         PREFIX##_entry(b, ldb, !tb, major, j, l);             \
                *r = s * -2 + beta * *r;                                       \
            }                                                                  \
        if (beta == 0)                                                         \
            for (j = 0; j < nouter; j++)                                       \
                for (i = 0; i < ninner; i++) c[i + j * ldc] = NAN;             \
        openblas_set_num_threads(NTHREADS);                                    \
        CBLAS_GEMM(major, ctrans[ta], ctrans[tb], M, N, K, -2, a, lda, b, ldb, \
                   beta, c, ldc);                                              \
        openblas_set_num_threads(threads);                                     \
        for (i = 0; i < ldc * nouter; i++) bad += c[i] != ref[i];              \
        free(a);                                                               \
        free(b);                                                               \
        free(c);                                                               \
        free(ref);                                                             \
        return bad;                                                            \
    }                                                                          \
    static int PREFIX##_syrk(enum CBLAS_UPLO uplo, enum CBLAS_TRANSPOSE trans, \
                             TYPE beta)                                        \
    {                                                                          \
        int nt = IS_NT(trans), lda = (nt ? SYRK_N : K) + PAD;                  \
        int na = nt ? K : SYRK_N, ldc = SYRK_N + PAD;                          \
        int threads = openblas_get_num_threads(), i, j, l, bad = 0;            \
        TYPE *a = malloc(sizeof(TYPE) * lda * na);                             \
        TYPE *c = malloc(sizeof(TYPE) * ldc * SYRK_N);                         \
        TYPE *ref = malloc(sizeof(TYPE) * ldc * SYRK_N);                       \
        for (i = 0; i < lda * na; i++) a[i] = (i * 7 + i / 5) % 5 - 2;         \
        for (i = 0; i < ldc * SYRK_N; i++) c[i] = i % 11;                      \
        memcpy(ref, c, sizeof(TYPE) * ldc * SYRK_N);                           \
        for (j = 0; j < SYRK_N; j++)                                           \
            for (i = uplo == CblasUpper ? 0 : j;                               \
                 i < (uplo == CblasUpper ? j + 1 : SYRK_N); i++) {             \
                TYPE s = 0;                                                    \
                for (l = 0; l < K; l++)                                        \
                    s += nt ? a[i + l * lda] * a[j + l * lda]                  \
                            : a[l + i * lda] * a[l + j * lda];                 \
                ref[i + j * ldc] = s * -2 + beta * ref[i + j * ldc];           \
                if (beta == 0) c[i + j * ldc] = NAN;                           \
            }                                                                  \
        openblas_set_num_threads(NTHREADS);                                    \
        CBLAS_SYRK(CblasColMajor, uplo, trans, SYRK_N, K, -2, a, lda, beta, c, \
                   ldc);                                                       \
        openblas_set_num_threads(threads);                                     \
        for (i = 0; i < ldc * SYRK_N; i++) bad += c[i] != ref[i];              \
        free(a);                                                               \
        free(c);                                                               \
        free(ref);                                                             \
        return bad;                                                            \
    }

#define RUN_GEMM(PREFIX)                                                       \
    do {                                                                       \
        int ta, tb;                                                            \
        for (ta = 0; ta < 2; ta++)                                             \
            for (tb = 0; tb < 2; tb++) {                                       \
                ASSERT_EQUAL(0, PREFIX##_gemm(CblasColMajor, ta, tb, 0));      \
                ASSERT_EQUAL(0, PREFIX##_gemm(CblasColMajor, ta, tb, 0.5));    \
                ASSERT_EQUAL(0, PREFIX##_gemm(CblasRowMajor, ta, tb, 0.5));    \
            }                                                                  \
    } while (0)

#define RUN_SYRK(PREFIX)                                                       \
    do {                                                                       \
        enum CBLAS_UPLO uplos[] = {CblasUpper, CblasLower};                    \
        enum CBLAS_TRANSPOSE transs[] = {CblasNoTrans, CblasTrans};            \
        int u, t;                                                              \
        for (u = 0; u < 2; u++)                                                \
            for (t = 0; t < 2; t++) {                                          \
                ASSERT_EQUAL(0, PREFIX##_syrk(uplos[u], transs[t], 0));        \
                ASSERT_EQUAL(0, PREFIX##_syrk(uplos[u], transs[t], 0.5));      \
            }                                                                  \
    } while (0)

#ifdef BUILD_DOUBLE
DEFINE_CHECKS(d, double, cblas_dgemm, cblas_dsyrk)
CTEST(splitk, dgemm) { RUN_GEMM(d); }
CTEST(splitk, dsyrk) { RUN_SYRK(d); }
#endif

#ifdef BUILD_SINGLE
DEFINE_CHECKS(s, float, cblas_sgemm, cblas_ssyrk)
CTEST(splitk, sgemm) { RUN_GEMM(s); }
CTEST(splitk, ssyrk) { RUN_SYRK(s); }
#endif
