/*********************************************************************/
/* Copyright 2009, 2010 The University of Texas at Austin.           */
/* All rights reserved.                                              */
/*                                                                   */
/* Redistribution and use in source and binary forms, with or        */
/* without modification, are permitted provided that the following   */
/* conditions are met:                                               */
/*                                                                   */
/*   1. Redistributions of source code must retain the above         */
/*      copyright notice, this list of conditions and the following  */
/*      disclaimer.                                                  */
/*                                                                   */
/*   2. Redistributions in binary form must reproduce the above      */
/*      copyright notice, this list of conditions and the following  */
/*      disclaimer in the documentation and/or other materials       */
/*      provided with the distribution.                              */
/*                                                                   */
/*    THIS  SOFTWARE IS PROVIDED  BY THE  UNIVERSITY OF  TEXAS AT    */
/*    AUSTIN  ``AS IS''  AND ANY  EXPRESS OR  IMPLIED WARRANTIES,    */
/*    INCLUDING, BUT  NOT LIMITED  TO, THE IMPLIED  WARRANTIES OF    */
/*    MERCHANTABILITY  AND FITNESS FOR  A PARTICULAR  PURPOSE ARE    */
/*    DISCLAIMED.  IN  NO EVENT SHALL THE UNIVERSITY  OF TEXAS AT    */
/*    AUSTIN OR CONTRIBUTORS BE  LIABLE FOR ANY DIRECT, INDIRECT,    */
/*    INCIDENTAL,  SPECIAL, EXEMPLARY,  OR  CONSEQUENTIAL DAMAGES    */
/*    (INCLUDING, BUT  NOT LIMITED TO,  PROCUREMENT OF SUBSTITUTE    */
/*    GOODS  OR  SERVICES; LOSS  OF  USE,  DATA,  OR PROFITS;  OR    */
/*    BUSINESS INTERRUPTION) HOWEVER CAUSED  AND ON ANY THEORY OF    */
/*    LIABILITY, WHETHER  IN CONTRACT, STRICT  LIABILITY, OR TORT    */
/*    (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY WAY OUT    */
/*    OF  THE  USE OF  THIS  SOFTWARE,  EVEN  IF ADVISED  OF  THE    */
/*    POSSIBILITY OF SUCH DAMAGE.                                    */
/*                                                                   */
/* The views and conclusions contained in the software and           */
/* documentation are those of the authors and should not be          */
/* interpreted as representing official policies, either expressed   */
/* or implied, of The University of Texas at Austin.                 */
/*********************************************************************/

#include <stdio.h>
#include <ctype.h>
#include "common.h"
#ifdef FUNCTION_PROFILE
#include "functable.h"
#endif

#ifndef COMPLEX
#define SMP_THRESHOLD_MIN 109944.
#ifdef XDOUBLE
#define ERROR_NAME "QSYRK "
#elif defined(DOUBLE)
#define ERROR_NAME "DSYRK "
#else
#define ERROR_NAME "SSYRK "
#endif
#else
#define SMP_THRESHOLD_MIN 14824.
#ifndef HEMM
#ifdef XDOUBLE
#define ERROR_NAME "XSYRK "
#elif defined(DOUBLE)
#define ERROR_NAME "ZSYRK "
#else
#define ERROR_NAME "CSYRK "
#endif
#else
#ifdef XDOUBLE
#define ERROR_NAME "XHERK "
#elif defined(DOUBLE)
#define ERROR_NAME "ZHERK "
#else
#define ERROR_NAME "CHERK "
#endif
#endif
#endif

#ifndef GEMM_MULTITHREAD_THRESHOLD
#define GEMM_MULTITHREAD_THRESHOLD 4
#endif

static int (*syrk[])(blas_arg_t *, BLASLONG *, BLASLONG *, FLOAT *, FLOAT *, BLASLONG) = {
#ifndef HEMM
  SYRK_UN, SYRK_UC, SYRK_LN, SYRK_LC,
#if defined(SMP) && !defined(USE_SIMPLE_THREADED_LEVEL3)
  SYRK_THREAD_UN, SYRK_THREAD_UC, SYRK_THREAD_LN, SYRK_THREAD_LC,
#endif
#else
  HERK_UN, HERK_UC, HERK_LN, HERK_LC,
#if defined(SMP) && !defined(USE_SIMPLE_THREADED_LEVEL3)
  HERK_THREAD_UN, HERK_THREAD_UC, HERK_THREAD_LN, HERK_THREAD_LC,
#endif
#endif
};

#if defined(SMP) && !defined(COMPLEX) && !defined(XDOUBLE) && !defined(USE_SIMPLE_THREADED_LEVEL3)
#define SYRK_SPLITK 1
#include <math.h>

struct syrk_splitk_job {
  blas_arg_t args;  /* first member: exec_blas passes &job as the args */
  int (*fn)(blas_arg_t *, BLASLONG *, BLASLONG *, FLOAT *, FLOAT *, BLASLONG);
  FLOAT *buf;
  BLASLONG nbuf;
  int uplo;
};

/* Same as the split-K gemm in gemm.c, for X.T @ X with few columns. */
static BLASLONG syrk_splitk_nchunks(BLASLONG n, BLASLONG k, int nthreads) {
  BLASLONG p = nthreads;
  if (nthreads < 2 || n * n >= (BLASLONG)nthreads * 128 * 128) return 1;
  if (k / p < 512) p = k / 512;
  BLASLONG cap = (24L << 20) / (n * n * (BLASLONG)sizeof(FLOAT));
  if (p - 1 > cap) p = cap + 1;
  if (p < nthreads && (2 * p < nthreads || n * n >= (BLASLONG)nthreads * 64 * 64)) return 1;
  return p;
}

static int syrk_splitk_worker(blas_arg_t *args, BLASLONG *range_m, BLASLONG *range_n,
                              FLOAT *sa, FLOAT *sb, BLASLONG mypos) {
  struct syrk_splitk_job *job = (struct syrk_splitk_job *)args;
  return job->fn(&job->args, NULL, NULL, sa, sb, 0);
}

static int syrk_splitk_reduce(blas_arg_t *args, BLASLONG *range_m, BLASLONG *range_n,
                              FLOAT *sa, FLOAT *sb, BLASLONG mypos) {
  struct syrk_splitk_job *job = (struct syrk_splitk_job *)args;
  BLASLONG n = job->args.n, ldc = job->args.ldc;
  FLOAT *c = (FLOAT *)job->args.c;
  for (BLASLONG j = range_n[0]; j < range_n[1]; j++) {
    BLASLONG i0 = job->uplo ? j : 0, i1 = job->uplo ? n : j + 1;
    FLOAT *cj = c + j * ldc;
    for (BLASLONG p = 0; p < job->nbuf; p++) {
      FLOAT *bj = job->buf + p * n * n + j * n;
      for (BLASLONG i = i0; i < i1; i++) cj[i] += bj[i];
    }
  }
  return 0;
}

static int syrk_splitk(blas_arg_t *args, int uplo, int trans, FLOAT *sa, FLOAT *sb) {
  BLASLONG n = args->n, k = args->k;
  if (((FLOAT *)args->alpha)[0] == ZERO) return 0;
  BLASLONG p = syrk_splitk_nchunks(n, k, args->nthreads);
  if (p < 2) return 0;
  FLOAT *buf = (FLOAT *)malloc((size_t)(p - 1) * n * n * sizeof(FLOAT));
  if (buf == NULL) return 0;
  static FLOAT zero = ZERO;
#ifdef DOUBLE
  int mode = BLAS_DOUBLE | BLAS_REAL;
#else
  int mode = BLAS_SINGLE | BLAS_REAL;
#endif
  struct syrk_splitk_job jobs[MAX_CPU_NUMBER];
  blas_queue_t queue[MAX_CPU_NUMBER];
  BLASLONG range[MAX_CPU_NUMBER + 1];
  BLASLONG i, k0 = 0;
  for (i = 0; i < p; i++) {
    BLASLONG kw = (k - k0) / (p - i);
    jobs[i].args = *args;
    jobs[i].args.k = kw;
    jobs[i].args.nthreads = 1;
    /* N: A is n x k, chunk = columns k0.. ; T: A is k x n, chunk = rows k0.. */
    jobs[i].args.a = (FLOAT *)args->a + (trans ? k0 : k0 * args->lda);
    if (i > 0) {
      jobs[i].args.c = buf + (i - 1) * n * n;
      jobs[i].args.ldc = n;
      jobs[i].args.beta = &zero;
    }
    jobs[i].fn = syrk[(uplo << 1) | trans];
    queue[i].mode = mode;
    queue[i].routine = (void *)syrk_splitk_worker;
    queue[i].args = (blas_arg_t *)&jobs[i];
    queue[i].range_m = NULL;
    queue[i].range_n = NULL;
    queue[i].sa = NULL;
    queue[i].sb = NULL;
    queue[i].next = &queue[i + 1];
    k0 += kw;
  }
  queue[0].sa = sa;
  queue[0].sb = sb;
  queue[p - 1].next = NULL;
  exec_blas(p, queue);

  /* Column ranges with equal triangle area: column j of the upper triangle
     holds j + 1 entries, of the lower n - j. */
  BLASLONG nr = MIN(p, n);
  jobs[0].args = *args;
  jobs[0].buf = buf;
  jobs[0].nbuf = p - 1;
  jobs[0].uplo = uplo;
  range[0] = 0;
  for (i = 1; i < nr; i++) {
    double f = uplo ? 1.0 - sqrt(1.0 - (double)i / nr) : sqrt((double)i / nr);
    range[i] = MAX(range[i - 1], MIN(n, (BLASLONG)(f * n)));
  }
  range[nr] = n;
  for (i = 0; i < nr; i++) {
    queue[i].mode = mode;
    queue[i].routine = (void *)syrk_splitk_reduce;
    queue[i].args = (blas_arg_t *)&jobs[0];
    queue[i].range_m = NULL;
    queue[i].range_n = &range[i];
    queue[i].sa = NULL;
    queue[i].sb = NULL;
    queue[i].next = &queue[i + 1];
  }
  queue[0].sa = sa;
  queue[0].sb = sb;
  queue[nr - 1].next = NULL;
  exec_blas(nr, queue);
  free(buf);
  return 1;
}
#endif

OPENBLAS_EXPORT

#ifndef CBLAS

void NAME(char *UPLO, char *TRANS,
         blasint *N, blasint *K,
         FLOAT *alpha, FLOAT *a, blasint *ldA,
         FLOAT *beta,  FLOAT *c, blasint *ldC){

  char uplo_arg  = *UPLO;
  char trans_arg = *TRANS;

  blas_arg_t args;

  FLOAT *buffer;
  FLOAT *sa, *sb;

#ifdef SMP
  double NNK;
#ifdef USE_SIMPLE_THREADED_LEVEL3
#ifndef COMPLEX
#ifdef XDOUBLE
  int mode  =  BLAS_XDOUBLE | BLAS_REAL;
#elif defined(DOUBLE)
  int mode  =  BLAS_DOUBLE  | BLAS_REAL;
#else
  int mode  =  BLAS_SINGLE  | BLAS_REAL;
#endif
#else
#ifdef XDOUBLE
  int mode  =  BLAS_XDOUBLE | BLAS_COMPLEX;
#elif defined(DOUBLE)
  int mode  =  BLAS_DOUBLE  | BLAS_COMPLEX;
#else
  int mode  =  BLAS_SINGLE  | BLAS_COMPLEX;
#endif
#endif
#endif
#endif

  blasint info;
  int uplo;
  int trans;
  int nrowa;

  PRINT_DEBUG_NAME;

  args.n = *N;
  args.k = *K;

  args.a = (void *)a;
  args.c = (void *)c;

  args.lda = *ldA;
  args.ldc = *ldC;

  args.alpha = (void *)alpha;
  args.beta  = (void *)beta;

  TOUPPER(uplo_arg);
  TOUPPER(trans_arg);

  uplo  = -1;
  trans = -1;

  if (uplo_arg  == 'U') uplo  = 0;
  if (uplo_arg  == 'L') uplo  = 1;


#ifndef COMPLEX
  if (trans_arg == 'N') trans = 0;
  if (trans_arg == 'T') trans = 1;
  if (trans_arg == 'C') trans = 1;
#else
#ifdef HEMM
  if (trans_arg == 'N') trans = 0;
  if (trans_arg == 'C') trans = 1;
#else
  if (trans_arg == 'N') trans = 0;
  if (trans_arg == 'T') trans = 1;
#endif

#endif

  nrowa = args.n;
  if (trans & 1) nrowa = args.k;

  info = 0;

  if (args.ldc < MAX(1,args.n)) info = 10;
  if (args.lda < MAX(1,nrowa))  info =  7;
  if (args.k < 0)               info =  4;
  if (args.n < 0)               info =  3;
  if (trans < 0)                info =  2;
  if (uplo  < 0)                info =  1;

  if (info != 0) {
    BLASFUNC(xerbla)(ERROR_NAME, &info, sizeof(ERROR_NAME));
    return;
  }

#else

void CNAME(enum CBLAS_ORDER order, enum CBLAS_UPLO Uplo, enum CBLAS_TRANSPOSE Trans,
	   blasint n, blasint k,
#if !defined(COMPLEX) || defined(HEMM)
	   FLOAT alpha,
#else
	   void *valpha,
#endif
#if !defined(COMPLEX)
	   FLOAT *a, blasint lda,
#else
	   void *va, blasint lda,
#endif
#if !defined(COMPLEX) || defined(HEMM)
	   FLOAT beta,
#else
	   void *vbeta,
#endif
#if !defined(COMPLEX)
	   FLOAT *c, blasint ldc) {
#else
	   void *vc, blasint ldc) {
#endif

#ifdef COMPLEX
#if !defined(HEMM)
  FLOAT* alpha = (FLOAT*) valpha;
  FLOAT* beta = (FLOAT*) vbeta;
#endif
  FLOAT* a = (FLOAT*) va;
  FLOAT* c = (FLOAT*) vc;
#endif

  blas_arg_t args;
  int uplo, trans;
  blasint info, nrowa;

  FLOAT *buffer;
  FLOAT *sa, *sb;

#ifdef SMP
double NNK;

#ifdef USE_SIMPLE_THREADED_LEVEL3
#ifndef COMPLEX
#ifdef XDOUBLE
  int mode  =  BLAS_XDOUBLE | BLAS_REAL;
#elif defined(DOUBLE)
  int mode  =  BLAS_DOUBLE  | BLAS_REAL;
#else
  int mode  =  BLAS_SINGLE  | BLAS_REAL;
#endif
#else
#ifdef XDOUBLE
  int mode  =  BLAS_XDOUBLE | BLAS_COMPLEX;
#elif defined(DOUBLE)
  int mode  =  BLAS_DOUBLE  | BLAS_COMPLEX;
#else
  int mode  =  BLAS_SINGLE  | BLAS_COMPLEX;
#endif
#endif
#endif
#endif

  PRINT_DEBUG_CNAME;

  args.n = n;
  args.k = k;

  args.a = (void *)a;
  args.c = (void *)c;

  args.lda = lda;
  args.ldc = ldc;

#if !defined(COMPLEX) || defined(HEMM)
  args.alpha = (void *)&alpha;
  args.beta  = (void *)&beta;
#else
  args.alpha = (void *)alpha;
  args.beta  = (void *)beta;
#endif

  trans = -1;
  uplo  = -1;
  info  =  0;

  if (order == CblasColMajor) {
    if (Uplo == CblasUpper) uplo  = 0;
    if (Uplo == CblasLower) uplo  = 1;

    if (Trans == CblasNoTrans)     trans = 0;
#ifndef COMPLEX
    if (Trans == CblasTrans)       trans = 1;
    if (Trans == CblasConjNoTrans) trans = 0;
    if (Trans == CblasConjTrans)   trans = 1;
#elif !defined(HEMM)
    if (Trans == CblasTrans)       trans = 1;
#else
    if (Trans == CblasConjTrans)   trans = 1;
#endif

    info = -1;

    nrowa = args.n;
    if (trans & 1) nrowa = args.k;

    if (args.ldc < MAX(1,args.n)) info = 10;
    if (args.lda < MAX(1,nrowa))  info =  7;
    if (args.k < 0)               info =  4;
    if (args.n < 0)               info =  3;
    if (trans < 0)                info =  2;
    if (uplo  < 0)                info =  1;
  }

  if (order == CblasRowMajor) {
    if (Uplo == CblasUpper) uplo  = 1;
    if (Uplo == CblasLower) uplo  = 0;

    if (Trans == CblasNoTrans)     trans = 1;
#ifndef COMPLEX
    if (Trans == CblasTrans)       trans = 0;
    if (Trans == CblasConjNoTrans) trans = 1;
    if (Trans == CblasConjTrans)   trans = 0;
#elif !defined(HEMM)
    if (Trans == CblasTrans)       trans = 0;
#else
    if (Trans == CblasConjTrans)   trans = 0;
#endif

    info = -1;

    nrowa = args.n;
    if (trans & 1) nrowa = args.k;

    if (args.ldc < MAX(1,args.n)) info = 10;
    if (args.lda < MAX(1,nrowa))  info =  7;
    if (args.k < 0)               info =  4;
    if (args.n < 0)               info =  3;
    if (trans < 0)                info =  2;
    if (uplo  < 0)                info =  1;
  }

  if (info >= 0) {
    BLASFUNC(xerbla)(ERROR_NAME, &info, sizeof(ERROR_NAME));
    return;
  }

  if (args.n == 0) return;

#ifdef DYNAMIC_ARCH
extern char* gotoblas_corename(void);
#endif
	
#if !defined(COMPLEX) && !defined(DOUBLE) && !defined(BFLOAT16)  && !defined(HFLOAT16)
#if defined(ARCH_ARM64) && (defined(USE_SSYRK_KERNEL_DIRECT)||defined(DYNAMIC_ARCH))
#if defined(DYNAMIC_ARCH)
if (strcmp(gotoblas_corename(), "armv9sme") == 0
#if defined(__clang__)
 || strcmp(gotoblas_corename(), "vortexm4") == 0
#endif
)
#endif
  if (order == CblasRowMajor && n == ldc) {
    if (Trans == CblasNoTrans && k == lda) {
      (Uplo == CblasUpper ? SSYRK_DIRECT_ALPHA_BETA_UN : SSYRK_DIRECT_ALPHA_BETA_LN)(n, k, alpha, a, lda, beta, c, ldc);
      return;
    } else if (Trans == CblasTrans && n == lda){
      (Uplo == CblasUpper ? SSYRK_DIRECT_ALPHA_BETA_UT : SSYRK_DIRECT_ALPHA_BETA_LT)(n, k, alpha, a, lda, beta, c, ldc);
      return;
    }
  }
#endif
#endif

#endif


  IDEBUG_START;

  FUNCTION_PROFILE_START();

  buffer = (FLOAT *)blas_memory_alloc(0);
  if (!buffer) {
    info = -999;
    BLASFUNC(xerbla)(ERROR_NAME, &info, sizeof(ERROR_NAME));
    return;
  }

  sa = (FLOAT *)((BLASLONG)buffer + GEMM_OFFSET_A);
  sb = (FLOAT *)(((BLASLONG)sa + ((GEMM_P * GEMM_Q * COMPSIZE * SIZE + GEMM_ALIGN) & ~GEMM_ALIGN)) + GEMM_OFFSET_B);

#ifdef SMP
#ifdef USE_SIMPLE_THREADED_LEVEL3
  if (!trans){
    mode |= (BLAS_TRANSA_N | BLAS_TRANSB_T);
  } else {
    mode |= (BLAS_TRANSA_T | BLAS_TRANSB_N);
  }
  mode |= (uplo  << BLAS_UPLO_SHIFT);
#endif

  args.common = NULL;

  NNK = (double)(args.n+1)*(double)args.n*(double)args.k;
  if (NNK <= (SMP_THRESHOLD_MIN * GEMM_MULTITHREAD_THRESHOLD)) {
  args.nthreads = 1;
  } else {
  args.nthreads = num_cpu_avail(3);
  }

#ifdef SYRK_SPLITK
  if (args.nthreads > 1 && syrk_splitk(&args, uplo, trans, sa, sb)) {
  } else
#endif
  if (args.nthreads == 1) {
#endif

    (syrk[(uplo << 1) | trans ])(&args, NULL, NULL, sa, sb, 0);

#ifdef SMP
  } else {

#ifndef USE_SIMPLE_THREADED_LEVEL3

    (syrk[4 | (uplo << 1) | trans ])(&args, NULL, NULL, sa, sb, 0);

#else

    syrk_thread(mode, &args, NULL, NULL, syrk[(uplo << 1) | trans ], sa, sb, args.nthreads);

#endif

  }
#endif

 blas_memory_free(buffer);

  FUNCTION_PROFILE_END(COMPSIZE * COMPSIZE, args.n * args.k + args.n * args.n / 2, args.n * args.n * args.k);

  IDEBUG_END;

  return;
}
