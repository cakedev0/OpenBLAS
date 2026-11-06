/* Split-K for gemm.c and syrk.c.

   OpenBLAS splits gemm and syrk over the tiles of C, so a small C (X.T @ Y with
   few columns and many rows) only gets as many threads as it has tiles. When
   each thread's share of C is under 128 x 128 and k is long, run one
   single-threaded driver call per k chunk instead. Chunk 0 writes C with the
   user's beta, the others write private buffers with beta = 0, and a second
   exec_blas adds the buffers to C. */

#ifndef SPLITK_H
#define SPLITK_H

#include <math.h>

typedef int (*splitk_driver_t)(blas_arg_t *, BLASLONG *, BLASLONG *, IFLOAT *, IFLOAT *, BLASLONG);

typedef struct {
  blas_arg_t args;  /* first member: exec_blas passes &chunk as the args */
  splitk_driver_t driver;
  BLASLONG k0;      /* offset of this chunk in k */
  BLASLONG range[2];
  FLOAT *c, *buf;   /* for the reduction */
  BLASLONG m, nbuf, ldc;
  int tri;
} splitk_chunk_t;

enum { SPLITK_FULL, SPLITK_UPPER, SPLITK_LOWER };

/* Number of k chunks, or 1 to keep the usual M x N split. */
static BLASLONG splitk_nchunks(BLASLONG size_c, BLASLONG k, int nthreads) {
  BLASLONG p = nthreads;
  if (nthreads < 2 || size_c >= (BLASLONG)nthreads * 128 * 128) return 1;
  /* Chunks of 512 amortize packing. When the M x N split would leave each
     thread less than 64 x 64 of C, chunks of 256 still beat it, and one chunk
     per thread keeps the load even on cores shared by two threads. */
  int tiny = size_c < (BLASLONG)nthreads * 64 * 64;
  if (k / p < (tiny ? 256 : 512)) p = k / 512;
  /* Keep the partial products under 24 MiB: glibc maps larger blocks anew
     on every call, and the page faults cost more than the split gains. */
  BLASLONG cap = (24L << 20) / (size_c * (BLASLONG)sizeof(FLOAT));
  if (p - 1 > cap) p = cap + 1;
  /* With fewer chunks than threads, the M x N split is faster unless C is tiny. */
  if (p < nthreads && (2 * p < nthreads || !tiny)) return 1;
  return p;
}

static int splitk_worker(blas_arg_t *args, BLASLONG *range_m, BLASLONG *range_n,
                         IFLOAT *sa, IFLOAT *sb, BLASLONG mypos) {
  splitk_chunk_t *chunk = (splitk_chunk_t *)args;
  return chunk->driver(&chunk->args, NULL, NULL, sa, sb, 0);
}

static int splitk_reduce(blas_arg_t *args, BLASLONG *range_m, BLASLONG *range_n,
                         IFLOAT *sa, IFLOAT *sb, BLASLONG mypos) {
  splitk_chunk_t *chunk = (splitk_chunk_t *)args;
  BLASLONG m = chunk->m;
  for (BLASLONG j = range_n[0]; j < range_n[1]; j++) {
    BLASLONG i0 = chunk->tri == SPLITK_LOWER ? j : 0;
    BLASLONG i1 = chunk->tri == SPLITK_UPPER ? j + 1 : m;
    for (BLASLONG p = 0; p < chunk->nbuf; p++)
      AXPYU_K(i1 - i0, 0, 0, ONE, chunk->buf + p * m * chunk->args.n + j * m + i0, 1,
              chunk->c + j * chunk->ldc + i0, 1, NULL, 0);
  }
  return 0;
}

/* One allocation holds p chunks, then p queue entries (an array: the OpenMP
   exec_blas indexes them and ignores next), then p - 1 partial products. */
#define SPLITK_QUEUE(chunks, p) ((blas_queue_t *)((chunks) + (p)))
#define SPLITK_BUF(chunks, p) ((FLOAT *)(SPLITK_QUEUE(chunks, p) + (p)))

static void splitk_queue(splitk_chunk_t *chunks, BLASLONG p, BLASLONG count, void *routine, IFLOAT *sa, IFLOAT *sb) {
#ifdef DOUBLE
  int mode = BLAS_DOUBLE | BLAS_REAL;
#else
  int mode = BLAS_SINGLE | BLAS_REAL;
#endif
  blas_queue_t *queue = SPLITK_QUEUE(chunks, p);
  for (BLASLONG i = 0; i < count; i++) {
    blas_queue_t *q = &queue[i];
    q->mode = mode;
    q->routine = routine;
    q->args = (blas_arg_t *)&chunks[i];
    q->range_m = NULL;
    q->range_n = chunks[i].range;
    q->sa = NULL;
    q->sb = NULL;
    q->next = i + 1 < count ? &queue[i + 1] : NULL;
  }
  queue[0].sa = sa;
  queue[0].sb = sb;
  exec_blas(count, queue);
}

/* Returns p chunks of k with the arguments common to gemm and syrk set and the
   level 3 lock held, or NULL to use the usual path. Does not wait for the lock:
   concurrent callers are faster on the usual path than taking turns here. */
static splitk_chunk_t *splitk_begin(blas_arg_t *args, splitk_driver_t driver, BLASLONG p) {
  static FLOAT zero = ZERO;
  BLASLONG i, k0 = 0, size_c = args->m * args->n;
  splitk_chunk_t *chunks;
  FLOAT *buf;
  if (p < 2 || !blas_level3_thread_try_enter()) return NULL;
  chunks = (splitk_chunk_t *)malloc(p * (sizeof(splitk_chunk_t) + sizeof(blas_queue_t)) +
                                    (p - 1) * size_c * sizeof(FLOAT));
  if (chunks == NULL) {
    blas_level3_thread_leave();
    return NULL;
  }
  buf = SPLITK_BUF(chunks, p);
  for (i = 0; i < p; i++) {
    BLASLONG kw = (args->k - k0) / (p - i);
    chunks[i].args = *args;
    chunks[i].args.k = kw;
    chunks[i].args.nthreads = 1;
    if (i > 0) {
      chunks[i].args.c = buf + (i - 1) * size_c;
      chunks[i].args.ldc = args->m;
      chunks[i].args.beta = &zero;
    }
    chunks[i].driver = driver;
    chunks[i].k0 = k0;
    k0 += kw;
  }
  return chunks;
}

/* Runs the chunks, then adds their buffers to C. */
static void splitk_finish(blas_arg_t *args, splitk_chunk_t *chunks, BLASLONG p, int tri,
                          IFLOAT *sa, IFLOAT *sb) {
  BLASLONG i, n = args->n, nr = MIN(p, n), prev = 0;
  splitk_queue(chunks, p, p, (void *)splitk_worker, sa, sb);

  /* Column ranges with equal area: column j of the upper triangle holds j + 1
     entries, of the lower n - j. */
  for (i = 0; i < nr; i++) {
    double f = (double)(i + 1) / nr;
    BLASLONG end = (BLASLONG)((tri == SPLITK_UPPER ? sqrt(f)
                               : tri == SPLITK_LOWER ? 1.0 - sqrt(1.0 - f) : f) * n);
    end = i + 1 == nr ? n : MAX(prev, MIN(n, end));
    chunks[i].range[0] = prev;
    chunks[i].range[1] = end;
    chunks[i].args = *args;
    chunks[i].c = (FLOAT *)args->c;
    chunks[i].buf = SPLITK_BUF(chunks, p);
    chunks[i].m = args->m;
    chunks[i].nbuf = p - 1;
    chunks[i].ldc = args->ldc;
    chunks[i].tri = tri;
    prev = end;
  }
  splitk_queue(chunks, p, nr, (void *)splitk_reduce, sa, sb);
  free(chunks);
  blas_level3_thread_leave();
}

#endif
