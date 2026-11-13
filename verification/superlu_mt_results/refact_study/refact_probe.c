/*
 * Phase-0 probe: does SuperLU_MT's refact=YES actually save time on a
 * second factorization with the same sparsity pattern and new values?
 *
 * Not part of the project build.  Reads the flat KKT dump written by
 * dump_kkt.py, factors it, then produces a second factorization of a
 * value-perturbed matrix with the SAME pattern in one of three ways:
 *
 *   fresh     full get_perm_c + pdgstrf_init(refact=NO) + pdgstrf
 *   refact    pdgstrf_init(refact=YES) + pdgstrf(refact=YES)
 *   refact0   pdgstrf(refact=YES) only; pdgstrf_init() is skipped because
 *             AC->nzval aliases A->nzval and perm_c/colbeg/colend are
 *             already the composed (postordered) ones from call 1
 *
 * Each mode reports per-stage times and the solve residual, so a saving
 * that comes with a wrong answer is visible.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "slu_mt_ddefs.h"

static int_t n, nnz;
static int_t *indptr, *indices;
static double *values0;   /* original   */
static double *values1;   /* perturbed  */

static void read_dump(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(2); }
    int32_t hdr[2];
    if (fread(hdr, sizeof(int32_t), 2, f) != 2) { fprintf(stderr, "short header\n"); exit(2); }
    n = hdr[0]; nnz = hdr[1];
    indptr  = intMalloc(n + 1);
    indices = intMalloc(nnz);
    values0 = doubleMalloc(nnz);
    values1 = doubleMalloc(nnz);
    if (fread(indptr,  sizeof(int_t),  n + 1, f) != (size_t)(n + 1)) exit(2);
    if (fread(indices, sizeof(int_t),  nnz,   f) != (size_t)nnz)     exit(2);
    if (fread(values0, sizeof(double), nnz,   f) != (size_t)nnz)     exit(2);
    fclose(f);

    /* Same pattern, different numbers: stands in for the next Newton tangent. */
    unsigned long long s = 0x9E3779B97F4A7C15ULL;
    for (int_t k = 0; k < nnz; ++k) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        double r = (double)(s >> 11) / (double)(1ULL << 53) - 0.5;
        values1[k] = values0[k] * (1.0 + 2.0e-2 * r);
    }
}

/* Residual ||A x - b||_inf / ||b||_inf for CSC A built from `vals`. */
static double residual(const double *vals, const double *x, const double *b)
{
    double *r = doubleMalloc(n);
    for (int_t i = 0; i < n; ++i) r[i] = -b[i];
    for (int_t j = 0; j < n; ++j)
        for (int_t p = indptr[j]; p < indptr[j + 1]; ++p)
            r[indices[p]] += vals[p] * x[j];
    double rn = 0.0, bn = 0.0;
    for (int_t i = 0; i < n; ++i) {
        if (fabs(r[i]) > rn) rn = fabs(r[i]);
        if (fabs(b[i]) > bn) bn = fabs(b[i]);
    }
    SUPERLU_FREE(r);
    return bn > 0.0 ? rn / bn : rn;
}

/* b = A * ones, so the exact solution is known to be all ones. */
static void make_rhs(const double *vals, double *b)
{
    for (int_t i = 0; i < n; ++i) b[i] = 0.0;
    for (int_t j = 0; j < n; ++j)
        for (int_t p = indptr[j]; p < indptr[j + 1]; ++p)
            b[indices[p]] += vals[p];
}

static double solve_and_check(SuperMatrix *L, SuperMatrix *U, int_t *perm_r,
                              int_t *perm_c, Gstat_t *stat, const double *vals,
                              double *solve_seconds)
{
    double *b = doubleMalloc(n), *rhs = doubleMalloc(n);
    make_rhs(vals, b);
    memcpy(rhs, b, (size_t)n * sizeof(double));

    SuperMatrix B;
    dCreate_Dense_Matrix(&B, n, 1, rhs, n, SLU_DN, SLU_D, SLU_GE);
    int_t info = 0;
    double t0 = SuperLU_timer_();
    dgstrs(NOTRANS, L, U, perm_r, perm_c, &B, stat, &info);
    *solve_seconds = SuperLU_timer_() - t0;

    double res = (info == 0) ? residual(vals, rhs, b) : INFINITY;
    Destroy_SuperMatrix_Store(&B);
    SUPERLU_FREE(rhs);
    SUPERLU_FREE(b);
    return res;
}

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "usage: %s DUMP NPROCS {fresh|refact|refact0} [FILL]\n"
                "  FILL: initial L/U storage estimate applied to\n"
                "        fill_lusup/fill_ucol/fill_lsub (0 = sp_ienv default,\n"
                "        negative = growth factor on nnz(A)).  Applies to the\n"
                "        full factorizations only; under refact=YES the library\n"
                "        reuses the first factorization's allocation.\n", argv[0]);
        return 2;
    }
    const char *dump = argv[1];
    int_t nprocs = atoi(argv[2]);
    const char *mode = argv[3];
    int fill = argc > 4 ? atoi(argv[4]) : 0;
    int do_fresh   = strcmp(mode, "fresh")   == 0;
    int do_refact  = strcmp(mode, "refact")  == 0;
    int do_refact0 = strcmp(mode, "refact0") == 0;
    if (!do_fresh && !do_refact && !do_refact0) {
        fprintf(stderr, "unknown mode %s\n", mode); return 2;
    }

    read_dump(dump);
    printf("n = %d, nnz = %d, nprocs = %d, mode = %s, fill = %d\n",
           (int)n, (int)nnz, (int)nprocs, mode, fill);

    int_t panel_size = sp_ienv(1), relax = sp_ienv(2);
    int_t *perm_c = intMalloc(n), *perm_r = intMalloc(n);

    /* A's nzval is the live buffer we overwrite between factorizations;
       AC->nzval aliases it, which is what makes refact0 possible. */
    double *a = doubleMalloc(nnz);
    int_t *asub = intMalloc(nnz), *xa = intMalloc(n + 1);
    memcpy(a, values0, (size_t)nnz * sizeof(double));
    memcpy(asub, indices, (size_t)nnz * sizeof(int_t));
    memcpy(xa, indptr, (size_t)(n + 1) * sizeof(int_t));

    SuperMatrix A, AC, L, U;
    dCreate_CompCol_Matrix(&A, n, n, nnz, a, asub, xa, SLU_NC, SLU_D, SLU_GE);

    superlumt_options_t options;
    Gstat_t stat;
    StatAlloc(n, nprocs, panel_size, relax, &stat);
    StatInit(n, nprocs, &stat);

    /* ---------------- factorization 1 (always a full one) ---------------- */
    double t0 = SuperLU_timer_();
    get_perm_c(3 /* COLAMD */, &A, perm_c);
    double t_perm1 = SuperLU_timer_() - t0;

    t0 = SuperLU_timer_();
    pdgstrf_init(nprocs, DOFACT, NOTRANS, NO, panel_size, relax, 1.0, NO, 0.0,
                 perm_c, perm_r, NULL, 0, &A, &AC, &options, &stat);
    double t_init1 = SuperLU_timer_() - t0;
    options.fill_lusup = fill;
    options.fill_ucol  = fill;
    options.fill_lsub  = fill;


    int_t info = 0;
    t0 = SuperLU_timer_();
    pdgstrf(&options, &AC, perm_r, &L, &U, &stat, &info);
    double t_fact1 = SuperLU_timer_() - t0;
    if (info != 0) { fprintf(stderr, "factor 1 failed info=%d\n", (int)info); return 1; }

    double t_solve1, res1 = solve_and_check(&L, &U, perm_r, perm_c, &stat,
                                            values0, &t_solve1);
    printf("factor 1 (full):   perm=%8.4f  init=%8.4f  pdgstrf=%9.4f  "
           "solve=%7.4f  residual=%.3e\n",
           t_perm1, t_init1, t_fact1, t_solve1, res1);
    printf("                   nnz(L)=%d nnz(U)=%d\n",
           (int)((SCPformat *)L.Store)->nnz, (int)((NCPformat *)U.Store)->nnz);

    /* Swap in the new values; the pattern is untouched. */
    memcpy(a, values1, (size_t)nnz * sizeof(double));

    /* ---------------- factorization 2 ---------------- */
    double t_perm2 = 0.0, t_init2 = 0.0, t_fact2 = 0.0;

    if (do_fresh) {
        /* Tear the first factorization down completely, then redo everything. */
        Destroy_SuperNode_SCP(&L);
        Destroy_CompCol_NCP(&U);
        pxgstrf_finalize(&options, &AC);

        t0 = SuperLU_timer_();
        get_perm_c(3, &A, perm_c);
        t_perm2 = SuperLU_timer_() - t0;

        t0 = SuperLU_timer_();
        pdgstrf_init(nprocs, DOFACT, NOTRANS, NO, panel_size, relax, 1.0, NO, 0.0,
                     perm_c, perm_r, NULL, 0, &A, &AC, &options, &stat);
        t_init2 = SuperLU_timer_() - t0;
        options.fill_lusup = fill;
        options.fill_ucol  = fill;
        options.fill_lsub  = fill;

    } else if (do_refact) {
        /* perm_c is already the composed one; sp_colorder skips the etree. */
        t0 = SuperLU_timer_();
        options.refact = YES;
        pdgstrf_init(nprocs, DOFACT, NOTRANS, YES, panel_size, relax, 1.0, NO, 0.0,
                     perm_c, perm_r, NULL, 0, &A, &AC, &options, &stat);
        t_init2 = SuperLU_timer_() - t0;
    } else {
        /* refact0: nothing to redo at all -- AC already describes the new A. */
        options.refact = YES;
    }

    info = 0;
    t0 = SuperLU_timer_();
    pdgstrf(&options, &AC, perm_r, &L, &U, &stat, &info);
    t_fact2 = SuperLU_timer_() - t0;
    if (info != 0) { fprintf(stderr, "factor 2 failed info=%d\n", (int)info); return 1; }

    double t_solve2, res2 = solve_and_check(&L, &U, perm_r, perm_c, &stat,
                                            values1, &t_solve2);
    printf("factor 2 (%-7s): perm=%8.4f  init=%8.4f  pdgstrf=%9.4f  "
           "solve=%7.4f  residual=%.3e\n",
           mode, t_perm2, t_init2, t_fact2, t_solve2, res2);
    printf("                   nnz(L)=%d nnz(U)=%d\n",
           (int)((SCPformat *)L.Store)->nnz, (int)((NCPformat *)U.Store)->nnz);

    double total1 = t_perm1 + t_init1 + t_fact1;
    double total2 = t_perm2 + t_init2 + t_fact2;
    printf("TOTAL              factor1=%9.4f  factor2=%9.4f  "
           "saving=%+7.4f s (%+6.2f%%)\n",
           total1, total2, total1 - total2,
           100.0 * (total1 - total2) / total1);
    printf("RESULT %s nprocs=%d fill=%d f1=%.4f f2=%.4f res2=%.3e "
           "p1=%.4f i1=%.4f g1=%.4f p2=%.4f i2=%.4f g2=%.4f\n",
           mode, (int)nprocs, fill, total1, total2, res2,
           t_perm1, t_init1, t_fact1, t_perm2, t_init2, t_fact2);
    return 0;
}
