/*
 * Standalone reproducer: sporadic wrong solutions from multithreaded
 * SuperLU_MT (pdgssv) on a small unsymmetric saddle-point matrix.
 *
 * Reads a Matrix Market file from stdin, builds b = A * x_exact for a fixed
 * pseudo-random x_exact, then repeatedly calls pdgssv() with NPROCS threads
 * and checks the solution.  With one thread every solve is accurate; with
 * four or more threads a fraction of the solves returns x with a large
 * residual while pdgssv reports info = 0.
 *
 * Build (against a CMake build of SuperLU_MT in BUILD, source in SRC):
 *   cc -O2 -o superlu_mt_race_repro superlu_mt_race_repro.c \
 *      -I SRC/SRC -D__PTHREAD \
 *      BUILD/SRC/libsuperlu_mt_PTHREAD.a BUILD/CBLAS/libblas.a -lpthread -lm
 * Run:
 *   ./superlu_mt_race_repro NPROCS REPS [FILL] < matrix.mtx
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "slu_mt_ddefs.h"

int main(int argc, char *argv[])
{
    if (argc < 3) {
        fprintf(stderr, "usage: %s NPROCS REPS [FILL] [TOL] < matrix.mtx\n"
                "  FILL: initial L/U storage estimate passed through\n"
                "        superlumt_options_t.fill_lusup/fill_ucol/fill_lsub\n"
                "        (0 = library default from sp_ienv(6..8); negative =\n"
                "        growth factor on nnz(A); positive = number of entries)\n", argv[0]);
        return 2;
    }
    int_t nprocs = atoi(argv[1]);
    int reps = atoi(argv[2]);
    int_t fill = argc > 3 ? atoi(argv[3]) : 0;
    double tol = argc > 4 ? atof(argv[4]) : 1.0e-8;

    int_t m, n, nnz, *asub, *xa;
    double *a;
    dreadMM(&m, &n, &nnz, &a, &asub, &xa);   /* reads stdin */
    if (m != n) { fprintf(stderr, "matrix must be square\n"); return 2; }

    /* Fixed pseudo-random exact solution and b = A * x_exact. */
    double *xexact = doubleMalloc(n), *b = doubleMalloc(n);
    unsigned long long s = 0x9E3779B97F4A7C15ULL;
    for (int_t j = 0; j < n; ++j) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        xexact[j] = (double)(s >> 11) / (double)(1ULL << 53) - 0.5;
    }
    for (int_t i = 0; i < n; ++i) b[i] = 0.0;
    double bmax = 0.0;
    for (int_t j = 0; j < n; ++j)
        for (int_t p = xa[j]; p < xa[j + 1]; ++p) b[asub[p]] += a[p] * xexact[j];
    for (int_t i = 0; i < n; ++i) if (fabs(b[i]) > bmax) bmax = fabs(b[i]);

    printf("n = %d, nnz = %d, nprocs = %d, reps = %d, fill = %d, tol = %g\n",
           (int)n, (int)nnz, (int)nprocs, reps, (int)fill, tol);

    int failures = 0, info_failures = 0;
    double worst = 0.0;
    for (int r = 0; r < reps; ++r) {
        SuperMatrix A, L, U, B;
        int_t info;
        /* pdgssv() keeps pointers into A; give it a fresh copy each time. */
        double *ac = doubleMalloc(nnz); int_t *asubc = intMalloc(nnz), *xac = intMalloc(n + 1);
        memcpy(ac, a, nnz * sizeof(double));
        memcpy(asubc, asub, nnz * sizeof(int_t));
        memcpy(xac, xa, (n + 1) * sizeof(int_t));
        dCreate_CompCol_Matrix(&A, n, n, nnz, ac, asubc, xac, SLU_NC, SLU_D, SLU_GE);

        double *rhs = doubleMalloc(n);
        memcpy(rhs, b, n * sizeof(double));
        dCreate_Dense_Matrix(&B, n, 1, rhs, n, SLU_DN, SLU_D, SLU_GE);

        int_t *perm_c = intMalloc(n), *perm_r = intMalloc(n);
        get_perm_c(3, &A, perm_c);           /* 3 = COLAMD */

        /* Same sequence as pdgssv(), with the L/U storage estimates exposed. */
        SuperMatrix AC;
        superlumt_options_t options;
        Gstat_t stat;
        int_t panel_size = sp_ienv(1), relax = sp_ienv(2);
        StatAlloc(n, nprocs, panel_size, relax, &stat);
        StatInit(n, nprocs, &stat);
        pdgstrf_init(nprocs, DOFACT, NOTRANS, NO, panel_size, relax, 1.0, NO, 0.0,
                     perm_c, perm_r, NULL, 0, &A, &AC, &options, &stat);
        if (fill != 0) {
            options.fill_lusup = fill;
            options.fill_ucol = fill;
            options.fill_lsub = fill;
        }
        pdgstrf(&options, &AC, perm_r, &L, &U, &stat, &info);
        if (info == 0 && r == 0) {
            superlu_memusage_t mem;
            superlu_dQuerySpace(nprocs, &L, &U, panel_size, &mem);
            printf("nnz(L) = %d, nnz(U) = %d, L\\U storage %.2f MB, expansions during factorization = %d\n",
                   (int)((SCPformat *)L.Store)->nnz, (int)((NCPformat *)U.Store)->nnz,
                   mem.for_lu / 1.0e6, (int)mem.expansions);
        }
        if (info == 0) dgstrs(NOTRANS, &L, &U, perm_r, perm_c, &B, &stat, &info);
        pxgstrf_finalize(&options, &AC);
        StatFree(&stat);

        if (info != 0) {
            ++info_failures;
            printf("rep %4d: pdgssv info = %d\n", r, (int)info);
        } else {
            /* residual r = A x - b, relative to max|b|, and max|x - x_exact| */
            double *x = rhs, rmax = 0.0, emax = 0.0;
            double *res = doubleMalloc(n);
            for (int_t i = 0; i < n; ++i) res[i] = -b[i];
            for (int_t j = 0; j < n; ++j)
                for (int_t p = xa[j]; p < xa[j + 1]; ++p) res[asub[p]] += a[p] * x[j];
            for (int_t i = 0; i < n; ++i) {
                if (fabs(res[i]) > rmax) rmax = fabs(res[i]);
                if (fabs(x[i] - xexact[i]) > emax) emax = fabs(x[i] - xexact[i]);
            }
            SUPERLU_FREE(res);
            double rel = rmax / bmax;
            if (rel > worst) worst = rel;
            if (rel > tol) {
                ++failures;
                printf("rep %4d: WRONG SOLUTION  |Ax-b|/|b| = %.3e  max|x-x_exact| = %.3e  (info = 0)\n",
                       r, rel, emax);
            }
            Destroy_SuperNode_SCP(&L);
            Destroy_CompCol_NCP(&U);
        }
        SUPERLU_FREE(perm_c); SUPERLU_FREE(perm_r);
        Destroy_SuperMatrix_Store(&B); SUPERLU_FREE(rhs);
        Destroy_CompCol_Matrix(&A);
    }
    printf("summary: nprocs = %d, %d/%d solves wrong (rel. residual > %g), %d with info != 0, worst rel. residual = %.3e\n",
           (int)nprocs, failures, reps, tol, info_failures, worst);
    SUPERLU_FREE(a); SUPERLU_FREE(asub); SUPERLU_FREE(xa); SUPERLU_FREE(xexact); SUPERLU_FREE(b);
    return failures || info_failures ? 1 : 0;
}
