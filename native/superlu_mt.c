// TODO: Harden the current
/*
 - clean partial-allocation failures correctly;
 - make cleanup safe/idempotent;
 - add the 32-bit int_t compile-time check;
 - define consistent error codes;
 - remove incidental printing from normal operation.
 */

#include "slu_mt_ddefs.h"

#include <stdlib.h>
#include <string.h>


typedef struct {
    int_t n;
    int_t nprocs;

    SuperMatrix A;
    SuperMatrix AC;
    SuperMatrix L;
    SuperMatrix U;

    int_t *perm_c;
    int_t *perm_r;

    superlumt_options_t options;
    Gstat_t stat;

    int stat_allocated;
    int initialized;
    int factor_called;
} slumt_handle;


/*
 * Returns an opaque factorization handle.
 *
 * setup_seconds:
 *   get_perm_c() + pdgstrf_init()
 *
 * factor_seconds:
 *   pdgstrf() only
 */
void *
slumt_factor(
    int nprocs,
    int n,
    int nnz,
    const double *data,
    const int_t *indices,
    const int_t *indptr,
    int permc_spec,
    double *setup_seconds,
    double *factor_seconds,
    int *info_out
)
{
    slumt_handle *h = calloc(1, sizeof(*h));

    if (!h) {
        *info_out = -999;
        return NULL;
    }

    h->n = (int_t)n;
    h->nprocs = (int_t)nprocs;

    double *a = doubleMalloc((int_t)nnz);
    int_t *asub = intMalloc((int_t)nnz);
    int_t *xa = intMalloc((int_t)n + 1);

    h->perm_c = intMalloc((int_t)n);
    h->perm_r = intMalloc((int_t)n);

    if (!a || !asub || !xa || !h->perm_c || !h->perm_r) {
        *info_out = -998;
        return h;
    }

    memcpy(a, data, (size_t)nnz * sizeof(double));
    memcpy(asub, indices, (size_t)nnz * sizeof(int_t));
    memcpy(xa, indptr, ((size_t)n + 1) * sizeof(int_t));

    dCreate_CompCol_Matrix(
        &h->A,
        (int_t)n,
        (int_t)n,
        (int_t)nnz,
        a,
        asub,
        xa,
        SLU_NC,
        SLU_D,
        SLU_GE
    );

    int_t panel_size = sp_ienv(1);
    int_t relax = sp_ienv(2);

    StatAlloc(
        (int_t)n,
        (int_t)nprocs,
        panel_size,
        relax,
        &h->stat
    );
    h->stat_allocated = 1;

    StatInit(
        (int_t)n,
        (int_t)nprocs,
        &h->stat
    );

    double t0 = SuperLU_timer_();

    get_perm_c(
        (int_t)permc_spec,
        &h->A,
        h->perm_c
    );

    pdgstrf_init(
        (int_t)nprocs,
        EQUILIBRATE,
        NOTRANS,
        NO,
        panel_size,
        relax,
        1.0,
        NO,
        0.0,
        h->perm_c,
        h->perm_r,
        NULL,
        0,
        &h->A,
        &h->AC,
        &h->options,
        &h->stat
    );

    h->initialized = 1;

    *setup_seconds = SuperLU_timer_() - t0;

    int_t info = 0;

    t0 = SuperLU_timer_();

    pdgstrf(
        &h->options,
        &h->AC,
        h->perm_r,
        &h->L,
        &h->U,
        &h->stat,
        &info
    );

    h->factor_called = 1;

    *factor_seconds = SuperLU_timer_() - t0;
    *info_out = (int)info;

    return h;
}


int
slumt_solve(
    void *handle,
    const double *b,
    double *x,
    int nrhs,
    double *solve_seconds
)
{
    slumt_handle *h = (slumt_handle *)handle;

    if (!h || !h->factor_called)
        return -999;

    size_t count = (size_t)h->n * (size_t)nrhs;

    double *rhs = doubleMalloc((int_t)count);
    if (!rhs)
        return -998;

    memcpy(rhs, b, count * sizeof(double));

    SuperMatrix B;

    dCreate_Dense_Matrix(
        &B,
        h->n,
        (int_t)nrhs,
        rhs,
        h->n,
        SLU_DN,
        SLU_D,
        SLU_GE
    );

    int_t info = 0;

    double t0 = SuperLU_timer_();

    dgstrs(
        NOTRANS,
        &h->L,
        &h->U,
        h->perm_r,
        h->perm_c,
        &B,
        &h->stat,
        &info
    );

    *solve_seconds = SuperLU_timer_() - t0;

    if (info == 0)
        memcpy(x, rhs, count * sizeof(double));

    Destroy_SuperMatrix_Store(&B);
    SUPERLU_FREE(rhs);

    return (int)info;
}


// Use this only after factorization/solve is done and not needed anymore
void
slumt_free(void *handle)
{
    slumt_handle *h = (slumt_handle *)handle;

    if (!h) return;

    if (h->initialized) pxgstrf_finalize(&h->options, &h->AC);

    if (h->factor_called) {
        Destroy_SuperNode_SCP(&h->L);
        Destroy_CompCol_NCP(&h->U);
    }

    if (h->stat_allocated) StatFree(&h->stat);

    if (h->perm_r) SUPERLU_FREE(h->perm_r);

    if (h->perm_c) SUPERLU_FREE(h->perm_c);

    if (h->A.Store) Destroy_CompCol_Matrix(&h->A);

    free(h);
}
