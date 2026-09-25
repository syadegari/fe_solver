// TODO: 
/*
  - Add diagnostics after successful pdgstrf():
     collect the followings:
       - Actual nnz(L), nnz(U)
       - Factorization flop from Gstat_t
       - Various storages from QuerySpace and other routines in pdmemory.c
    These could be useful diagnostics if we wanna tackle the fixed sizes
    of U and L and memory errors they generate for large meshes
*/

#include "slu_mt_ddefs.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>


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


typedef struct {
    int permc_spec;
    double diag_pivot_thresh;
    int panel_size;
    int relax;
} slumt_factor_config;


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
    const slumt_factor_config *config,
    double *setup_seconds,
    double *factor_seconds,
    int *info_out
)
{
    if (!config)
    {
        *info_out = -997;
        return NULL;
    }

    if (nprocs < 1 ||
        config->permc_spec < 0 ||
        config->permc_spec > 3 ||
        !isfinite(config->diag_pivot_thresh) ||
        config->diag_pivot_thresh < 0.0 ||
        config->diag_pivot_thresh > 1.0 ||
        config->panel_size == 0 ||
        config->panel_size < -1 ||
        config->relax < -1)
    {
        *info_out = -997;
        return NULL;
    }

    /*
    * sp_ienv current mapping in 4.0.2:
    *   1 = panel_size
    *   2 = relax
    *   3 = max_supernode_size
    *   4 = min_row_dim_for_2D_blocking
    *   5 = min_col_dim_for_2D_blocking
    *   6 = LUSUP_storage
    *   7 = UCOL/USUB_storage
    *   8 = LSUB_storage
    */

    /* TODO: Describe permc_spec optoins */

    int_t panel_size = config->panel_size == -1 
    ? sp_ienv(1) 
    : (int_t)config->panel_size;

    int_t relax = config->relax == -1 
    ? sp_ienv(2) 
    : (int_t)config->relax;

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



    /*
    * TODO(symbolic-reuse):
    * Keep refact=NO and usepr=NO until the FE backend explicitly owns
    * and validates symbolic-factorization reuse. Reuse is only valid
    * while the sparsity pattern and the required permutation/symbolic
    * state remain compatible.
    */
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
        (int_t)config->permc_spec,
        &h->A,
        h->perm_c
    );

    pdgstrf_init(
        (int_t)nprocs,
        DOFACT,
        NOTRANS,
        NO,             /* refact */
        panel_size,
        relax,
        config->diag_pivot_thresh,
        NO,             /* usepr */
        0.0,            /* drop_tol: unused */
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
    //
    // TODO: Optional improvement
    /*
        Since h is initialized using calloc, use the following to free L and U 
        instead of relying on `factor_called`
        if (h->L.Store) {
            Destroy_SuperNodeSCP(&h->L);
            h->L.Store = NULL;
        }
        if (h->U.Store) {
            Destroy_SuperNodeSCP(&h->U);
            h->U.Store = NULL;
        }
    */
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
