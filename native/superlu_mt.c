/*
  Diagnostics after a successful pdgstrf() -- nnz(L), nnz(U), factorization
  flops, QuerySpace storage -- are captured in slumt_factor() and exposed via
  slumt_factor_diagnostics().
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

    /* Snapshot taken right after a successful pdgstrf(); see
       slumt_factor_diagnostics(). */
    double diag_nnz_L;
    double diag_nnz_U;
    double diag_factor_flops;
    double diag_lu_storage_bytes;
    double diag_total_needed_bytes;
    double diag_expansions;
} slumt_handle;


typedef struct {
    int permc_spec;
    double diag_pivot_thresh;
    int panel_size;
    int relax;
    /*
     * Initial storage estimates for the factors, forwarded to
     * superlumt_options_t.fill_lusup/fill_ucol/fill_lsub (upstream commit
     * a05e0a0, in v4.1.0).  Semantics match sp_ienv(6/7/8): negative is a
     * fill-growth factor on nnz(A), positive is an absolute entry count,
     * 0 keeps the library default (sp_ienv: -50 / -50 / -30).  Undersized
     * storage is fatal: the library exits the whole process when the
     * factors outgrow these arrays mid-factorization.
     */
    int fill_lusup;
    int fill_ucol;
    int fill_lsub;
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

    /*
     * pdgstrf_init() resets fill_lusup/fill_ucol/fill_lsub to 0 ("use
     * sp_ienv"), so any requested storage estimate must be applied after it
     * and before pdgstrf(), which reads them in pdgstrf_MemInit().
     */
    h->options.fill_lusup = (int_t)config->fill_lusup;
    h->options.fill_ucol = (int_t)config->fill_ucol;
    h->options.fill_lsub = (int_t)config->fill_lsub;

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

    /*
     * Snapshot factor diagnostics while L, U and the Gstat are guaranteed
     * alive.  superlu_dQuerySpace() must run exactly once per factorization:
     * it decrements the library's static expansion counter on every call.
     */
    if (info == 0) {
        SCPformat *Lstore = (SCPformat *)h->L.Store;
        NCPformat *Ustore = (NCPformat *)h->U.Store;
        superlu_memusage_t memusage;

        h->diag_nnz_L = (double)Lstore->nnz;
        h->diag_nnz_U = (double)Ustore->nnz;
        /* pdgstrf() accumulates flops per processor; only the pdgssvx()
           driver folds them into ops[FACT], so sum them here the same way. */
        h->diag_factor_flops = 0.0;
        for (int_t i = 0; i < (int_t)nprocs; ++i) {
            h->diag_factor_flops += (double)h->stat.procstat[i].fcops;
        }
        if (superlu_dQuerySpace(
                (int_t)nprocs, &h->L, &h->U, panel_size, &memusage) == 0) {
            h->diag_lu_storage_bytes = (double)memusage.for_lu;
            h->diag_total_needed_bytes = (double)memusage.total_needed;
            h->diag_expansions = (double)memusage.expansions;
        }
    }

    return h;
}


/*
 * Copies the diagnostics captured after the factorization into six doubles:
 *   nnz_L, nnz_U       stored entries in the factors (L includes supernode
 *                      padding; the unit of the fill_lusup/fill_lsub budget)
 *   factor_flops       floating-point operations spent in pdgstrf()
 *   lu_storage_bytes   memory holding L and U
 *   total_needed_bytes lu_storage_bytes plus factorization working storage
 *   expansions         times the library had to grow the factor arrays
 *                      beyond the initial fill_* estimate
 *
 * Returns 0 on success, -1 if the handle has no successful factorization.
 */
int
slumt_factor_diagnostics(
    void *handle,
    double *nnz_L,
    double *nnz_U,
    double *factor_flops,
    double *lu_storage_bytes,
    double *total_needed_bytes,
    double *expansions
)
{
    slumt_handle *h = handle;

    if (!h || !h->factor_called)
    {
        return -1;
    }

    *nnz_L = h->diag_nnz_L;
    *nnz_U = h->diag_nnz_U;
    *factor_flops = h->diag_factor_flops;
    *lu_storage_bytes = h->diag_lu_storage_bytes;
    *total_needed_bytes = h->diag_total_needed_bytes;
    *expansions = h->diag_expansions;

    return 0;
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
