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
    /*
     * Blocking parameters, served by this file's sp_ienv() override below.
     * -1 keeps the upstream default.
     */
    int max_supernode;      /* sp_ienv(3) */
    int rowblk;             /* sp_ienv(4) */
    int colblk;             /* sp_ienv(5) */
} slumt_factor_config;


/*
 * sp_ienv() override.
 *
 * SuperLU_MT reads its blocking parameters through sp_ienv(), which takes no
 * context argument, so the only way to influence sp_ienv(3..5) is to provide
 * our own definition.  This is the mechanism upstream itself uses: EXAMPLE/
 * and TESTING/ each ship an sp_ienv.c that replaces the SRC/ one.  Because
 * superlu_mt is linked as a static archive and sp_ienv.o defines no other
 * symbol, the linker resolves every internal reference to this definition and
 * never pulls in the upstream object.
 *
 * The returned defaults replicate the generic (non-MACH) branch of
 * SRC/sp_ienv.c exactly.  If the upstream file or the build's MACH setting
 * ever changes, these must be revisited; slumt_sp_ienv_defaults() exists so a
 * test can assert they still agree.
 *
 * Threading: the overrides are plain globals, read by worker threads during
 * pdgstrf().  slumt_factor() sets them before pdgstrf() and does not touch
 * them while it runs, which is safe for the solver's one-factorization-at-a-
 * time usage.  Concurrent factorizations in a single process would race and
 * are not supported.
 */
static int slumt_env_max_supernode = -1;
static int slumt_env_rowblk = -1;
static int slumt_env_colblk = -1;

int_t
slumt_sp_ienv_defaults(int_t ispec)
{
    switch (ispec) {
        case 1: return (20);
        case 2: return (6);
        case 3: return (200);
        case 4: return (200);
        case 5: return (100);
        case 6: return (-50);
        case 7: return (-50);
        case 8: return (-30);
    }
    return (0);
}

/* Counts calls reaching this override, so a test can prove the library's
   internals resolve sp_ienv() here and not to the upstream SRC/sp_ienv.o. */
static long slumt_sp_ienv_calls = 0;

long
slumt_sp_ienv_call_count(void)
{
    return slumt_sp_ienv_calls;
}

int_t
sp_ienv(int_t ispec)
{
    ++slumt_sp_ienv_calls;
    switch (ispec) {
        case 3:
            if (slumt_env_max_supernode > 0) return (int_t)slumt_env_max_supernode;
            break;
        case 4:
            if (slumt_env_rowblk > 0) return (int_t)slumt_env_rowblk;
            break;
        case 5:
            if (slumt_env_colblk > 0) return (int_t)slumt_env_colblk;
            break;
        default:
            break;
    }
    return slumt_sp_ienv_defaults(ispec);
}


/*
 * SuperLU_MT reports fatal storage exhaustion by printing to stderr and
 * calling exit(-1) from inside pdgstrf(), so no error can be returned to
 * Python and no exception can be raised.  exit() still runs atexit()
 * handlers, which is the one place left to tell the user what to do.
 */
static volatile int slumt_factorization_in_progress = 0;

static void
slumt_report_abort_during_factorization(void)
{
    if (!slumt_factorization_in_progress) {
        return;
    }
    fprintf(
        stderr,
        "\n"
        "[fe-solver] SuperLU_MT terminated the process during factorization.\n"
        "[fe-solver] If the message above says \"Storage for ... exceeded\", the\n"
        "[fe-solver] factors outgrew their allocation. Re-run with a larger\n"
        "[fe-solver] estimate for the array it names:\n"
        "[fe-solver]   L supernodes -> fill_lusup, U columns -> fill_ucol,\n"
        "[fe-solver]   L subscripts -> fill_lsub\n"
        "[fe-solver] e.g. put 'fill_lusup = -100' under [linear_solver.superlu_mt]\n"
        "[fe-solver] in the deck, or under [superlu_mt] in a parameter file named\n"
        "[fe-solver] by the deck's linear_solver.superlu_mt_params_file.\n"
        "[fe-solver] See docs/SUPERLU_MT_FACTOR_STORAGE.md\n"
    );
    fflush(stderr);
}


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
        config->relax < -1 ||
        config->max_supernode == 0 ||
        config->max_supernode < -1 ||
        config->rowblk == 0 ||
        config->rowblk < -1 ||
        config->colblk == 0 ||
        config->colblk < -1)
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

    /* Must precede every sp_ienv() consumer, including the two calls below. */
    slumt_env_max_supernode = config->max_supernode;
    slumt_env_rowblk = config->rowblk;
    slumt_env_colblk = config->colblk;

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
    * refact=NO and usepr=NO are deliberate and settled; this is not a TODO.
    *
    * refact=YES does not skip the symbolic factorization -- SuperLU is
    * left-looking, so the column DFS and supernode detection run inside
    * pdgstrf() on every call and never read the flag. It only skips the
    * etree/column counts and the reuse of the previous factor allocation,
    * measured at 0-2% of a factorization and ~0% on the larger of the two
    * test systems. In exchange the factor storage would be frozen at the
    * first factorization's size (a later Newton iteration needing more
    * exits the process, see docs/SUPERLU_MT_FACTOR_STORAGE.md) and the
    * reused sizes come from a process-global static in the library, so
    * they belong to whichever handle factored last.
    *
    * docs/SUPERLU_MT_INTERNALS.md section 7 and
    * verification/superlu_mt_results/refact_study/ have the measurements.
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

    {
        static int atexit_registered = 0;
        if (!atexit_registered) {
            atexit(slumt_report_abort_during_factorization);
            atexit_registered = 1;
        }
    }
    slumt_factorization_in_progress = 1;

    pdgstrf(
        &h->options,
        &h->AC,
        h->perm_r,
        &h->L,
        &h->U,
        &h->stat,
        &info
    );

    slumt_factorization_in_progress = 0;
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
