"""Small frozen-system measurements, independent of PETSc and FE integration."""
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import json
import h5py
import numpy as np
from scipy import sparse, linalg
from scipy.sparse.linalg import spsolve


def main():
    matrices = []
    constraints = []
    result = {}
    for i in range(2):
        with h5py.File(f'kkt_debug_t0_iter{i}.h5') as f:
            g = f[f't_n=0.0/iteration={i}']
            def read(name):
                q = g[name]
                return sparse.csr_matrix((q['data'][:], q['indices'][:], q['indptr'][:]), shape=q.attrs['shape'])
            K, C = read('K'), read('C')
            b = -np.r_[g['r_u'][:], g['r_c'][:]]
            saved = g['correction'][:]
        n = K.shape[0]
        if n > 3000:
            raise ValueError('Dense spectral diagnostics are limited to these small snapshots')
        A = sparse.bmat([[K, C.T], [C, None]], format='csr')
        direct = spsolve(A.tocsc(), b)
        knorm = sparse.linalg.norm(K)
        translations = np.tile(np.eye(3), (n // 3, 1))
        # SVD handles nonsymmetry; do not infer eigenvalues of K from its part.
        _, singular, vh = linalg.svd(K.toarray(), full_matrices=False)
        numerical_null = vh[singular < singular[0] * 1e-12].T
        cc_eig = linalg.eigvalsh((C @ C.T).toarray())
        record = dict(
            shape=list(K.shape), K_frobenius=knorm,
            symmetry_relative=sparse.linalg.norm(K-K.T)/knorm,
            diagonal_range=[float(K.diagonal().min()), float(np.median(K.diagonal())), float(K.diagonal().max())],
            rhs_block_norms=[float(np.linalg.norm(b[:n])), float(np.linalg.norm(b[n:]))],
            saved_residual=float(np.linalg.norm(A@saved-b)/np.linalg.norm(b)),
            direct_residual=float(np.linalg.norm(A@direct-b)/np.linalg.norm(b)),
            saved_direct_relative_difference=float(np.linalg.norm(saved-direct)/np.linalg.norm(direct)),
            translation_residuals=(np.linalg.norm(K@translations, axis=0)/(knorm*np.linalg.norm(translations,axis=0))).tolist(),
            smallest_singular_values=singular[-10:].tolist(), largest_singular_value=float(singular[0]),
            nullity_at_relative_1e12=numerical_null.shape[1],
            constraint_action_on_null_singular_values=linalg.svdvals(C@numerical_null).tolist(),
            CCt_eigenvalue_range=[float(cc_eig[0]), float(cc_eig[-1])],
        )
        result[str(i)] = record
        matrices.append(K)
        constraints.append(C)
    result['relative_K_change'] = sparse.linalg.norm(matrices[1]-matrices[0])/sparse.linalg.norm(matrices[0])
    result['C_difference'] = sparse.linalg.norm(constraints[1]-constraints[0])
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
