"""Dump a KKT snapshot to a flat binary file for the standalone C probe."""
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy import sparse


def read_csr(group, name):
    g = group[name]
    shape = tuple(int(v) for v in g.attrs["shape"])
    return sparse.csr_matrix(
        (np.asarray(g["data"]), np.asarray(g["indices"]), np.asarray(g["indptr"])),
        shape=shape, copy=False,
    )


h5path, group_name, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
with h5py.File(h5path, "r") as h5:
    g = h5[group_name]
    K, C = read_csr(g, "K"), read_csr(g, "C")

zero = sparse.csr_matrix((C.shape[0], C.shape[0]), dtype=np.float64)
A = sparse.bmat([[K, C.T], [C, zero]], format="csc")
assert A.has_canonical_format and A.shape[0] == A.shape[1]

n, nnz = A.shape[0], A.nnz
with out.open("wb") as f:
    np.array([n, nnz], dtype=np.int32).tofile(f)
    A.indptr.astype(np.int32).tofile(f)
    A.indices.astype(np.int32).tofile(f)
    A.data.astype(np.float64).tofile(f)
print(f"wrote {out}  n={n} nnz={nnz}")
