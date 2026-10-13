# PETSc + petsc4py Build Notes

This documents the working installation used for PETSc 3.25.5 with OpenMPI, petsc4py 3.25.5, and HYPRE support.

The environment used here was a Conda environment with Python 3.14.

## 1. Install OpenMPI

```bash
sudo apt install openmpi-bin libopenmpi-dev
```

Confirm that the MPI compiler wrappers are available:

```bash
which mpicc
which mpicxx
which mpiexec
```

Install the Python MPI bindings in the same active environment if absent:

```bash
python -m pip install mpi4py
```

The MPI runtime loaded by mpi4py must match the one used to build PETSc.

## 2. Point PETSc's build at OpenMPI

```bash
export MPICC="$(command -v mpicc)"
export MPICXX="$(command -v mpicxx)"
```

## 3. Build and install PETSc with HYPRE

```bash
PETSC_CONFIGURE_OPTIONS="--download-hypre" \
    python -m pip install --no-cache-dir "petsc==3.25.5"
```

Check the installed PETSc Python package and installation prefix:

```bash
python - <<'PY'
import petsc
from importlib.metadata import version

print("petsc package:", version("petsc"))
print("PETSC_DIR:", petsc.get_config()["PETSC_DIR"])
PY
```

The installation prefix is environment-specific; obtain it from the command above
rather than copying another machine's path.

## 4. Set PETSC_DIR for building petsc4py

Use the PETSc installation just built:

```bash
export PETSC_DIR="$(python -m petsc --prefix)"
export PETSC_ARCH=""
```

Check it:

```bash
echo "$PETSC_DIR"
```

## 5. Install Cython

Because petsc4py is built with `--no-build-isolation`, its build requirements must already exist in the active Python environment.

```bash
python -m pip install "Cython>=3.0"
```

Verify:

```bash
python - <<'PY'
import Cython
print(Cython.__version__)
PY
```

The tested environment reported:

```text
3.3.0
```

## 6. Build petsc4py against this PETSc installation

Install the matching petsc4py version while preventing pip from resolving or replacing PETSc:

```bash
python -m pip install \
    --no-cache-dir \
    --no-deps \
    --no-build-isolation \
    "petsc4py==3.25.5"
```

The important options are:

- `--no-deps`: do not let pip install or replace PETSc as a dependency.
- `--no-build-isolation`: build petsc4py against the packages and PETSc
  installation already present in the active environment.

Keeping PETSc and petsc4py at the same version avoids accidentally loading a different PETSc runtime.

## 7. Verify PETSc, petsc4py, and HYPRE

```bash
python - <<'PY'
from importlib.metadata import version
import petsc
import petsc4py
from petsc4py import PETSc

print("petsc distribution:", version("petsc"))
print("petsc4py distribution:", version("petsc4py"))
print("PETSC_DIR:", petsc.get_config()["PETSC_DIR"])
print("runtime PETSc:", PETSc.Sys.getVersion())
print("hypre:", PETSc.Sys.hasExternalPackage("hypre"))
print("COMM_WORLD size:", PETSc.COMM_WORLD.getSize())
PY
```

Working result:

```text
petsc distribution: 3.25.5
petsc4py distribution: 3.25.5
runtime PETSc: (3, 25, 5)
hypre: True
COMM_WORLD size: 1
```

`COMM_WORLD size: 1` is expected when the script is launched normally with:

```bash
python script.py
```

MPI does not automatically create one process per CPU core.

## 8. Test MPI execution

Run four MPI ranks:

```bash
mpiexec -n 4 python -c '
from petsc4py import PETSc
comm = PETSc.COMM_WORLD
print(f"rank {comm.getRank()} of {comm.getSize()}")
'
```

A successful run prints all four ranks, for example:

```text
rank 3 of 4
rank 0 of 4
rank 1 of 4
rank 2 of 4
```

The order of the output is not significant. MPI ranks execute independently, and output from different ranks is not guaranteed to appear in rank order.

Inside this run:

```python
PETSc.COMM_WORLD.getSize()
```

returns `4`.

## 9. X11 authorization messages

The installation and MPI tests may print messages such as:

```text
Authorization required, but no authorization protocol specified
```

One way to get around this (I checked this and worked on my system) is to set the following env variable before the run or put it in `.bashrc`:

```bash
export HWLOC_COMPONENTS=-gl
```

## Final verified configuration

```text
PETSc:     3.25.5
petsc4py:  3.25.5
Cython:    3.3.0
MPI:       OpenMPI
HYPRE:     enabled
```

This environment is ready for the petsc4py KKT solver experiments.
