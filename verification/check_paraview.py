"""Exercise real ParaView readers and warping; not just XML syntax or HDF selection."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys


def _check_in_paraview(database: Path, xdmf: Path) -> None:
    import h5py
    import numpy as np
    from paraview import servermanager
    from paraview.simple import Delete, WarpByVector, Xdmf3ReaderS, Xdmf3ReaderT
    from vtkmodules.util.numpy_support import vtk_to_numpy

    def leaves(data):
        if data.IsA("vtkCompositeDataSet"):
            iterator = data.NewIterator()
            iterator.SkipEmptyNodesOn()
            iterator.InitTraversal()
            while not iterator.IsDoneWithTraversal():
                block = iterator.GetCurrentDataObject()
                if block is not None and not block.IsA("vtkCompositeDataSet"):
                    yield block
                iterator.GoToNextItem()
        else:
            yield data

    def check_array(fields, name, expected):
        array = fields.GetArray(name)
        if array is None:
            raise AssertionError(f"ParaView dropped array {name!r}")
        values = vtk_to_numpy(array)
        assert array.GetNumberOfTuples() == expected.shape[0], name
        assert array.GetNumberOfComponents() == int(np.prod(expected.shape[1:])), name
        np.testing.assert_array_equal(values.reshape(expected.shape), expected, err_msg=name)

    with h5py.File(database, "r") as source:
        complete = int(source["results"].attrs["n_complete_steps"])
        times = source["results/time"][:complete]
        coordinates = source["mesh/reference_coordinates"][:]
        blocks = sorted(source["mesh/blocks"].keys())
        for factory in (Xdmf3ReaderS, Xdmf3ReaderT):
            reader = factory(FileName=[str(xdmf)])
            reader.UpdatePipelineInformation()
            np.testing.assert_allclose(reader.TimestepValues, times, rtol=0, atol=1e-14)
            warp = WarpByVector(Input=reader)
            warp.Vectors = ["POINTS", "displacement"]
            warp.ScaleFactor = 1.0
            for step, time in enumerate(times):
                reader.UpdatePipeline(float(time))
                meshes = list(leaves(servermanager.Fetch(reader)))
                assert len(meshes) == len(blocks)
                for mesh, block in zip(meshes, blocks):
                    assert mesh.GetNumberOfCells() == len(source[f"mesh/blocks/{block}/connectivity"])
                    np.testing.assert_array_equal(vtk_to_numpy(mesh.GetPoints().GetData()), coordinates)
                    for name, dataset in source["results/nodal"].items():
                        check_array(mesh.GetPointData(), name, dataset[step])
                    fields = source[f"results/blocks/{block}"]
                    for name in ("cauchy_stress", "green_lagrange_strain", "euler_almansi_strain"):
                        values = fields[name][step]
                        check_array(mesh.GetCellData(), name, values)
                        for label in ("11", "22", "33", "12", "23", "13"):
                            assert mesh.GetCellData().GetArray(f"{name}_{label}") is None
                    for name, dataset in fields["state"].items():
                        check_array(mesh.GetCellData(), f"state_{name}", dataset[step])
                warp.UpdatePipeline(float(time))
                warped = list(leaves(servermanager.Fetch(warp)))
                assert len(warped) == len(blocks)
                for mesh in warped:
                    np.testing.assert_allclose(
                        vtk_to_numpy(mesh.GetPoints().GetData()),
                        coordinates + source["results/nodal/displacement"][step], rtol=0, atol=1e-12,
                    )
            print(f"{factory.__name__}: {len(times)} times, {len(blocks)} blocks; all arrays and warped coordinates match HDF5")
            Delete(warp)
            Delete(reader)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("--xdmf", type=Path)
    parser.add_argument("--pvpython", default=shutil.which("pvpython"))
    parser.add_argument("--in-paraview", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    database = args.database.resolve()
    xdmf = args.xdmf.resolve() if args.xdmf else database.with_suffix(".xdmf")
    if args.in_paraview:
        _check_in_paraview(database, xdmf)
        return
    if not args.pvpython:
        parser.error("provide --pvpython /path/to/paraview/bin/pvpython")
    result = subprocess.run(
        [str(args.pvpython), "--no-mpi", "--disable-registry", str(Path(__file__).resolve()),
         str(database), "--xdmf", str(xdmf), "--in-paraview"],
        capture_output=True, text=True, timeout=300,
    )
    print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")
    if result.returncode or "Skipping unrecognized array type" in result.stderr:
        raise RuntimeError("ParaView reader/warp verification failed")


if __name__ == "__main__":
    main()
