#!/usr/bin/env python

import sys
import h5py
import numpy as np
import vtk
import re
from pathlib import Path
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray


def convert_hdf5_to_vtk(h5_file):
    data_folder = Path.cwd()
    vtk_folder = data_folder.joinpath("vtk")
    # Read data from .h5 file
    with h5py.File(h5_file, "r") as f:
        time_grps = sorted(
            [grp for grp in f if grp.startswith("t")],
            key=lambda x: int(re.findall(r"\d+", x)[0]),
        )

        # Make a vtk folder if it doesn't exist
        if not data_folder.joinpath("vtk").exists():
            vtk_folder.mkdir()
        else:
            for file in vtk_folder.glob("*.vtk"):
                file.unlink()

        for i, grp in enumerate(time_grps):
            filaments = f[grp]["filaments"][:]
            nucleators = f[grp]["nucleators"][:]
            centrosomes = f[grp]["centrosomes"][:]
            vtk_file_name = vtk_folder / f"t_{i}.vtk"

            points = vtk.vtkPoints()
            lines = vtk.vtkCellArray()

            # Vectorized operations to add points
            minus_ends = filaments[:, 1:4]
            plus_ends = (
                filaments[:, 1:4] + filaments[:, 4:7] * filaments[:, 7][:, np.newaxis]
            )

            all_points = np.vstack((minus_ends, plus_ends))
            num_points = all_points.shape[0]

            # Add points in bulk
            vtk_points = vtk.vtkPoints()
            vtk_points.SetData(numpy_to_vtk(all_points))

            # Add lines in bulk
            vtk_lines = vtk.vtkCellArray()
            num_filaments = filaments.shape[0]
            connectivity = np.hstack(
                [
                    np.full(
                        (num_filaments, 1), 2, dtype=np.int64
                    ),  # Number of points per line (2)
                    np.arange(num_filaments, dtype=np.int64).reshape(-1, 1),
                    (np.arange(num_filaments, dtype=np.int64) + num_filaments).reshape(
                        -1, 1
                    ),
                ]
            ).flatten()

            vtk_lines.SetCells(num_filaments, numpy_to_vtkIdTypeArray(connectivity))

            # Add length values
            length_values = filaments[
                :, 7
            ]  # Assuming filament length is the scalar value
            vtk_length_values = numpy_to_vtk(length_values, deep=True)
            vtk_length_values.SetName("Length")

            # Add filament ID
            filament_id = filaments[:, 0]
            vtk_filament_id = numpy_to_vtk(filament_id, deep=True)
            vtk_filament_id.SetName("FilamentID")

            # Add polymerization state data
            polymerization_state = filaments[:, 8]
            vtk_polymerization_state = numpy_to_vtk(polymerization_state, deep=True)
            vtk_polymerization_state.SetName("PolyState")

            # Add centrosome number
            centrosome_numbers = filaments[:, 10]
            vtk_centrosome_numbers = numpy_to_vtk(centrosome_numbers, deep=True)
            vtk_centrosome_numbers.SetName("Centrosome")

            # Create a polyline data structure
            poly_data = vtk.vtkPolyData()
            poly_data.SetPoints(vtk_points)
            poly_data.SetLines(vtk_lines)
            poly_data.GetCellData().AddArray(vtk_length_values)
            poly_data.GetCellData().AddArray(vtk_polymerization_state)
            poly_data.GetCellData().AddArray(vtk_centrosome_numbers)
            poly_data.GetCellData().AddArray(vtk_filament_id)

            # Write to a VTK file
            writer = vtk.vtkPolyDataWriter()
            writer.SetFileName(str(vtk_file_name))
            writer.SetInputData(poly_data)
            writer.Write()


if __name__ == "__main__":
    h5_file = sys.argv[1]
    convert_hdf5_to_vtk(h5_file)
