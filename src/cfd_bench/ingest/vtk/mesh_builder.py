"""Build vtkUnstructuredGrid meshes from decoded CAE zones."""

from __future__ import annotations

import numpy as np
import vtk
from vtkmodules.util import numpy_support

from cfd_bench.ingest.decoder import CAE_Decoder, Zone_3D


def _append_cell_array(mesh: vtk.vtkUnstructuredGrid, arr: np.ndarray, name: str) -> vtk.vtkUnstructuredGrid:
    vtk_array = numpy_support.numpy_to_vtk(arr)
    vtk_array.SetName(name)
    mesh.GetCellData().AddArray(vtk_array)
    return mesh


def construct_polyhedral_mesh(zone: Zone_3D) -> vtk.vtkUnstructuredGrid:
    """Build a polyhedral vtkUnstructuredGrid from a decoded zone."""
    points = np.stack(
        [zone.Node_Coordinates[0], zone.Node_Coordinates[1], zone.Node_Coordinates[2]],
        axis=1,
    )
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_support.numpy_to_vtk(points, deep=False))

    vtk_faces = vtk.vtkCellArray()
    vtk_faces.SetNumberOfCells(zone.Face_count)
    for face_idx in range(zone.Face_count):
        face_nodes = zone.FN[face_idx]
        vtk_faces.InsertNextCell(len(face_nodes), list(face_nodes))

    vtk_face_locations = vtk.vtkCellArray()
    vtk_face_locations.SetNumberOfCells(zone.Element_count)
    for elem_idx in range(zone.Element_count):
        element_faces = zone.EF[elem_idx]
        vtk_face_locations.InsertNextCell(len(element_faces), list(element_faces))

    vtk_cell_types = vtk.vtkUnsignedCharArray()
    vtk_cells = vtk.vtkCellArray()
    vtk_cells.SetNumberOfCells(zone.Element_count)
    for elem_idx in range(zone.Element_count):
        vtk_cell_types.InsertNextValue(vtk.VTK_POLYHEDRON)
        element_nodes = zone.EN[elem_idx]
        vtk_cells.InsertNextCell(len(element_nodes), list(element_nodes))

    mesh = vtk.vtkUnstructuredGrid()
    mesh.SetPoints(vtk_points)
    mesh.SetPolyhedralCells(vtk_cell_types, vtk_cells, vtk_face_locations, vtk_faces)
    return mesh


def _attach_cell_scalars(mesh: vtk.vtkUnstructuredGrid, zone: Zone_3D) -> vtk.vtkUnstructuredGrid:
    """Attach cell-centered variables (Variables[3:]) and stable cell_ids."""
    cell_ids = np.arange(zone.Element_count, dtype=np.int64)
    mesh = _append_cell_array(mesh, cell_ids, "cell_ids")

    for var_idx in range(3, len(zone.Variables)):
        name = zone.Variables[var_idx]
        values = zone.Element_Variables[var_idx - 3]
        vtk_array = numpy_support.numpy_to_vtk(values, deep=True, array_type=vtk.VTK_DOUBLE)
        vtk_array.SetName(name)
        mesh.GetCellData().AddArray(vtk_array)
    return mesh


def _attach_point_velocity(mesh: vtk.vtkUnstructuredGrid, zone: Zone_3D) -> vtk.vtkUnstructuredGrid:
    """Interpolate U/V/W to points and add a combined Velocity vector array."""
    cell_to_point = vtk.vtkCellDataToPointData()
    cell_to_point.SetInputData(mesh)
    cell_to_point.Update()
    mesh = cell_to_point.GetOutput()

    # Restore cell-centered arrays for workload scalar queries.
    mesh.GetCellData().Initialize()
    mesh = _attach_cell_scalars(mesh, zone)

    u_array = mesh.GetPointData().GetArray("U")
    v_array = mesh.GetPointData().GetArray("V")
    w_array = mesh.GetPointData().GetArray("W")
    if not (u_array and v_array and w_array):
        return mesh

    velocity = vtk.vtkDoubleArray()
    velocity.SetName("Velocity")
    velocity.SetNumberOfComponents(3)
    velocity.SetNumberOfTuples(mesh.GetNumberOfPoints())
    for i in range(mesh.GetNumberOfPoints()):
        velocity.SetTuple3(i, u_array.GetValue(i), v_array.GetValue(i), w_array.GetValue(i))
    mesh.GetPointData().AddArray(velocity)
    return mesh


def build_meshes_from_dat(dat_path: str) -> tuple[vtk.vtkUnstructuredGrid, vtk.vtkUnstructuredGrid]:
    """Decode one .dat file and return (fluid_mesh, hull_mesh)."""
    if not dat_path.lower().endswith(".dat"):
        raise ValueError(f"not a .dat file: {dat_path}")

    decoder = CAE_Decoder(3)
    decoder.Decode_dat_file(dat_path)
    if len(decoder.Zones) < 2:
        raise RuntimeError(f"expected at least 2 zones in {dat_path}")

    fluid_zone: Zone_3D = decoder.Zones[0]
    hull_zone: Zone_3D = decoder.Zones[1]

    fluid_mesh = construct_polyhedral_mesh(fluid_zone)
    hull_mesh = construct_polyhedral_mesh(hull_zone)
    fluid_mesh = _attach_cell_scalars(fluid_mesh, fluid_zone)
    fluid_mesh = _attach_point_velocity(fluid_mesh, fluid_zone)
    hull_mesh = _attach_cell_scalars(hull_mesh, hull_zone)
    return fluid_mesh, hull_mesh


def write_vtk_mesh(mesh: vtk.vtkUnstructuredGrid, base_path: str) -> str:
    """Write a binary VTK file (legacy format). ``base_path`` may omit the ``.vtk`` suffix."""
    out = base_path if base_path.lower().endswith(".vtk") else f"{base_path}.vtk"
    writer = vtk.vtkUnstructuredGridWriter()
    writer.SetFileName(out)
    writer.SetInputData(mesh)
    writer.SetFileTypeToBinary()
    writer.Write()
    return out
