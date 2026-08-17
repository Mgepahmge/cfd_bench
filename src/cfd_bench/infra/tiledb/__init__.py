"""TileDB infrastructure package with lazy optional-dependency imports."""

from .config import TileDBConfig

__all__ = ["TileDBConfig", "TileDBRepository", "MeshRuntime"]


def __getattr__(name):
    if name == "TileDBRepository":
        from .repository import TileDBRepository

        return TileDBRepository
    if name == "MeshRuntime":
        from .mesh_runtime import MeshRuntime

        return MeshRuntime
    raise AttributeError(name)
