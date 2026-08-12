"""PostgreSQL infrastructure package with lazy optional dependencies."""

__all__ = ["PGMeshBackend"]


def __getattr__(name):
    if name == "PGMeshBackend":
        from .client import PGMeshBackend

        return PGMeshBackend
    raise AttributeError(name)
