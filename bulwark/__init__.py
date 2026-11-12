"""Bulwark: base video model for the state of a person on camera (research code)."""

from importlib.metadata import PackageNotFoundError, version

try:
    # The distribution is `inferensys-bulwark`; the import package stays `bulwark`.
    __version__ = version("inferensys-bulwark")
except PackageNotFoundError:  # a source tree on PYTHONPATH that was never installed
    __version__ = "0+unknown"

__all__ = ["__version__"]
