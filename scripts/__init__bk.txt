import importlib.metadata

try:
    __version__ = importlib.metadata.version("drugseqpy")
except importlib.metadata.PackageNotFoundError:
    # Handle case where package isn't installed (e.g., during development)
    __version__ = "unknown"