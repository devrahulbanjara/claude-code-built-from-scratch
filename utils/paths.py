from pathlib import Path


def resolve_path(base: str | Path, path: str | Path):
    path = Path(path)
    if path.is_absolute():
        return path.resolve()
    return Path(base).resolve() / path


def is_binary_file(path: str | Path) -> bool:
    try:
        with open(path, "rb") as f:
            # checking for null bytes in first 8 kilobytes
            chunk = f.read(8192)
            # if present then return `True`` it is a binary file
            return "\x00" in chunk
    except OSError:
        return False
