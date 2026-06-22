from packaging.version import Version


def is_version_folder(name: str) -> bool:
    """Check if a folder name looks like a vX.Y.Z version."""
    try:
        Version(name.lstrip("v"))
        return len(name.lstrip("v").split(".")) == 3
    except Exception:
        return False


def same_minor(v: str, major: int, minor: int) -> bool:
    parsed = Version(v.lstrip("v"))
    return parsed.major == major and parsed.minor == minor


def earlier_minor(v: str, major: int, minor: int) -> bool:
    parsed = Version(v.lstrip("v"))
    return (parsed.major, parsed.minor) < (major, minor)


def latest(versions: list[str]) -> str | None:
    return max(versions, key=lambda v: Version(v.lstrip("v")), default=None)
