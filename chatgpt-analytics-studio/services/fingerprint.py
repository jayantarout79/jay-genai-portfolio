import hashlib
import os
from typing import Optional


def compute_folder_fingerprint(folder_path: str) -> Optional[str]:
    """
    Build a stable fingerprint from file paths + size + mtime.
    Returns None if folder does not exist.
    """
    if not os.path.isdir(folder_path):
        return None

    entries = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            full_path = os.path.join(root, file)
            try:
                stat = os.stat(full_path)
            except OSError:
                # Skip files that disappear during scan.
                continue
            rel_path = os.path.relpath(full_path, folder_path)
            entries.append((rel_path, stat.st_size, int(stat.st_mtime)))

    # Deterministic ordering before hashing.
    entries.sort()
    hasher = hashlib.sha256()
    for rel_path, size, mtime in entries:
        hasher.update(rel_path.encode("utf-8"))
        hasher.update(str(size).encode("utf-8"))
        hasher.update(str(mtime).encode("utf-8"))
    return hasher.hexdigest()
