"""Build a self-contained AirSketch distributable zip.

Includes the full project (source, docs, tests, training), all runtime models,
and the one-click onedir exe (dist/AirSketch). Excludes dev/cruft and the
redundant raw HuggingFace caches (the loaders use the flat model dirs).

Output: <parent of project>/AirSketch_dist.zip   (extracts into AirSketch/)
Run:    python build_dist.py
"""
import os
import sys
import time
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))            # the AirSketch dir
OUT = os.path.join(os.path.dirname(ROOT), "AirSketch_dist.zip")

# Whole directories (matched on any path segment) to skip.
EXCLUDE_DIR_NAMES = {".venv", ".git", ".pytest_cache", "build", "data",
                     "outputs", "__pycache__", ".mypy_cache", ".idea"}


def excluded(rel: str) -> bool:
    rel = rel.replace("\\", "/")
    parts = rel.split("/")
    if any(p in EXCLUDE_DIR_NAMES for p in parts):
        return True
    if rel.endswith(".pyc") or rel.endswith(".log"):
        return True
    # redundant raw HF caches under models/ (flat dirs are what the app loads)
    if len(parts) >= 2 and parts[0] == "models" and parts[1].startswith("models--"):
        return True
    if rel == "models/teacher_voice.json":   # personal enrolled voice — don't ship
        return True
    if os.path.basename(rel) in {"icon32_zoom.png", "exe_check.log", "he_preview.png",
                                 "exe_app.log", "exe_check2.log"}:
        return True
    if rel == os.path.basename(OUT):
        return True
    return False


def main() -> int:
    files = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        rel_dir = os.path.relpath(dirpath, ROOT)
        rel_dir = "" if rel_dir == "." else rel_dir
        # prune excluded dirs in place (faster + avoids descending into .venv etc.)
        dirnames[:] = [d for d in dirnames
                       if not excluded(os.path.join(rel_dir, d) if rel_dir else d)]
        for f in filenames:
            rel = os.path.join(rel_dir, f) if rel_dir else f
            if not excluded(rel):
                files.append(rel)

    total = sum(os.path.getsize(os.path.join(ROOT, r)) for r in files)
    print(f"Including {len(files)} files, {total/1e9:.2f} GB uncompressed")
    print(f"Writing {OUT} ...", flush=True)

    t0 = time.time()
    # ZIP_STORED: model weights are already compressed, so storing is far faster
    # and barely larger. ZIP64 for >4 GB.
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
        for i, rel in enumerate(files, 1):
            z.write(os.path.join(ROOT, rel), arcname="AirSketch/" + rel.replace("\\", "/"))
            if i % 200 == 0:
                print(f"  {i}/{len(files)} ...", flush=True)

    size = os.path.getsize(OUT)
    print(f"Done in {time.time()-t0:.0f}s -> {OUT}  ({size/1e9:.2f} GB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
