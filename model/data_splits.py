"""Create reproducible train/validation/test CSVs before fitting TabDAT preprocessors."""

import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


SPLIT_NAMES = ("train", "validation", "test")


def _file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_data_splits(
    source_path,
    output_dir,
    *,
    validation_ratio=0.1,
    test_ratio=0.2,
    seed=42,
    stratify_col=None,
):
    """Reuse a recorded split or create one without silently overwriting it.

    Ratios refer to the complete, non-missing source table. Any official or
    grouped/temporal split must instead be prepared externally; this helper
    only performs a seeded row-level split.
    """
    if not (0 < validation_ratio < 1 and 0 < test_ratio < 1):
        raise ValueError("validation_ratio and test_ratio must be in (0, 1)")
    if validation_ratio + test_ratio >= 1:
        raise ValueError("validation_ratio + test_ratio must be less than 1")

    source_path = Path(source_path).resolve()
    output_dir = Path(output_dir)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    paths = {name: output_dir / f"{name}.csv" for name in SPLIT_NAMES}
    manifest_path = output_dir / "split_manifest.json"
    manifest = {
        "source_path": str(source_path),
        "source_sha256": _file_sha256(source_path),
        "validation_ratio": validation_ratio,
        "test_ratio": test_ratio,
        "seed": seed,
        "stratify_col": stratify_col,
    }

    existing = [path for path in (*paths.values(), manifest_path) if path.exists()]
    if existing:
        if len(existing) != len(paths) + 1:
            raise FileExistsError(f"Incomplete split in {output_dir}; inspect it before retrying")
        with manifest_path.open(encoding="utf-8") as handle:
            saved = json.load(handle)
        if any(saved.get(key) != value for key, value in manifest.items()):
            raise ValueError(f"Existing split in {output_dir} has different source or settings")
        return paths

    source_frame = pd.read_csv(source_path)
    frame = source_frame.dropna()
    if len(frame) < 3:
        raise ValueError("At least three complete rows are required for a three-way split")
    if stratify_col is not None and stratify_col not in frame.columns:
        raise ValueError(f"Unknown stratification column: {stratify_col}")
    labels = frame[stratify_col] if stratify_col is not None else None
    development, test = train_test_split(
        frame, test_size=test_ratio, random_state=seed, stratify=labels
    )
    development_labels = development[stratify_col] if stratify_col is not None else None
    train, validation = train_test_split(
        development,
        test_size=validation_ratio / (1 - test_ratio),
        random_state=seed,
        stratify=development_labels,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, split in zip(SPLIT_NAMES, (train, validation, test)):
        split.to_csv(paths[name], index=False)
    manifest.update({
        "rows": {
            name: len(split)
            for name, split in zip(SPLIT_NAMES, (train, validation, test))
        },
        "dropped_missing_rows": len(source_frame) - len(frame),
    })
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    return paths
