"""Safe CelebA/CelebA-HQ attribute and partition alignment."""
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import numpy as np


def _stem(value: str) -> str:
    try:
        return str(int(Path(value).stem))
    except ValueError:
        return Path(value).stem


def read_attributes(path) -> Tuple[List[str], Dict[str, np.ndarray]]:
    with open(path) as f:
        _ = f.readline()
        names = f.readline().split()
        rows = {}
        for line in f:
            fields = line.split()
            if fields:
                rows[_stem(fields[0])] = np.asarray(fields[1:], dtype=np.int8)
    if len(names) != 40 or any(len(v) != 40 for v in rows.values()):
        raise ValueError(f"expected 40 CelebA attributes in {path}")
    return names, rows


def read_partitions(path) -> Dict[str, int]:
    with open(path) as f:
        return {_stem(p[0]): int(p[1]) for line in f if (p := line.split())}


def find_mapping(attr_path) -> Path:
    root = Path(attr_path).parent
    candidates = list(root.glob("*CelebA*HQ*mapping*.txt")) + list(root.glob("*mapping*.txt"))
    if not candidates:
        raise FileNotFoundError(
            "CelebA-HQ image ids do not match the attribute ids. Provide "
            f"CelebA-HQ-to-CelebA-mapping.txt in {root}.")
    return sorted(candidates)[0]


def read_mapping(path) -> Dict[str, str]:
    mapping = {}
    with open(path) as f:
        for line in f:
            fields = line.split()
            if not fields or not fields[0].lstrip('-').isdigit():
                continue
            # Standard file: idx orig_idx orig_file. Prefer orig_file because it
            # uses the exact one-based CelebA filename convention.
            mapping[_stem(fields[0])] = _stem(fields[2] if len(fields) > 2 else fields[1])
    return mapping


def align_attributes(image_ids: Iterable[str], attr_path, partition_path, output_path):
    image_ids = [str(int(x)) if str(x).isdigit() else str(x) for x in image_ids]
    names, attrs_by_id = read_attributes(attr_path)
    partitions_by_id = read_partitions(partition_path)
    if set(image_ids).issubset(attrs_by_id):
        source_ids, case, mapping_path = image_ids, "A", None
    else:
        mapping_path = find_mapping(attr_path)
        mapping = read_mapping(mapping_path)
        missing = [x for x in image_ids if x not in mapping]
        if missing:
            raise ValueError(f"mapping misses {len(missing)} CelebA-HQ ids, first: {missing[:5]}")
        source_ids, case = [mapping[x] for x in image_ids], "B"
    missing = [x for x in source_ids if x not in attrs_by_id or x not in partitions_by_id]
    if missing:
        raise ValueError(f"attribute/partition files miss {len(missing)} aligned ids, first: {missing[:5]}")
    attrs = np.stack([attrs_by_id[x] for x in source_ids]).astype(np.int8)
    partitions = np.asarray([partitions_by_id[x] for x in source_ids], dtype=np.int8)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, attrs=attrs, partitions=partitions,
             image_ids=np.asarray(image_ids), attribute_names=np.asarray(names))
    return {"attribute_names": names, "alignment_case": case,
            "mapping_path": str(mapping_path) if mapping_path else None}
