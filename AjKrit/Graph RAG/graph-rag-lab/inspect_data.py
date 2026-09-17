"""Inspect the unfiltered MITRE ATT&CK bundle and answer Task 2."""

from __future__ import annotations

import argparse
from collections import Counter

from download_data import project_path, read_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="enterprise-attack.json")
    args = parser.parse_args()
    try:
        bundle, digest = read_bundle(project_path(args.input))
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Cannot read dataset: {exc}\nRun python download_data.py first.\n")

    objects = bundle["objects"]
    counts = Counter(obj["type"] for obj in objects)
    print(f"Dataset: {project_path(args.input)}")
    print(f"SHA256: {digest}")
    print("Counts below include revoked/deprecated objects (the raw bundle).")
    print(f"\nTotal objects: {len(objects):,}")
    print("\nFirst 10 objects:")
    for obj in objects[:10]:
        print(f"  {obj['type']}: {obj.get('name', obj['id'])}")
    print("\nAll object types:")
    for object_type, count in sorted(counts.items()):
        print(f"  {object_type}: {count:,}")

    print("\nTask 2 answers:")
    print(f"  1. Total objects: {len(objects):,}")
    for number, object_type in enumerate(
        ("attack-pattern", "malware", "intrusion-set", "course-of-action", "relationship"),
        start=2,
    ):
        print(f"  {number}. {object_type}: {counts[object_type]:,}")


if __name__ == "__main__":
    main()
