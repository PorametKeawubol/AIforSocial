"""Download the current MITRE ATT&CK Enterprise STIX 2.1 bundle (Task 1)."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from urllib.request import Request, urlopen


PROJECT_DIR = Path(__file__).resolve().parent
DATASET_URL = (
    "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/"
    "master/enterprise-attack/enterprise-attack.json"
)


def project_path(value: str | Path) -> Path:
    """Resolve relative CLI paths beside these scripts, independent of the shell cwd."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_DIR / path


def validate_bundle(data: object) -> dict:
    """Reject HTML/error responses and malformed STIX bundles before using them."""
    if not isinstance(data, dict) or data.get("type") != "bundle":
        raise ValueError("The dataset must be a STIX JSON object with type 'bundle'.")
    objects = data.get("objects")
    if not isinstance(objects, list) or not objects:
        raise ValueError("The STIX bundle must contain a nonempty 'objects' list.")
    for index, obj in enumerate(objects):
        if not isinstance(obj, dict):
            raise ValueError(f"Object {index} is not a JSON object.")
        if not isinstance(obj.get("type"), str) or not isinstance(obj.get("id"), str):
            raise ValueError(f"Object {index} must have string 'type' and 'id' fields.")
    return data


def read_bundle(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    return validate_bundle(json.loads(raw)), hashlib.sha256(raw).hexdigest()


def write_json_atomic(path: Path, data: object) -> None:
    """A failed write cannot leave a partially written JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as output:
            temporary_path = Path(output.name)
            json.dump(data, output, ensure_ascii=False, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def download_dataset(output_path: Path, *, refresh: bool = False) -> dict:
    if output_path.exists() and not refresh:
        bundle, digest = read_bundle(output_path)
        print(f"Keeping existing dataset: {output_path}")
        print(f"Total objects: {len(bundle['objects']):,}")
        print(f"SHA256: {digest}")
        print("Use --refresh to fetch the current upstream version.")
        return {"sha256": digest, "object_count": len(bundle["objects"])}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        request = Request(DATASET_URL, headers={"User-Agent": "graph-rag-lab/1.0"})
        print(f"Downloading: {DATASET_URL}")
        with urlopen(request, timeout=120) as response:
            last_modified = response.headers.get("Last-Modified")
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=output_path.parent,
                prefix=f".{output_path.name}.", suffix=".tmp", delete=False,
            ) as output:
                temporary_path = Path(output.name)
                while block := response.read(1024 * 1024):
                    output.write(block)
                output.flush()
                os.fsync(output.fileno())

        bundle, digest = read_bundle(temporary_path)
        metadata = {
            "source_url": DATASET_URL,
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
            "http_last_modified": last_modified,
            "sha256": digest,
            "bytes": temporary_path.stat().st_size,
            "bundle_id": bundle.get("id"),
            "object_count": len(bundle["objects"]),
        }
        os.replace(temporary_path, output_path)
        metadata_path = output_path.with_suffix(".metadata.json")
        write_json_atomic(metadata_path, metadata)
        print(f"Saved dataset: {output_path}")
        print(f"Saved download metadata: {metadata_path}")
        print(f"Total objects: {metadata['object_count']:,}")
        print(f"SHA256: {digest}")
        return metadata
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="enterprise-attack.json")
    parser.add_argument("--refresh", action="store_true", help="Replace an existing dataset.")
    args = parser.parse_args()
    try:
        download_dataset(project_path(args.output), refresh=args.refresh)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Download failed: {exc}\n")


if __name__ == "__main__":
    main()
