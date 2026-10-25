from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
import os
from pathlib import Path, PurePosixPath


PUBLIC_BUCKET = "vesuvius-challenge-open-data"


@dataclass(frozen=True)
class ObjectInfo:
    key: str
    size: int
    etag: str
    last_modified: str


@dataclass(frozen=True)
class DownloadItem:
    source: ObjectInfo
    destination: Path


@dataclass(frozen=True)
class DownloadPlan:
    items: tuple[DownloadItem, ...]
    total_bytes: int
    max_bytes: int
    free_bytes: int


def validate_object_key(key: str) -> str:
    if not key or key.startswith("/") or key.startswith("s3://") or "\\" in key:
        raise ValueError(f"Unsafe public object key: {key!r}")
    path = PurePosixPath(key)
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"Unsafe public object key: {key!r}")
    return key


def unsigned_aws_environment(source=None) -> dict:
    environment = dict(os.environ if source is None else source)
    for key in list(environment):
        if key.startswith("AWS_"):
            environment.pop(key)
    return environment


def aws_list_command(prefix: str, bucket: str = PUBLIC_BUCKET) -> list[str]:
    prefix = validate_object_key(prefix)
    return [
        "aws",
        "s3api",
        "list-objects-v2",
        "--no-sign-request",
        "--bucket",
        bucket,
        "--prefix",
        prefix,
        "--output",
        "json",
    ]


def aws_download_command(key: str, destination: Path, bucket: str = PUBLIC_BUCKET) -> list[str]:
    key = validate_object_key(key)
    return [
        "aws",
        "s3",
        "cp",
        "--no-sign-request",
        f"s3://{bucket}/{key}",
        str(destination),
    ]


def parse_s3_listing(payload: str) -> list[ObjectInfo]:
    parsed = json.loads(payload)
    objects = []
    for item in parsed.get("Contents", []):
        objects.append(
            ObjectInfo(
                key=validate_object_key(item["Key"]),
                size=int(item["Size"]),
                etag=str(item.get("ETag", "")).strip('"'),
                last_modified=str(item.get("LastModified", "")),
            )
        )
    return objects


def inventory_prefix(prefix: str, bucket: str = PUBLIC_BUCKET) -> list[ObjectInfo]:
    completed = subprocess.run(
        aws_list_command(prefix, bucket=bucket),
        env=unsigned_aws_environment(),
        check=True,
        capture_output=True,
        text=True,
    )
    return parse_s3_listing(completed.stdout)


def save_inventory(path: str | Path, prefix: str, objects: list[ObjectInfo]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "bucket": PUBLIC_BUCKET,
        "prefix": prefix,
        "object_count": len(objects),
        "total_bytes": sum(item.size for item in objects),
        "objects": [asdict(item) for item in objects],
    }
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return destination


def load_inventory(path: str | Path) -> list[ObjectInfo]:
    payload = json.loads(Path(path).read_text())
    return [ObjectInfo(**item) for item in payload["objects"]]


def plan_downloads(
    objects: list[ObjectInfo],
    destination_root: str | Path,
    *,
    max_bytes: int,
    free_bytes: int | None = None,
) -> DownloadPlan:
    destination_root = Path(destination_root).resolve()
    total_bytes = sum(int(item.size) for item in objects)
    max_bytes = int(max_bytes)
    if total_bytes > max_bytes:
        raise ValueError(
            f"download size {total_bytes} bytes exceeds configured budget {max_bytes} bytes"
        )
    if free_bytes is None:
        destination_root.mkdir(parents=True, exist_ok=True)
        free_bytes = shutil.disk_usage(destination_root).free
    free_bytes = int(free_bytes)
    if total_bytes > free_bytes:
        raise ValueError(
            f"download size {total_bytes} bytes exceeds free disk {free_bytes} bytes"
        )

    items = []
    for source in objects:
        key = validate_object_key(source.key)
        destination = (destination_root / PurePosixPath(key)).resolve()
        if destination_root not in destination.parents:
            raise ValueError(f"Object escapes destination root: {key}")
        items.append(DownloadItem(source=source, destination=destination))
    return DownloadPlan(tuple(items), total_bytes, max_bytes, free_bytes)


def execute_download_plan(plan: DownloadPlan, bucket: str = PUBLIC_BUCKET) -> None:
    environment = unsigned_aws_environment()
    for item in plan.items:
        item.destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            aws_download_command(item.source.key, item.destination, bucket=bucket),
            env=environment,
            check=True,
        )


def build_parser():
    parser = argparse.ArgumentParser(description="Unsigned, budgeted Vesuvius S3 inventory")
    subparsers = parser.add_subparsers(dest="command", required=True)
    inventory = subparsers.add_parser("inventory")
    inventory.add_argument("--prefix", required=True)
    inventory.add_argument("--output", required=True)

    download = subparsers.add_parser("download")
    download.add_argument("--inventory", required=True)
    download.add_argument("--key", action="append", required=True)
    download.add_argument("--destination", required=True)
    download.add_argument("--max-gib", type=float, required=True)
    return parser


def main():
    args = build_parser().parse_args()
    if args.command == "inventory":
        objects = inventory_prefix(args.prefix)
        path = save_inventory(args.output, args.prefix, objects)
        print(
            f"saved={path} objects={len(objects)} "
            f"bytes={sum(item.size for item in objects)}"
        )
        return

    objects_by_key = {item.key: item for item in load_inventory(args.inventory)}
    missing = [key for key in args.key if key not in objects_by_key]
    if missing:
        raise ValueError(f"keys are not present in the signed-off inventory: {missing}")
    selected = [objects_by_key[key] for key in args.key]
    plan = plan_downloads(
        selected,
        args.destination,
        max_bytes=int(args.max_gib * 1024**3),
    )
    print(f"objects={len(plan.items)} bytes={plan.total_bytes} free={plan.free_bytes}")
    execute_download_plan(plan)


if __name__ == "__main__":
    main()
