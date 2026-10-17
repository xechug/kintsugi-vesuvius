from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


class DataManifestError(ValueError):
    pass


@dataclass(frozen=True)
class DataSource:
    name: str
    zarr: str
    mask: str
    ink: str
    split_manifest: str
    splits: dict[str, str]

    def split_for(self, role: str) -> str | None:
        return self.splits.get(role)


@dataclass(frozen=True)
class DataManifest:
    schema_version: int
    sources: tuple[DataSource, ...]
    path: str

    def for_role(self, role: str) -> list[DataSource]:
        return [source for source in self.sources if source.split_for(role) is not None]

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "sources": [asdict(source) for source in self.sources],
        }

    @classmethod
    def load(cls, path: str | Path) -> "DataManifest":
        source_path = Path(path).resolve()
        payload = json.loads(source_path.read_text())
        if payload.get("schema_version") != 1:
            raise DataManifestError("unsupported data manifest schema version")
        rows = payload.get("sources")
        if not isinstance(rows, list) or not rows:
            raise DataManifestError("data manifest must contain sources")

        allowed_roles = {"train", "validation", "test"}
        base = source_path.parent
        parsed = []
        for row in rows:
            required = {"name", "zarr", "mask", "ink", "split_manifest", "splits"}
            missing = required.difference(row)
            if missing:
                raise DataManifestError(
                    f"data source is missing fields: {', '.join(sorted(missing))}"
                )
            splits = row["splits"]
            if not isinstance(splits, dict) or not splits:
                raise DataManifestError(f"source {row['name']} has no split roles")
            unknown = set(splits).difference(allowed_roles)
            if unknown:
                raise DataManifestError(
                    f"source {row['name']} has unknown roles: {', '.join(sorted(unknown))}"
                )

            def resolve(value):
                candidate = Path(str(value))
                return str(candidate if candidate.is_absolute() else (base / candidate).resolve())

            parsed.append(
                DataSource(
                    name=str(row["name"]),
                    zarr=resolve(row["zarr"]),
                    mask=resolve(row["mask"]),
                    ink=resolve(row["ink"]),
                    split_manifest=resolve(row["split_manifest"]),
                    splits={str(role): str(name) for role, name in splits.items()},
                )
            )

        names = [source.name for source in parsed]
        if len(set(names)) != len(names):
            raise DataManifestError("data source names must be unique")
        manifest = cls(1, tuple(parsed), str(source_path))
        if not manifest.for_role("train"):
            raise DataManifestError("data manifest requires at least one train source")
        if not manifest.for_role("validation"):
            raise DataManifestError("data manifest requires at least one validation source")
        return manifest
