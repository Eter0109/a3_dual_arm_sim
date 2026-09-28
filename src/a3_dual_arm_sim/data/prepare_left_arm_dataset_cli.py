"""Make an independent left-only training dataset without changing source data."""

import argparse
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from a3_dual_arm_sim.data.audit import audit_training_dataset
from a3_dual_arm_sim.paths import project_root

FIELDS = ("observation.state", "action")


def convert(source, destination):
    audit_training_dataset(source, repo_id="local/a3")
    info = json.loads((source / "meta/info.json").read_text())
    if any(info["features"][key]["shape"] != [16] for key in FIELDS):
        raise ValueError("Expected a dual-arm source dataset")
    summary = json.loads((source / "collection_summary.json").read_text())
    if summary["task"] != "cookie_transfer":
        raise ValueError("Left-only conversion is restricted to cookie_transfer")
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source / "meta", destination / "meta")
    for name in ("a3_episode_metadata.jsonl", "attempts.jsonl"):
        if (source / name).exists():
            shutil.copy2(source / name, destination / name)
    for name in ("images", "videos"):
        if (source / name).exists():
            (destination / name).symlink_to((source / name).resolve(), target_is_directory=True)
    for path in sorted((source / "data").rglob("*.parquet")):
        table = pq.read_table(path)
        for key in FIELDS:
            values = [row[:8] for row in table[key].to_pylist()]
            field = table.schema.field(key)
            dtype = (
                pa.list_(field.type.value_type, 8)
                if pa.types.is_fixed_size_list(field.type)
                else field.type
            )
            table = table.set_column(
                table.schema.get_field_index(key), key, pa.array(values, type=dtype)
            )
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table.replace_schema_metadata(None), target)
    for key in FIELDS:
        feature = info["features"][key]
        feature["shape"] = [8]
        names = feature.get("names")
        if isinstance(names, list) and len(names) == 16:
            feature["names"] = names[:8]
    (destination / "meta/info.json").write_text(json.dumps(info, indent=2))
    stats = json.loads((destination / "meta/stats.json").read_text())
    for key in FIELDS:
        for stat, value in stats[key].items():
            if isinstance(value, list) and len(value) == 16:
                stats[key][stat] = value[:8]
    (destination / "meta/stats.json").write_text(json.dumps(stats, indent=2))
    for path in (destination / "meta/episodes").rglob("*.parquet"):
        table = pq.read_table(path)
        for key in FIELDS:
            for name in table.column_names:
                if name.startswith(f"stats/{key}/") and name.rsplit("/", 1)[-1] != "count":
                    field = table.schema.field(name)
                    values = [v[:8] if v is not None else None for v in table[name].to_pylist()]
                    dtype = (
                        pa.list_(field.type.value_type, 8)
                        if pa.types.is_fixed_size_list(field.type)
                        else field.type
                    )
                    table = table.set_column(
                        table.schema.get_field_index(name), name, pa.array(values, type=dtype)
                    )
        pq.write_table(table, path)
    summary.update(
        training_arm_mode="left",
        source_dataset=str(source.resolve()),
        training_joint_order="L_q1..L_q7,L_gripper",
        environment_action_dim=16,
    )
    (destination / "collection_summary.json").write_text(json.dumps(summary, indent=2))
    return audit_training_dataset(destination, repo_id="local/a3-left")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(convert(project_root() / args.source, project_root() / args.output), indent=2))


if __name__ == "__main__":
    main()
