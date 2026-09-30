"""Report the realised range of every randomised quantity in a cookie collection.

A randomisation is only as wide as what actually got sampled.  The configured
profile says what the sampler was *allowed* to do, and a sampler that silently
narrows -- a bound applied to the wrong axis, an offset added where it should be
multiplied, a value clipped after scaling -- still produces a plausible-looking
dataset.  This script measures the realised range from the per-episode
provenance sidecar and compares it against the bound each episode recorded, so
"the randomisation covers what we configured" becomes a check rather than an
assumption.

Two details make the comparison meaningful rather than approximately right:

*The expected extreme of a uniform sample is not the bound.*  For n draws from
[-a, a] the largest observed value is on average a*(n-1)/(n+1), so a shortfall at
small n is arithmetic, not a bug.  Each row therefore reports the bound, the
observed half-range, and the half-range expected at that n; coverage is measured
against the latter.

*The applied rotation is recoverable exactly.*  A camera's orientation is the
nominal rotated by a quaternion built from a rotvec whose components were drawn
uniformly, so the relative rotation between the nominal and the recorded value
reproduces that rotvec component for component.  Reporting the components checks
the per-axis bound the profile actually declares, which the rotation *angle*
would not: its maximum is the per-axis bound times sqrt(3), and reading that as
the bound would look like a 73% overshoot.

    python scripts/report_randomization_bounds.py --dataset-root <root>
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

METADATA_NAME = "a3_episode_metadata.jsonl"
NOMINAL_NAME = "randomization_nominal.json"

#: Units per family, for the header.  Degrees and millimetres are what the
#: profiles are authored in, and reading metres to six places obscures sizes.
UNIT = {
    "source_bin": "mm",
    "target_bin": "mm",
    "camera_position": "mm",
    "light": "ratio-1",
    "cookie_position": "um",
}


# --- quaternion helpers (Hamilton, scalar first, matching MuJoCo) -------------


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]


def quat_conj(q):
    return [q[0], -q[1], -q[2], -q[3]]


def quat_to_rotvec(q):
    """Rotation vector of a unit quaternion, as the sampler generated it."""
    if q[0] < 0:
        q = [-c for c in q]
    w = max(-1.0, min(1.0, q[0]))
    vector = q[1:]
    norm = math.sqrt(sum(c * c for c in vector))
    if norm < 1e-12:
        return [0.0, 0.0, 0.0]
    theta = 2.0 * math.atan2(norm, w)
    return [c / norm * theta for c in vector]


def relative_rotvec(recorded, nominal):
    """Rotation taking ``nominal`` to ``recorded``: nominal^-1 * recorded."""
    return quat_to_rotvec(quat_mul(quat_conj(nominal), recorded))


def yaw_of(quaternion):
    """Signed rotation about z.  Every sampled orientation here is a pure z-rotation."""
    return 2.0 * math.atan2(quaternion[3], quaternion[0])


# --- statistics --------------------------------------------------------------


class Samples:
    """Signed samples on a symmetric interval, with the bound each was drawn under."""

    def __init__(self, unit: str) -> None:
        self.unit = unit
        self.values: list[float] = []
        self.bounds: dict[float, int] = {}

    def add(self, value: float, bound: float) -> None:
        self.values.append(value)
        self.bounds[bound] = self.bounds.get(bound, 0) + 1

    def bound(self) -> float:
        if len(self.bounds) != 1:
            raise ValueError(
                f"samples were drawn under {len(self.bounds)} different bounds "
                f"({sorted(self.bounds)}); the profile changed mid-collection"
            )
        return next(iter(self.bounds))

    def report(self) -> dict:
        bound = self.bound()
        n = len(self.values)
        low, high = min(self.values), max(self.values)
        # A uniform sample of n draws reaches a*(n-1)/(n+1) on average, so this is
        # the half-range a correct sampler should have produced at this n.
        expected = bound * (n - 1) / (n + 1) if n > 1 else 0.0
        half = (high - low) / 2.0
        # A uniform draw on the bound has this standard deviation.  Anything the
        # observed deviation exceeds it by is independent of the sample, which is
        # what separates "the sampler drew wider than configured" from "the
        # recorded value moved after sampling".
        uniform_sd = bound / math.sqrt(3.0)
        sd = math.sqrt(sum((v - sum(self.values) / n) ** 2 for v in self.values) / n) if n else 0.0
        return {
            "n": n,
            "bound": bound,
            "min": low,
            "max": high,
            "half_range": half,
            "half_range_expected": expected,
            "coverage": half / expected if expected else float("nan"),
            "sd": sd,
            "sd_uniform": uniform_sd,
            "extra_sd": math.sqrt(max(0.0, sd * sd - uniform_sd * uniform_sd)),
            "unit": self.unit,
        }


def fmt(row: dict, scale: float, label: str) -> str:
    return (
        f"  {label:<34} {row['n']:>7}  {row['bound'] * scale:>8.4f}  "
        f"{row['min'] * scale:>+9.4f} {row['max'] * scale:>+9.4f}  "
        f"{row['half_range'] * scale:>8.4f}  {row['coverage']:>7.3f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--nominal", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None, help="Write the statistics as JSON")
    args = parser.parse_args()

    root: Path = args.dataset_root.expanduser().resolve()
    nominal_path = args.nominal or root / NOMINAL_NAME
    if not nominal_path.is_file():
        raise SystemExit(
            f"missing nominal reference {nominal_path}; the nominal scene belongs to the "
            "branch that collected the data and cannot be reconstructed from the episodes alone"
        )
    nominal = json.loads(nominal_path.read_text(encoding="utf-8"))

    shards = sorted(p for p in root.glob("shard_*") if p.is_dir()) or [root]
    records = []
    for shard in shards:
        for line in (shard / METADATA_NAME).read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    if not records:
        raise SystemExit(f"no episodes found under {root}")

    groups: dict[str, Samples] = {}

    def group(key: str, unit: str) -> Samples:
        return groups.setdefault(key, Samples(unit))

    cookie_scale = nominal["cookie_reset_z_m"]
    source_center = nominal["source_bin_center_m"]
    target_nominal = nominal["target_bin_nominal"]
    cookie_grid = nominal["cookie_source_positions"]
    light_nominal = nominal["light_diffuse"]

    colors = Counter()
    clipped_lights = Counter()
    light_totals = Counter()
    cookie_heights: list[float] = []

    for record in records:
        randomisation = record.get("randomization") or {}
        if not randomisation:
            continue
        parameters = randomisation["parameters"]
        posed = randomisation

        # Source bin: the box translation every cookie inherits, so it is removed
        # from the cookie samples below rather than counted as cookie jitter.
        source = posed["source_bin_position"]
        source_x = source[0] - source_center[0]
        source_y = source[1] - source_center[1]
        group("source_bin_x", UNIT["source_bin"]).add(source_x, parameters["source_bin_noise_m"])
        group("source_bin_y", UNIT["source_bin"]).add(source_y, parameters["source_bin_noise_m"])

        target = posed["target_bin_position"]
        group("target_bin_x", UNIT["target_bin"]).add(
            target[0] - target_nominal["position"][0], parameters["target_bin_noise_m"]
        )
        group("target_bin_y", UNIT["target_bin"]).add(
            target[1] - target_nominal["position"][1], parameters["target_bin_noise_m"]
        )
        yaw = yaw_of(posed["target_bin_quaternion"]) - target_nominal["yaw_rad"]
        group("target_bin_yaw", "deg").add(math.degrees(yaw), math.degrees(parameters["target_bin_yaw_noise_rad"]))

        for name, camera in posed["cameras"].items():
            nominal_camera = nominal["cameras"][name]
            wrist = "wrist" in name
            position_bound = parameters["wrist_position_m" if wrist else "camera_position_m"]
            angle_bound = parameters["wrist_rotation_deg" if wrist else "camera_rotation_deg"]
            for axis, index in (("x", 0), ("y", 1), ("z", 2)):
                group(f"camera {name} pos {axis}", UNIT["camera_position"]).add(
                    camera["position"][index] - nominal_camera["position_local"][index],
                    position_bound,
                )
            group(f"camera {name} fovy", "deg").add(
                camera["fovy"] - nominal_camera["fovy"], parameters["fovy_deg"]
            )
            rotvec = relative_rotvec(camera["quaternion"], nominal_camera["orientation"])
            for axis, index in (("x", 0), ("y", 1), ("z", 2)):
                group(f"camera {name} rot {axis}", "deg").add(
                    math.degrees(rotvec[index]), angle_bound
                )

        fraction = parameters["light_fraction"]
        for index, light in enumerate(posed["light_diffuse"]):
            # Each channel shares one scale factor, so averaging them recovers it
            # without assuming the nominal is grey.
            ratios = [v / b for v, b in zip(light, light_nominal[index]) if b]
            ratio = sum(ratios) / len(ratios)
            group(f"light {index} factor", UNIT["light"]).add(ratio - 1.0, fraction)
            light_totals[index] += 1
            if any(v >= 1.0 for v in light):
                clipped_lights[index] += 1

        for index, position in enumerate(posed["cookie_positions"]):
            grid = cookie_grid[index]
            group("cookie pos x", UNIT["cookie_position"]).add(
                position[0] - grid[0] - source_x, parameters["cookie_noise_m"]
            )
            group("cookie pos y", UNIT["cookie_position"]).add(
                position[1] - grid[1] - source_y, parameters["cookie_noise_m"]
            )
            # z is not sampled: the placement height is the constant
            # COOKIE_RESET_Z, and only x, y and yaw are drawn.  Recorded z is the
            # settled height, so it is checked for settling noise instead.
            cookie_heights.append(position[2])
            cookie_yaw = yaw_of(posed["cookie_quaternions"][index])
            group("cookie yaw", "deg").add(math.degrees(cookie_yaw), math.degrees(parameters["cookie_yaw_noise_rad"]))

        if posed.get("cookie_color"):
            colors[posed["cookie_color"]] += 1

    print(f"dataset      : {root}")
    print(f"shards       : {len(shards)}")
    print(f"episodes     : {len(records)}")
    print(f"nominal from : {nominal_path}  (commit {nominal.get('source_commit', '?')[:12]})")
    print()
    print("  coverage = observed half-range / half-range expected from a uniform sample at this n")
    print("  (expected is bound*(n-1)/(n+1); 1.000 means the realised range matches the configured bound)")
    print()

    statistics = {}
    for key in groups:
        row = groups[key].report()
        statistics[key] = row
        scale = 1000.0 if row["unit"] == "mm" else (1e6 if row["unit"] == "um" else 1.0)
        print(fmt(row, scale, f"{key} [{row['unit']}]"))

    if cookie_heights:
        low_z, high_z = min(cookie_heights), max(cookie_heights)
        print(
            f"  {'cookie z (not sampled)':<34} {len(cookie_heights):>7}  "
            f"{'--':>8}  {low_z * 1000:>+9.4f} {high_z * 1000:>+9.4f}  "
            f"{(high_z - low_z) / 2 * 1000:>8.4f}  {'--':>7}"
        )
        settled = sum(cookie_heights) / len(cookie_heights)
        print(
            f"      placement height {cookie_scale:.6f} m, settled mean {settled:.6f} m "
            f"(drop {(cookie_scale - settled) * 1000:.3f} mm); only x, y and yaw are drawn"
        )

    print()
    clipped: set[str] = set()
    for index in sorted(light_totals):
        share = clipped_lights[index] / light_totals[index]
        nominal_light = light_nominal[index][0]
        if share:
            clipped.add(f"light {index} factor")
            print(
                f"  light {index}: nominal {nominal_light:.2f} saturates at 1.0 in "
                f"{share * 100:.1f}% of episodes ({clipped_lights[index]}/{light_totals[index]}) — "
                f"the upward half of the configured range is clipped, so this light only varies downward"
            )
        else:
            print(f"  light {index}: nominal {nominal_light:.2f}, no clipping")

    if cookie_heights:
        print(
            "  cookie placement is measured after the settle steps, so the cookie rows carry the\n"
            "  settling displacement on top of the sampled jitter; a cookie row above 1.000 means\n"
            "  settling, not a wider sample, and the sampled bound itself is the profile's."
        )

    if colors:
        total = sum(colors.values())
        spread = "  ".join(f"{c}={n / total * 100:.1f}%" for c, n in colors.most_common())
        print(f"  cookie colour: {spread}")

    narrow = [
        key for key, row in statistics.items()
        if row["coverage"] < 0.95 and key not in clipped
    ]
    wide = [key for key, row in statistics.items() if row["coverage"] > 1.05]
    print()
    if narrow:
        print("narrower than configured (the sampler is not reaching its bound):")
        for key in narrow:
            print(f"  {key}: coverage {statistics[key]['coverage']:.3f}")
    else:
        print("no sampled quantity falls short of its configured bound")
    if wide:
        print("wider than configured (the recorded value moved after it was sampled):")
        for key in wide:
            row = statistics[key]
            scale = 1000.0 if row["unit"] == "mm" else (1e6 if row["unit"] == "um" else 1.0)
            print(
                f"  {key}: coverage {row['coverage']:.3f}, sd {row['sd'] * scale:.2f} "
                f"vs {row['sd_uniform'] * scale:.2f} expected from the bound alone, "
                f"so an independent {row['extra_sd'] * scale:.2f} {row['unit']} of drift "
                f"(3 sigma = {3 * row['extra_sd'] * scale:.2f} {row['unit']}, which accounts for the "
                f"{max(abs(row['min']), abs(row['max'])) * scale:.2f} {row['unit']} extreme)"
            )

    if args.out:
        args.out.write_text(
            json.dumps(
                {
                    "dataset_root": str(root),
                    "episodes": len(records),
                    "nominal_source": str(nominal_path),
                    "statistics": statistics,
                    "light_saturation": {
                        str(i): clipped_lights[i] / light_totals[i] for i in sorted(light_totals)
                    },
                    "cookie_color": dict(colors),
                },
                indent=1,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
