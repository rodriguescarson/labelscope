"""Command line interface."""
from __future__ import annotations

import argparse
import os
import statistics
import sys
from typing import Dict, List

from labelscope import __version__
from labelscope.io import discover_pairs, probe_volume, read_volume
from labelscope.report import write_csv, write_html, write_json


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


# --------------------------------------------------------------------------- #
# scan
# --------------------------------------------------------------------------- #
def cmd_scan(args: argparse.Namespace) -> int:
    from labelscope.quality import audit_label

    pairs = discover_pairs(args.images, args.labels)
    if not pairs:
        _log("no volumes found")
        return 2
    _log(f"scanning {len(pairs)} volumes")

    records: List[Dict] = []
    for n, pair in enumerate(pairs, 1):
        record: Dict = {"name": pair.name, "has_image": pair.image is not None,
                        "has_label": pair.label is not None}
        if pair.label:
            info = probe_volume(pair.label)
            record.update({
                "label_shape": info.shape, "label_dtype": info.dtype,
                "label_compression": info.compression, "label_bytes": info.file_size,
                "label_header_error": info.error,
            })
        if pair.image:
            info = probe_volume(pair.image)
            record.update({
                "image_shape": info.shape, "image_dtype": info.dtype,
                "image_compression": info.compression, "image_bytes": info.file_size,
                "image_header_error": info.error,
            })
        if pair.label and not args.headers_only:
            try:
                record.update(audit_label(read_volume(pair.label), deep=args.deep))
            except Exception as exc:
                record["label_read_error"] = f"{type(exc).__name__}: {exc}"
        records.append(record)
        if n % 25 == 0 or n == len(pairs):
            _log(f"  {n}/{len(pairs)}")

    os.makedirs(args.out, exist_ok=True)
    write_csv(records, os.path.join(args.out, "scan.csv"))
    summary = _summarise_scan(records)
    write_json(summary, os.path.join(args.out, "scan_summary.json"))
    _render_scan_html(records, summary, os.path.join(args.out, "scan.html"))
    _log(f"wrote {args.out}/scan.csv, scan_summary.json, scan.html")
    for line in summary["headline"]:
        print(line)
    return 0


def _summarise_scan(records: List[Dict]) -> Dict:
    total = len(records)
    unpaired = [r["name"] for r in records if not (r["has_image"] and r["has_label"])]
    shapes: Dict = {}
    compressions: Dict = {}
    schemes: Dict = {}
    bytes_by_compression: Dict = {}
    for r in records:
        shape = str(r.get("label_shape"))
        shapes[shape] = shapes.get(shape, 0) + 1
        comp = str(r.get("label_compression"))
        compressions[comp] = compressions.get(comp, 0) + 1
        bytes_by_compression[comp] = bytes_by_compression.get(comp, 0) + (r.get("label_bytes") or 0)
        if "values" in r:
            key = str(r["values"])
            schemes[key] = schemes.get(key, 0) + 1

    surface_classes: Dict = {}
    for r in records:
        if "surface_class" in r:
            key = str(r["surface_class"])
            surface_classes[key] = surface_classes.get(key, 0) + 1
    modal_scheme = max(schemes, key=schemes.get) if schemes else None
    off_scheme = [r["name"] for r in records if "values" in r and str(r["values"]) != modal_scheme]

    headline = [
        f"volumes: {total}",
        f"distinct label shapes: {len(shapes)}  {shapes}",
        f"label compressions: {compressions}",
    ]
    if modal_scheme:
        headline.append(f"modal class scheme: {modal_scheme} ({schemes[modal_scheme]}/{total})")
        headline.append(f"volumes off the modal scheme: {len(off_scheme)}")
    if unpaired:
        headline.append(f"unpaired volumes: {len(unpaired)}")
    if len(compressions) > 1:
        headline.append(
            "mixed compression: " + ", ".join(
                f"{k}={v} files / {bytes_by_compression[k] / 1e9:.2f} GB"
                for k, v in compressions.items()
            )
        )
    if surface_classes:
        headline.append(f"detected surface class: {surface_classes}")
    return {
        "n_volumes": total, "shapes": shapes, "compressions": compressions,
        "detected_surface_classes": surface_classes,
        "bytes_by_compression": bytes_by_compression, "class_schemes": schemes,
        "modal_scheme": modal_scheme, "off_scheme_volumes": off_scheme,
        "unpaired_volumes": unpaired, "headline": headline,
    }


def _render_scan_html(records, summary, path) -> None:
    cards = [("volumes", summary["n_volumes"], ""),
             ("label shapes", len(summary["shapes"]),
              "warn" if len(summary["shapes"]) > 1 else "ok"),
             ("compressions", len(summary["compressions"]),
              "warn" if len(summary["compressions"]) > 1 else "ok"),
             ("class schemes", len(summary["class_schemes"]),
              "warn" if len(summary["class_schemes"]) > 1 else "ok"),
             ("unpaired", len(summary["unpaired_volumes"]),
              "warn" if summary["unpaired_volumes"] else "ok")]
    thick = [r for r in records if r.get("surface_thickness_p95") is not None]
    thick.sort(key=lambda r: -r["surface_thickness_p95"])
    rows = [{"name": r["name"],
             "surface class": r.get("surface_class"),
             "surface fraction": r.get("surface_fraction"),
             "thickness median": r.get("surface_thickness_median"),
             "thickness p95": r.get("surface_thickness_p95"),
             "components": r.get("surface_components"),
             "fragments": r.get("surface_fragment_fraction"),
             "worst planarity": r.get("surface_worst_planarity")}
            for r in thick]
    sections = [
        ("Storage and schema", "One row per distinct combination found in the headers.",
         [{"property": k, "value": str(v)} for k, v in
          (("shapes", summary["shapes"]), ("compressions", summary["compressions"]),
           ("class schemes", summary["class_schemes"]))],
         ["property", "value"]),
        ("Thickest labels",
         "Highest 95th-percentile local thickness first — a fat tail is where a label "
         "has most likely swallowed the gap between two windings.",
         rows, ["name", "surface class", "surface fraction", "thickness median",
                "thickness p95", "components", "fragments", "worst planarity"]),
    ]
    write_html("labelscope — dataset scan",
               "Header inventory and label-only quality metrics.",
               cards, sections, path)


# --------------------------------------------------------------------------- #
# leakage
# --------------------------------------------------------------------------- #
def cmd_leakage(args: argparse.Namespace) -> int:
    from labelscope.geometry import (blocked_kfold, measure_split_leakage,
                                     nnunet_default_split, overlap_graph,
                                     parse_patch_names, simulate_random_kfold, write_splits)

    if args.names_file:
        with open(args.names_file) as handle:
            names = [line.strip() for line in handle if line.strip()]
        names = [n[:-4] if n.endswith(".tif") else n for n in names]
    else:
        names = [p.name for p in discover_pairs(args.images, args.labels)]
    if not names:
        _log("no volumes found")
        return 2

    size = tuple(args.patch_size) if len(args.patch_size) == 3 else tuple(args.patch_size * 3)
    patches, unparsed = parse_patch_names(names, size)
    if not patches:
        _log("no volume names carry z/y/x coordinates — nothing to check.\n"
             "labelscope leakage needs names like s1_z10240_y2560_x2560.")
        return 2
    _log(f"parsed {len(patches)} patch names ({len(unparsed)} without coordinates)")

    graph = overlap_graph(patches, buffer=args.buffer)
    n_pairs = sum(len(v) for v in graph.values()) // 2
    touched = [i for i in range(len(patches)) if graph.get(i)]
    max_overlap = [max(graph[i].values()) for i in touched] or [0.0]

    random_kfold = simulate_random_kfold(len(patches), graph, k=args.k, trials=args.trials)
    nnunet = measure_split_leakage(
        patches, nnunet_default_split([p.name for p in patches], k=args.k), graph)
    splits, split_stats = blocked_kfold(patches, k=args.k, buffer=args.buffer, mode=args.mode)

    os.makedirs(args.out, exist_ok=True)
    write_splits(splits, os.path.join(args.out, "splits_final.json"))
    summary = {
        "n_patches": len(patches), "patch_size": list(size), "buffer": args.buffer,
        "overlapping_pairs": n_pairs,
        "patches_with_overlapping_neighbour": len(touched),
        "patches_with_overlapping_neighbour_pct": 100.0 * len(touched) / len(patches),
        "max_overlap_fraction_median_pct": 100.0 * statistics.median(max_overlap),
        "max_overlap_fraction_max_pct": 100.0 * max(max_overlap),
        "random_kfold": random_kfold,
        "nnunet_default_split": nnunet,
        "blocked_split": split_stats,
    }
    write_json(summary, os.path.join(args.out, "leakage.json"))

    cards = [
        ("patches", len(patches), ""),
        ("overlapping pairs", n_pairs, "warn" if n_pairs else "ok"),
        ("patches touching another",
         f'{summary["patches_with_overlapping_neighbour_pct"]:.1f}%',
         "warn" if len(touched) else "ok"),
        ("nnU-Net default split leaked",
         f'{nnunet["val_patches_contaminated_pct"]:.1f}%', "warn"),
        ("after blocked split",
         f'{split_stats["residual_leaking_val_patches"]} patches', "ok"),
    ]
    prose = (
        f"The split nnU-Net writes when none is supplied — "
        f"<code>KFold({args.k}, shuffle=True, random_state=12345)</code> over the sorted "
        f"case names — puts {nnunet['val_patches_contaminated_pct']:.1f}% of validation "
        f"patches in contact with voxels the model trained on, sharing "
        f"{nnunet['mean_max_shared_volume_pct']:.1f}% of their volume on average and up to "
        f"{nnunet['worst_shared_volume_pct']:.1f}% at worst. Averaged over "
        f"{args.trials} random shuffles the figure is "
        f"{random_kfold['val_patches_contaminated_pct_mean']:.1f}%, so this is a property of "
        "the data, not of one unlucky seed. The blocked split written alongside this report "
        "removes that contact entirely."
    )
    sections = [
        ("Blocked split", prose,
         [{"fold": i, "train": t, "val": v, "dropped to buffer": d}
          for i, (t, v, d) in enumerate(zip(split_stats["train_fold_sizes"],
                                            split_stats["val_fold_sizes"],
                                            split_stats["buffer_dropped_per_fold"]))],
         ["fold", "train", "val", "dropped to buffer"]),
    ]
    write_html("labelscope — split leakage",
               "Spatial overlap between training patches, and a split that removes it.",
               cards, sections, os.path.join(args.out, "leakage.html"))
    _log(f"wrote {args.out}/splits_final.json, leakage.json, leakage.html")
    print(f"{len(patches)} patches, {n_pairs} overlapping pairs "
          f"({summary['patches_with_overlapping_neighbour_pct']:.1f}% of patches)")
    print(f"nnU-Net default split: {nnunet['val_patches_contaminated_pct']:.1f}% of "
          f"validation patches leak (random shuffles: "
          f"{random_kfold['val_patches_contaminated_pct_mean']:.1f}%)")
    print(f"blocked split: val folds {split_stats['val_fold_sizes']}, "
          f"{split_stats['residual_leaking_val_patches']} residual leaks, "
          f"{split_stats['buffer_dropped_pct_mean']:.1f}% of training patches dropped to buffer")
    return 0


# --------------------------------------------------------------------------- #
# align
# --------------------------------------------------------------------------- #
def cmd_align(args: argparse.Namespace) -> int:
    from labelscope.alignment import audit_alignment

    pairs = [p for p in discover_pairs(args.images, args.labels) if p.complete]
    if not pairs:
        _log("no complete image/label pairs found")
        return 2
    if args.limit:
        pairs = pairs[: args.limit]
    _log(f"aligning {len(pairs)} pairs")

    records, cache = [], {}
    for n, pair in enumerate(pairs, 1):
        try:
            image = read_volume(pair.image)
            label = read_volume(pair.label)
            record = {"name": pair.name}
            record.update(audit_alignment(
                image, label,
                surface_class=args.surface_class if args.surface_class >= 0 else None,
                orient_class=args.orient_class if args.orient_class >= 0 else None,
                radius=args.radius, n_samples=args.samples))
            records.append(record)
            if args.overlays:
                cache[pair.name] = (pair.image, pair.label,
                                    record.get("surface_class"), record.get("orient_class"))
        except Exception as exc:
            records.append({"name": pair.name, "error": f"{type(exc).__name__}: {exc}"})
        _log(f"  {n}/{len(pairs)} {pair.name}")

    if args.overlays:
        _render_overlays(records, cache, args.out, args.overlays)

    os.makedirs(args.out, exist_ok=True)
    write_csv(records, os.path.join(args.out, "align.csv"))
    good = [r for r in records if "median_abs_offset" in r]
    summary: Dict = {"n_pairs": len(records), "n_ok": len(good)}
    if good:
        for key in ("mean_signed_offset", "median_abs_offset", "frac_offset_ge_1vx",
                    "frac_offset_ge_2vx", "frac_flat_support", "median_prominence_norm",
                    "hf_energy_norm"):
            values = [r[key] for r in good if key in r]
            summary[key] = {"median": statistics.median(values),
                            "mean": statistics.fmean(values),
                            "min": min(values), "max": max(values)}
        summary["coverage_vs_difficulty"] = _difficulty_bins(good)
    write_json(summary, os.path.join(args.out, "align_summary.json"))
    _render_align_html(good, summary, os.path.join(args.out, "align.html"))
    _log(f"wrote {args.out}/align.csv, align_summary.json, align.html")
    if good:
        print(f"median |offset| = {summary['median_abs_offset']['median']:.2f} vx, "
              f"signed mean = {summary['mean_signed_offset']['median']:+.2f} vx, "
              f"{_pct(summary['frac_offset_ge_1vx']['median'])} of surface ≥1 vx off ridge")
    return 0


def _render_overlays(records, cache, out_dir, count) -> None:
    """Draw the worst-aligned patches, so a reader can see the drift, not just
    read a number for it."""
    from labelscope.visualize import render_drift_map, render_overlay

    usable = [r for r in records if "median_abs_offset" in r and r["name"] in cache]
    usable.sort(key=lambda r: -r["median_abs_offset"])
    directory = os.path.join(out_dir, "overlays")
    for record in usable[:count]:
        image_path, label_path, surface, orient = cache[record["name"]]
        try:
            import numpy as np
            image = read_volume(image_path)
            label = read_volume(label_path)
            orient_field = (label == orient).astype(np.float32) if orient is not None else None
            base = os.path.join(directory, record["name"])
            record["overlay_png"] = render_overlay(
                image, label, base + "_overlay.png", surface_class=surface)
            record["drift_png"] = render_drift_map(
                image, label, base + "_drift.png",
                surface_class=surface, orient_field=orient_field)
        except Exception as exc:
            _log(f"  overlay failed for {record['name']}: {type(exc).__name__}: {exc}")
    _log(f"wrote overlays to {directory}")


def _difficulty_bins(records: List[Dict], bins: int = 4) -> List[Dict]:
    """Are the labels worse where the scan is harder?  Bin by local contrast."""
    ranked = sorted(records, key=lambda r: r.get("hf_energy_norm", 0.0))
    if len(ranked) < bins:
        return []
    out = []
    size = len(ranked) // bins
    for b in range(bins):
        chunk = ranked[b * size: (b + 1) * size if b < bins - 1 else len(ranked)]
        out.append({
            "bin": b,
            "label": ["hardest (haziest)", "hard", "easier", "easiest (sharpest)"][b]
            if bins == 4 else str(b),
            "n": len(chunk),
            "hf_energy_norm": statistics.median(r["hf_energy_norm"] for r in chunk),
            "median_abs_offset": statistics.median(r["median_abs_offset"] for r in chunk),
            "frac_offset_ge_1vx": statistics.median(r["frac_offset_ge_1vx"] for r in chunk),
            "median_prominence_norm": statistics.median(r["median_prominence_norm"] for r in chunk),
            "surface_voxels": statistics.median(r["surface_voxels"] for r in chunk),
        })
    return out


def _render_align_html(records, summary, path) -> None:
    if not records:
        write_html("labelscope — alignment", "No usable pairs.", [], [], path)
        return
    cards = [
        ("pairs", summary["n_ok"], ""),
        ("median |offset|", f'{summary["median_abs_offset"]["median"]:.2f} vx', ""),
        ("signed mean", f'{summary["mean_signed_offset"]["median"]:+.2f} vx',
         "warn" if abs(summary["mean_signed_offset"]["median"]) > 0.2 else "ok"),
        ("surface ≥1 vx off", _pct(summary["frac_offset_ge_1vx"]["median"]), ""),
        ("flat support", _pct(summary["frac_flat_support"]["median"]),
         "warn" if summary["frac_flat_support"]["median"] > 0.05 else "ok"),
    ]
    worst = sorted(records, key=lambda r: -r["median_abs_offset"])
    rows = [{"name": r["name"], "median |offset|": r["median_abs_offset"],
             "signed mean": r["mean_signed_offset"],
             "≥1 vx": r["frac_offset_ge_1vx"], "flat support": r["frac_flat_support"],
             "hf energy": r["hf_energy_norm"]} for r in worst]
    sections = [
        ("Labels worst aligned with the CT ridge",
         "Offset is measured along the surface normal, in voxels, positive outward. "
         "A non-zero <em>signed</em> mean is systematic bias rather than annotator noise.",
         rows, ["name", "median |offset|", "signed mean", "≥1 vx", "flat support", "hf energy"]),
        ("Does label quality track scan difficulty?",
         "Patches binned by high-frequency energy — the label-free proxy for the "
         "compressed, hazy regions the Open Problems post describes.",
         summary.get("coverage_vs_difficulty", []),
         ["label", "n", "hf_energy_norm", "median_abs_offset", "frac_offset_ge_1vx",
          "median_prominence_norm", "surface_voxels"]),
    ]
    write_html("labelscope — CT/label alignment",
               "How far the labels sit from the ridge the scan actually shows.",
               cards, sections, path)


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="labelscope",
        description="Diagnostics for Vesuvius Challenge surface-label datasets.")
    parser.add_argument("--version", action="version", version=f"labelscope {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="inventory volumes and score label-only quality")
    scan.add_argument("--labels", required=True)
    scan.add_argument("--images")
    scan.add_argument("--out", default="labelscope_out")
    scan.add_argument("--headers-only", action="store_true",
                      help="probe headers only; never decode voxels")
    scan.add_argument("--deep", action="store_true",
                      help="also skeletonise each label (slow) to count branch points")
    scan.set_defaults(func=cmd_scan)

    leak = sub.add_parser("leakage",
                          help="find patches that share voxels and emit a split that does not")
    leak.add_argument("--labels", help="labelsTr directory")
    leak.add_argument("--images")
    leak.add_argument("--names-file",
                      help="newline-separated patch names, when the volumes are not "
                           "on this machine — the check only needs the names")
    leak.add_argument("--patch-size", type=int, nargs="+", default=[300],
                      help="one value for a cube, or three as z y x")
    leak.add_argument("--k", type=int, default=5)
    leak.add_argument("--buffer", type=int, default=0,
                      help="treat patches within this many voxels as neighbours too")
    leak.add_argument("--mode", choices=["block", "component"], default="block")
    leak.add_argument("--trials", type=int, default=200)
    leak.add_argument("--out", default="labelscope_out")
    leak.set_defaults(func=cmd_leakage)

    align = sub.add_parser("align", help="measure label offset from the CT's own ridge")
    align.add_argument("--images", required=True)
    align.add_argument("--labels", required=True)
    align.add_argument("--out", default="labelscope_out")
    align.add_argument("--surface-class", type=int, default=-1,
                       help="thin sheet class; -1 (default) detects it")
    align.add_argument("--orient-class", type=int, default=-1,
                       help="bulky region class defining 'outward', so the offset "
                            "sign is meaningful; -1 (default) detects it")
    align.add_argument("--radius", type=int, default=6)
    align.add_argument("--samples", type=int, default=20000)
    align.add_argument("--limit", type=int, default=0)
    align.add_argument("--overlays", type=int, default=0, metavar="N",
                       help="render CT/label overlay and drift-map PNGs for the N "
                            "worst-aligned patches")
    align.set_defaults(func=cmd_align)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
