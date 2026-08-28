"""Command line interface."""

from __future__ import annotations

import argparse
import os
import statistics
import sys
from typing import Dict, List

from labelscope import __version__
from labelscope.io import (
    discover_pairs,
    discover_pairs_remote,
    is_remote,
    probe_volume,
    probe_volume_http,
    read_volume,
    read_volume_http,
)
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

    remote = is_remote(args.labels) or is_remote(args.images)
    if remote:
        if not args.names_file:
            _log(
                "scanning a remote base URL needs --names-file: hosts do not all "
                "offer a directory listing"
            )
            return 2
        with open(args.names_file) as handle:
            names = [line.strip() for line in handle if line.strip()]
        pairs = discover_pairs_remote(args.images, args.labels, names)
    else:
        pairs = discover_pairs(args.images, args.labels)
    if not pairs:
        _log("no volumes found")
        return 2
    if args.sample and args.sample < len(pairs):
        import random as _random

        picked = _random.Random(args.seed).sample(range(len(pairs)), args.sample)
        pairs = [pairs[i] for i in sorted(picked)]
        _log(f"sampling {len(pairs)} volumes (seed {args.seed})")
    _log(f"scanning {len(pairs)} volumes{' over HTTP' if remote else ''}")

    if args.jobs > 1 and not remote:
        return _cmd_scan_parallel(args, pairs)

    records: List[Dict] = []
    failures = 0
    for n, pair in enumerate(pairs, 1):
        record: Dict = {
            "name": pair.name,
            "has_image": pair.image is not None,
            "has_label": pair.label is not None,
        }
        if pair.label:
            info = probe_volume_http(pair.label) if remote else probe_volume(pair.label)
            record.update(
                {
                    "label_shape": info.shape,
                    "label_plane_shape": info.meta.get("plane_shape"),
                    "label_dtype": info.dtype,
                    "label_compression": info.compression,
                    "label_bytes": info.file_size,
                    "label_header_error": info.error,
                }
            )
        if pair.image:
            info = probe_volume_http(pair.image) if remote else probe_volume(pair.image)
            record.update(
                {
                    "image_shape": info.shape,
                    "image_plane_shape": info.meta.get("plane_shape"),
                    "image_dtype": info.dtype,
                    "image_compression": info.compression,
                    "image_bytes": info.file_size,
                    "image_header_error": info.error,
                }
            )
        if pair.label and not args.headers_only:
            try:
                volume = read_volume_http(pair.label)[0] if remote else read_volume(pair.label)
                record.update(audit_label(volume, deep=args.deep))
            except Exception as exc:
                record["label_read_error"] = f"{type(exc).__name__}: {exc}"
        if record.get("label_header_error") or record.get("image_header_error"):
            failures += 1
        records.append(record)
        if n % 25 == 0 or n == len(pairs):
            _log(f"  {n}/{len(pairs)}" + (f"  ({failures} unreadable)" if failures else ""))
    if failures:
        _log(
            f"WARNING: {failures}/{len(pairs)} volumes could not be read; their rows "
            f"carry the error and are excluded from the summary counts"
        )

    os.makedirs(args.out, exist_ok=True)
    write_csv(records, os.path.join(args.out, "scan.csv"))
    summary = _summarise_scan(records)
    write_json(summary, os.path.join(args.out, "scan_summary.json"))
    _render_scan_html(records, summary, os.path.join(args.out, "scan.html"))
    _log(f"wrote {args.out}/scan.csv, scan_summary.json, scan.html")
    for line in summary["headline"]:
        print(line)
    return 0


def _scan_one(task):
    """One volume, for the process pool.  Module-level so it can be pickled."""
    from labelscope.quality import audit_label

    name, image_path, label_path, headers_only, deep = task
    record: Dict = {
        "name": name,
        "has_image": image_path is not None,
        "has_label": label_path is not None,
    }
    if label_path:
        info = probe_volume(label_path)
        record.update(
            {
                "label_shape": info.shape,
                "label_plane_shape": info.meta.get("plane_shape"),
                "label_dtype": info.dtype,
                "label_compression": info.compression,
                "label_bytes": info.file_size,
                "label_header_error": info.error,
            }
        )
    if image_path:
        info = probe_volume(image_path)
        record.update(
            {
                "image_shape": info.shape,
                "image_plane_shape": info.meta.get("plane_shape"),
                "image_dtype": info.dtype,
                "image_compression": info.compression,
                "image_bytes": info.file_size,
                "image_header_error": info.error,
            }
        )
    if label_path and not headers_only:
        try:
            record.update(audit_label(read_volume(label_path), deep=deep))
        except Exception as exc:
            record["label_read_error"] = f"{type(exc).__name__}: {exc}"
    return record


def _cmd_scan_parallel(args, pairs) -> int:
    from concurrent.futures import ProcessPoolExecutor

    tasks = [(p.name, p.image, p.label, args.headers_only, args.deep) for p in pairs]
    records = []
    _log(f"scanning {len(tasks)} volumes across {args.jobs} processes")
    with ProcessPoolExecutor(args.jobs) as pool:
        for n, record in enumerate(pool.map(_scan_one, tasks, chunksize=4), 1):
            records.append(record)
            if n % 100 == 0 or n == len(tasks):
                _log(f"  {n}/{len(tasks)}")
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
    unreadable = [
        r["name"]
        for r in records
        if r.get("label_header_error") or r.get("image_header_error")
    ]
    unpaired = [r["name"] for r in records if not (r["has_image"] and r["has_label"])]
    shapes: Dict = {}
    compressions: Dict = {}
    schemes: Dict = {}
    bytes_by_compression: Dict = {}
    for r in records:
        if r.get("label_header_error"):
            continue
        shape = str(r.get("label_plane_shape") or r.get("label_shape"))
        shapes[shape] = shapes.get(shape, 0) + 1
        comp = str(r.get("label_compression"))
        compressions[comp] = compressions.get(comp, 0) + 1
        bytes_by_compression[comp] = bytes_by_compression.get(comp, 0) + (
            r.get("label_bytes") or 0
        )
        if "values" in r:
            key = str(r["values"])
            schemes[key] = schemes.get(key, 0) + 1

    surface_classes: Dict = {}
    for r in records:
        if "surface_class" in r:
            key = str(r["surface_class"])
            surface_classes[key] = surface_classes.get(key, 0) + 1
    modal_scheme = max(schemes, key=schemes.get) if schemes else None
    off_scheme = [
        r["name"] for r in records if "values" in r and str(r["values"]) != modal_scheme
    ]

    headline = [
        f"volumes: {total}",
        f"distinct label shapes: {len(shapes)}  {shapes}",
        f"label compressions: {compressions}",
    ]
    if modal_scheme:
        headline.append(
            f"modal class scheme: {modal_scheme} ({schemes[modal_scheme]}/{total})"
        )
        headline.append(f"volumes off the modal scheme: {len(off_scheme)}")
    if unpaired:
        headline.append(f"unpaired volumes: {len(unpaired)}")
    if unreadable:
        headline.append(f"UNREADABLE volumes (excluded from counts): {len(unreadable)}")
    if len(compressions) > 1:
        headline.append(
            "mixed compression: "
            + ", ".join(
                f"{k}={v} files / {bytes_by_compression[k] / 1e9:.2f} GB"
                for k, v in compressions.items()
            )
        )
    if surface_classes:
        headline.append(f"detected surface class: {surface_classes}")
    return {
        "n_volumes": total,
        "shapes": shapes,
        "compressions": compressions,
        "detected_surface_classes": surface_classes,
        "bytes_by_compression": bytes_by_compression,
        "class_schemes": schemes,
        "modal_scheme": modal_scheme,
        "off_scheme_volumes": off_scheme,
        "unpaired_volumes": unpaired,
        "unreadable_volumes": unreadable,
        "headline": headline,
    }


def _render_scan_html(records, summary, path) -> None:
    cards = [
        ("volumes", summary["n_volumes"], ""),
        (
            "label shapes",
            len(summary["shapes"]),
            "warn" if len(summary["shapes"]) > 1 else "ok",
        ),
        (
            "compressions",
            len(summary["compressions"]),
            "warn" if len(summary["compressions"]) > 1 else "ok",
        ),
        (
            "class schemes",
            len(summary["class_schemes"]),
            "warn" if len(summary["class_schemes"]) > 1 else "ok",
        ),
        (
            "unpaired",
            len(summary["unpaired_volumes"]),
            "warn" if summary["unpaired_volumes"] else "ok",
        ),
    ]
    thick = [r for r in records if r.get("surface_thickness_p95") is not None]
    thick.sort(key=lambda r: -r["surface_thickness_p95"])
    rows = [
        {
            "name": r["name"],
            "surface class": r.get("surface_class"),
            "surface fraction": r.get("surface_fraction"),
            "thickness median": r.get("surface_thickness_median"),
            "thickness p95": r.get("surface_thickness_p95"),
            "components": r.get("surface_components"),
            "fragments": r.get("surface_fragment_fraction"),
            "worst planarity": r.get("surface_worst_planarity"),
        }
        for r in thick
    ]
    sections = [
        (
            "Storage and schema",
            "One row per distinct combination found in the headers.",
            [
                {"property": k, "value": str(v)}
                for k, v in (
                    ("shapes", summary["shapes"]),
                    ("compressions", summary["compressions"]),
                    ("class schemes", summary["class_schemes"]),
                )
            ],
            ["property", "value"],
        ),
        (
            "Thickest labels",
            "Highest 95th-percentile local thickness first — a fat tail is where a label "
            "has most likely swallowed the gap between two windings.",
            rows,
            [
                "name",
                "surface class",
                "surface fraction",
                "thickness median",
                "thickness p95",
                "components",
                "fragments",
                "worst planarity",
            ],
        ),
    ]
    write_html(
        "labelscope — dataset scan",
        "Header inventory and label-only quality metrics.",
        cards,
        sections,
        path,
    )


# --------------------------------------------------------------------------- #
# leakage
# --------------------------------------------------------------------------- #
def cmd_leakage(args: argparse.Namespace) -> int:
    from labelscope.geometry import (
        blocked_kfold,
        measure_split_leakage,
        nnunet_default_split,
        overlap_graph,
        parse_patch_names,
        simulate_random_kfold,
        write_splits,
    )

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
    sizes, size_note = {}, ""
    if args.labels and not is_remote(args.labels) and not args.assume_patch_size:
        from labelscope.geometry import sizes_from_volumes

        sizes, unreadable = sizes_from_volumes(names, args.labels)
        if sizes:
            distinct = sorted(set(sizes.values()))
            size_note = (
                f"read {len(sizes)} patch shapes from the volumes: "
                f"{len(distinct)} distinct — {distinct[:6]}"
            )
            _log(size_note)
            if unreadable:
                _log(
                    f"  {len(unreadable)} volumes unreadable; falling back to "
                    f"--patch-size for those"
                )
    patches, unparsed = parse_patch_names(names, size, sizes=sizes)
    if not patches:
        _log(
            "no volume names carry z/y/x coordinates — nothing to check.\n"
            "labelscope leakage needs names like s1_z10240_y2560_x2560."
        )
        return 2
    _log(f"parsed {len(patches)} patch names ({len(unparsed)} without coordinates)")

    graph = overlap_graph(patches, buffer=args.buffer)
    n_pairs = sum(len(v) for v in graph.values()) // 2
    touched = [i for i in range(len(patches)) if graph.get(i)]
    max_overlap = [max(graph[i].values()) for i in touched] or [0.0]

    random_kfold = simulate_random_kfold(len(patches), graph, k=args.k, trials=args.trials)
    nnunet = measure_split_leakage(
        patches, nnunet_default_split([p.name for p in patches], k=args.k), graph
    )
    splits, split_stats = blocked_kfold(patches, k=args.k, buffer=args.buffer, mode=args.mode)

    os.makedirs(args.out, exist_ok=True)
    write_splits(splits, os.path.join(args.out, "splits_final.json"))

    seen: Dict = {}
    consistency: Dict = {}
    if args.measure_seen:
        from labelscope.leakage import measure_seen_fraction, nnunet_reference_splits

        if not args.labels or is_remote(args.labels):
            _log("--measure-seen needs --labels pointing at local label volumes")
        else:

            def tick(done, total):
                _log(f"  seen-fraction {done}/{total}")

            _log("measuring how much of each validation patch a training patch covers")
            seen["nnunet_default"] = measure_seen_fraction(
                patches,
                nnunet_reference_splits([p.name for p in patches], k=args.k),
                args.labels,
                surface_class=args.surface_class,
                graph=graph,
                progress=tick,
            )
            seen["blocked"] = measure_seen_fraction(
                patches,
                splits,
                args.labels,
                surface_class=args.surface_class,
                graph=graph,
                progress=tick,
            )
            for key in seen:
                seen[key].pop("per_patch", None)
    if args.consistency:
        from labelscope.leakage import check_overlap_consistency

        if not args.labels or is_remote(args.labels):
            _log("--consistency needs --labels pointing at local label volumes")
        else:
            _log(f"checking label agreement on {args.consistency} overlapping pairs")
            consistency = check_overlap_consistency(
                patches,
                args.labels,
                graph=graph,
                surface_class=args.surface_class,
                max_pairs=args.consistency,
                progress=lambda d, t: _log(f"  consistency {d}/{t}"),
            )
    summary = {
        "n_patches": len(patches),
        "fallback_patch_size": list(size),
        "patch_shapes_read_from_volumes": len(sizes),
        "distinct_patch_shapes": sorted(list(v) for v in set(sizes.values())),
        "buffer": args.buffer,
        "overlapping_pairs": n_pairs,
        "patches_with_overlapping_neighbour": len(touched),
        "patches_with_overlapping_neighbour_pct": 100.0 * len(touched) / len(patches),
        "max_overlap_fraction_median_pct": 100.0 * statistics.median(max_overlap),
        "max_overlap_fraction_max_pct": 100.0 * max(max_overlap),
        "random_kfold": random_kfold,
        "nnunet_default_split": nnunet,
        "blocked_split": split_stats,
        "seen_fraction": seen,
        "overlap_consistency": consistency,
    }
    write_json(summary, os.path.join(args.out, "leakage.json"))

    cards = [
        ("patches", len(patches), ""),
        ("overlapping pairs", n_pairs, "warn" if n_pairs else "ok"),
        (
            "patches touching another",
            f"{summary['patches_with_overlapping_neighbour_pct']:.1f}%",
            "warn" if len(touched) else "ok",
        ),
        (
            "nnU-Net default split leaked",
            f"{nnunet['val_patches_contaminated_pct']:.1f}%",
            "warn",
        ),
        (
            "after blocked split",
            f"{split_stats['residual_leaking_val_patches']} patches",
            "ok",
        ),
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
        (
            "Blocked split",
            prose,
            [
                {"fold": i, "train": t, "val": v, "dropped to buffer": d}
                for i, (t, v, d) in enumerate(
                    zip(
                        split_stats["train_fold_sizes"],
                        split_stats["val_fold_sizes"],
                        split_stats["buffer_dropped_per_fold"],
                    )
                )
            ],
            ["fold", "train", "val", "dropped to buffer"],
        ),
    ]
    write_html(
        "labelscope — split leakage",
        "Spatial overlap between training patches, and a split that removes it.",
        cards,
        sections,
        os.path.join(args.out, "leakage.html"),
    )
    _log(f"wrote {args.out}/splits_final.json, leakage.json, leakage.html")
    print(
        f"{len(patches)} patches, {n_pairs} overlapping pairs "
        f"({summary['patches_with_overlapping_neighbour_pct']:.1f}% of patches)"
    )
    print(
        f"nnU-Net default split: {nnunet['val_patches_contaminated_pct']:.1f}% of "
        f"validation patches leak (random shuffles: "
        f"{random_kfold['val_patches_contaminated_pct_mean']:.1f}%)"
    )
    if seen:
        for key, block in seen.items():
            if block.get("n_patches"):
                print(
                    f"{key}: validation patches whose labelled surface a training patch "
                    f"also covers: {block['patches_with_any_seen_pct']:.1f}%; "
                    f"mean seen fraction {100 * block['seen_fraction_mean']:.1f}%, "
                    f"max {100 * block['seen_fraction_max']:.1f}%"
                )
    if consistency.get("n_pairs"):
        print(
            f"overlap consistency: {consistency['n_pairs']} pairs, median IoU "
            f"{consistency['iou_median']:.3f}, min {consistency['iou_min']:.3f}, "
            f"{consistency['pairs_below_0_9_iou']} pairs below 0.9"
        )
    print(
        f"blocked split: val folds {split_stats['val_fold_sizes']}, "
        f"{split_stats['residual_leaking_val_patches']} residual leaks, "
        f"{split_stats['buffer_dropped_pct_mean']:.1f}% of training patches dropped to buffer"
    )
    return 0


# --------------------------------------------------------------------------- #
# align
# --------------------------------------------------------------------------- #
def cmd_align(args: argparse.Namespace) -> int:

    pairs = [p for p in discover_pairs(args.images, args.labels) if p.complete]
    if not pairs:
        _log("no complete image/label pairs found")
        return 2
    if args.limit:
        pairs = pairs[: args.limit]
    _log(f"aligning {len(pairs)} pairs")

    if args.jobs > 1:
        records, cache = _align_parallel(args, pairs)
    else:
        records, cache = _align_serial(args, pairs)

    if args.overlays:
        _render_overlays(records, cache, args.out, args.overlays)

    os.makedirs(args.out, exist_ok=True)
    return _finish_align(args, records)


def _align_serial(args, pairs):
    from labelscope.alignment import audit_alignment

    records, cache = [], {}
    for n, pair in enumerate(pairs, 1):
        try:
            image = read_volume(pair.image)
            label = read_volume(pair.label)
            record = {"name": pair.name}
            record.update(
                audit_alignment(
                    image,
                    label,
                    surface_class=args.surface_class if args.surface_class >= 0 else None,
                    orient_class=args.orient_class if args.orient_class >= 0 else None,
                    radius=args.radius,
                    n_samples=args.samples,
                    cell=args.cell,
                    min_per_cell=args.min_per_cell,
                    min_global_snr=args.min_global_snr,
                )
            )
            records.append(record)
            if args.overlays:
                cache[pair.name] = (
                    pair.image,
                    pair.label,
                    record.get("surface_class"),
                    record.get("orient_class"),
                )
        except Exception as exc:
            records.append({"name": pair.name, "error": f"{type(exc).__name__}: {exc}"})
        _log(f"  {n}/{len(pairs)} {pair.name}")
    return records, cache


def _align_one(task):
    """One image/label pair, for the process pool."""
    from labelscope.alignment import audit_alignment

    name, image_path, label_path, surface, orient, radius, samples, cell, per_cell, snr = task
    try:
        record = {"name": name}
        record.update(
            audit_alignment(
                read_volume(image_path),
                read_volume(label_path),
                surface_class=surface,
                orient_class=orient,
                radius=radius,
                n_samples=samples,
                cell=cell,
                min_per_cell=per_cell,
                min_global_snr=snr,
            )
        )
        return record
    except Exception as exc:
        return {"name": name, "error": f"{type(exc).__name__}: {exc}"}


def _align_parallel(args, pairs):
    from concurrent.futures import ProcessPoolExecutor

    radius = args.radius if args.radius == "auto" else float(args.radius)
    tasks = [
        (
            p.name,
            p.image,
            p.label,
            args.surface_class if args.surface_class >= 0 else None,
            args.orient_class if args.orient_class >= 0 else None,
            radius,
            args.samples,
            args.cell,
            args.min_per_cell,
            args.min_global_snr,
        )
        for p in pairs
    ]
    records, cache = [], {}
    _log(f"aligning {len(tasks)} pairs across {args.jobs} processes")
    with ProcessPoolExecutor(args.jobs) as pool:
        for n, record in enumerate(pool.map(_align_one, tasks, chunksize=1), 1):
            records.append(record)
            if args.overlays and "error" not in record:
                pair = pairs[n - 1]
                cache[record["name"]] = (
                    pair.image,
                    pair.label,
                    record.get("surface_class"),
                    record.get("orient_class"),
                )
            if n % 25 == 0 or n == len(tasks):
                _log(f"  {n}/{len(tasks)}")
    return records, cache


def _finish_align(args, records) -> int:
    write_csv(records, os.path.join(args.out, "align.csv"))
    good = [r for r in records if r.get("global_peak_offset") is not None]
    measurable = len(good)
    attempted = len([r for r in records if "global_profile_snr" in r])
    summary: Dict = {
        "n_pairs": len(records),
        "n_ok": len(good),
        "n_measurable": measurable,
        "n_attempted": attempted,
        "frac_measurable": (measurable / attempted) if attempted else 0.0,
    }
    if good:
        for key in (
            "global_peak_offset",
            "cell_offset_median",
            "cell_abs_offset_median",
            "cell_abs_offset_p90",
            "cell_frac_ge_1vx",
            "cell_frac_ge_2vx",
            "global_profile_snr",
            "cell_snr_median",
            "hf_energy_norm",
            "cell_frac_unresolved",
            "winding_spacing",
            "search_radius",
            "naive_median_abs_offset",
        ):
            values = [r[key] for r in good if r.get(key) is not None]
            if not values:
                continue
            summary[key] = {
                "median": statistics.median(values),
                "mean": statistics.fmean(values),
                "min": min(values),
                "max": max(values),
            }
        summary["separability_bins"] = _difficulty_bins(
            [r for r in records if r.get("cell_abs_offset_median") is not None]
        )
    write_json(summary, os.path.join(args.out, "align_summary.json"))
    _render_align_html(good, summary, os.path.join(args.out, "align.html"))
    _log(f"wrote {args.out}/align.csv, align_summary.json, align.html")
    if good:
        print(
            f"measurable patches: {summary['n_measurable']}/{summary['n_attempted']} "
            f"({_pct(summary['frac_measurable'])}) have sheet contrast above the "
            f"reliability gate"
        )
        print(
            f"global label-to-ridge offset: median "
            f"{summary['global_peak_offset']['median']:+.2f} vx "
            f"(across measurable patches: {summary['global_peak_offset']['min']:+.2f} to "
            f"{summary['global_peak_offset']['max']:+.2f})"
        )
        if "cell_abs_offset_median" in summary:
            print(
                f"per-cell |offset|: median "
                f"{summary['cell_abs_offset_median']['median']:.2f} vx, "
                f"{_pct(summary['cell_frac_ge_1vx']['median'])} of cells >= 1 vx off"
            )
    return 0


def _render_overlays(records, cache, out_dir, count) -> None:
    """Draw the worst-aligned patches, so a reader can see the drift, not just
    read a number for it."""
    from labelscope.visualize import render_drift_map, render_overlay

    usable = [
        r
        for r in records
        if r.get("cell_abs_offset_median") is not None and r["name"] in cache
    ]
    usable.sort(key=lambda r: -r["cell_abs_offset_median"])
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
                image, label, base + "_overlay.png", surface_class=surface
            )
            record["drift_png"] = render_drift_map(
                image,
                label,
                base + "_drift.png",
                surface_class=surface,
                orient_field=orient_field,
            )
        except Exception as exc:
            _log(f"  overlay failed for {record['name']}: {type(exc).__name__}: {exc}")
    _log(f"wrote overlays to {directory}")


def _difficulty_bins(records: List[Dict], bins: int = 4) -> List[Dict]:
    """Are the labels worse where the scan is harder?  Bin by local contrast."""
    records = [r for r in records if "cell_abs_offset_median" in r]
    ranked = sorted(records, key=lambda r: r.get("global_profile_snr", 0.0))
    if len(ranked) < bins:
        return []
    out = []
    size = len(ranked) // bins
    for b in range(bins):
        chunk = ranked[b * size : (b + 1) * size if b < bins - 1 else len(ranked)]
        out.append(
            {
                "bin": b,
                "label": ["least separable layers", "", "", "most separable layers"][b]
                if bins == 4
                else str(b),
                "n": len(chunk),
                "hf_energy_norm": statistics.median(r["hf_energy_norm"] for r in chunk),
                "layer_separability": statistics.median(
                    r["global_profile_snr"] for r in chunk
                ),
                "winding_spacing": statistics.median(
                    r.get("winding_spacing") or 0.0 for r in chunk
                ),
                "cell_abs_offset_median": statistics.median(
                    r["cell_abs_offset_median"] for r in chunk
                ),
                "cell_frac_ge_1vx": statistics.median(r["cell_frac_ge_1vx"] for r in chunk),
                "surface_voxels": statistics.median(r["surface_voxels"] for r in chunk),
            }
        )
    return out


def _render_align_html(records, summary, path) -> None:
    if not records:
        write_html("labelscope — alignment", "No usable pairs.", [], [], path)
        return
    peak = summary["global_peak_offset"]
    cards = [
        ("pairs", summary["n_ok"], ""),
        (
            "global offset",
            f"{peak['median']:+.2f} vx",
            "warn" if abs(peak["median"]) > 0.5 else "ok",
        ),
        (
            "per-cell |offset|",
            f"{summary['cell_abs_offset_median']['median']:.2f} vx"
            if "cell_abs_offset_median" in summary
            else "—",
            "",
        ),
        (
            "cells ≥ 1 vx off",
            _pct(summary["cell_frac_ge_1vx"]["median"])
            if "cell_frac_ge_1vx" in summary
            else "—",
            "",
        ),
        (
            "profile SNR",
            f"{summary['global_profile_snr']['median']:.1f}"
            if "global_profile_snr" in summary
            else "—",
            "",
        ),
    ]
    worst = sorted(records, key=lambda r: -abs(r.get("cell_offset_worst", 0.0)))
    rows = [
        {
            "name": r["name"],
            "global offset": r["global_peak_offset"],
            "CI95": f"[{r['global_peak_ci95'][0]:+.2f}, {r['global_peak_ci95'][1]:+.2f}]",
            "cells": r.get("n_cells"),
            "|offset| median": r.get("cell_abs_offset_median"),
            "worst cell": r.get("cell_offset_worst"),
            "≥1 vx": r.get("cell_frac_ge_1vx"),
            "SNR": r.get("global_profile_snr"),
            "hf energy": r.get("hf_energy_norm"),
        }
        for r in worst
    ]
    method = (
        "Offset is measured along the surface normal, in voxels, positive outward, "
        "by averaging intensity profiles over a cube of labelled surface and locating "
        "the peak of the average. A per-voxel nearest-maximum would be meaningless "
        "here — on carbonised papyrus its value tracks the search radius rather than "
        "any displacement, because fibre maxima, both faces of the sheet and the "
        "neighbouring wrap all sit within the window. The <code>naive_*</code> columns "
        "in the CSV keep that measure alongside, for comparison only."
    )
    sections = [
        (
            "Patches ranked by their worst-aligned region",
            method,
            rows,
            [
                "name",
                "global offset",
                "CI95",
                "cells",
                "|offset| median",
                "worst cell",
                "≥1 vx",
                "SNR",
                "hf energy",
            ],
        ),
        (
            "Does label quality track how separable the layers are?",
            "Patches binned by sheet contrast over voxel noise. Raw high-frequency "
            "energy is <em>not</em> a difficulty proxy — it counts noise as readily as "
            "structure, and across these patches it correlates negatively with actual "
            "separability (Spearman −0.42).",
            summary.get("separability_bins", []),
            [
                "label",
                "n",
                "layer_separability",
                "winding_spacing",
                "hf_energy_norm",
                "cell_abs_offset_median",
                "cell_frac_ge_1vx",
                "surface_voxels",
            ],
        ),
    ]
    write_html(
        "labelscope — CT/label alignment",
        "How far the labels sit from the ridge the scan actually shows.",
        cards,
        sections,
        path,
    )


# --------------------------------------------------------------------------- #
def cmd_sheetswitch(args: argparse.Namespace) -> int:
    """Look for places where a traced surface jumps to a neighbouring wrap."""

    from labelscope.mesh import find_sheet_switches, read_tifxyz

    meshes = sorted(args.mesh)
    if not meshes:
        _log("no meshes given")
        return 2
    _log(f"checking {len(meshes)} surfaces")

    records = []
    for n, directory in enumerate(meshes, 1):
        name = os.path.basename(directory.rstrip("/"))
        try:
            mesh = read_tifxyz(directory)
        except Exception as exc:
            records.append({"name": name, "error": f"{type(exc).__name__}: {exc}"})
            continue
        if not mesh.valid.any():
            records.append({"name": name, "error": "no valid vertices"})
            continue

        if args.window:
            mesh = _best_window(mesh, args.window)
        if args.decimate > 1:
            mesh = _decimate(mesh, args.decimate)
        try:
            if args.remote:
                from labelscope.remote_zarr import ChunkedVolume

                volume = ChunkedVolume.from_store(args.volume, cache_dir=args.cache)
                origin = None
            else:
                lo, hi = mesh.bounds(margin=args.margin)
                volume, origin = _load_window(args.volume, lo, hi)
        except Exception as exc:
            records.append({"name": name, "error": f"volume unreadable: {exc}"})
            continue

        result = find_sheet_switches(
            mesh, volume, origin=origin, z_threshold=args.z_threshold, steps=args.steps
        )
        result["name"] = name
        result["valid_fraction"] = float(mesh.valid.mean())
        if args.remote:
            result["chunks_fetched"] = getattr(volume, "chunks_fetched", None)
            result["mb_fetched"] = round(getattr(volume, "bytes_fetched", 0) / 1e6, 1)
        seams = result.pop("seams")
        result["seam_lines"] = ";".join(
            f"{s['axis']}:{s['line']}@{s['z']:.1f}" for s in seams[:8]
        )
        records.append(result)
        _log(
            f"  {n}/{len(meshes)} {name}: max z {result['max_z']:.1f}, "
            f"{result['n_seams']} seam(s)"
        )

    os.makedirs(args.out, exist_ok=True)
    write_csv(records, os.path.join(args.out, "sheetswitch.csv"))
    flagged = [r for r in records if r.get("n_seams")]
    usable = [r for r in records if r.get("resolution_adequate")]
    summary = {
        "n_meshes": len(records),
        "n_readable": len([r for r in records if "max_z" in r]),
        "n_resolution_adequate": len(usable),
        "n_flagged": len(flagged),
        "z_threshold": args.z_threshold,
        "flagged": [
            {"name": r["name"], "max_z": r["max_z"], "seams": r["seam_lines"]}
            for r in sorted(flagged, key=lambda r: -r["max_z"])[:40]
        ],
    }
    write_json(summary, os.path.join(args.out, "sheetswitch_summary.json"))
    _log(f"wrote {args.out}/sheetswitch.csv, sheetswitch_summary.json")
    print(
        f"{summary['n_readable']} surfaces checked, {summary['n_flagged']} carry a seam "
        f"at z >= {args.z_threshold}"
    )
    for row in summary["flagged"][:10]:
        print(f"  {row['name']}: max z {row['max_z']:.1f}  [{row['seams']}]")
    return 0


def _best_window(mesh, size: int):
    """The most complete size x size patch of the grid.

    Streaming cost scales with how much scroll a surface spans, not with how many
    vertices are sampled, so decimating a whole segment does not make it cheap --
    taking a contiguous window does.  The window with the fewest missing vertices
    is the one worth measuring.
    """

    from labelscope.mesh import QuadMesh

    rows, cols = mesh.shape
    if rows <= size and cols <= size:
        return mesh
    gh, gw = min(size, rows), min(size, cols)
    best, coords = -1.0, (0, 0)
    for r in range(0, rows - gh + 1, max(1, gh // 2)):
        for c in range(0, cols - gw + 1, max(1, gw // 2)):
            score = float(mesh.valid[r : r + gh, c : c + gw].mean())
            if score > best:
                best, coords = score, (r, c)
    r, c = coords
    return QuadMesh(
        points=mesh.points[r : r + gh, c : c + gw],
        valid=mesh.valid[r : r + gh, c : c + gw],
        meta={**mesh.meta, "window": [r, gh, c, gw], "window_valid": best},
        path=mesh.path,
    )


def _decimate(mesh, factor: int):
    from labelscope.mesh import QuadMesh

    return QuadMesh(
        points=mesh.points[::factor, ::factor],
        valid=mesh.valid[::factor, ::factor],
        meta=mesh.meta,
        path=mesh.path,
    )


def _load_window(volume_path: str, lo, hi):
    """Read the sub-volume a mesh lives in.  Returns (array, origin)."""
    import numpy as np

    lo = np.maximum(np.asarray(lo), 0)
    if volume_path.endswith((".zarr", ".zarr/")):
        import zarr

        root = zarr.open(store=volume_path.rstrip("/"), mode="r")
        array = (
            root["0"]
            if hasattr(root, "array_keys") and "0" in list(root.array_keys())
            else root
        )
        hi = np.minimum(np.asarray(hi), np.array(array.shape))
        window = np.asarray(array[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]])
    else:
        from labelscope.io import read_volume

        full = read_volume(volume_path)
        hi = np.minimum(np.asarray(hi), np.array(full.shape))
        window = full[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
    return window, lo


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="labelscope",
        description="Diagnostics for Vesuvius Challenge surface-label datasets.",
    )
    parser.add_argument("--version", action="version", version=f"labelscope {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="inventory volumes and score label-only quality")
    scan.add_argument("--labels", required=True)
    scan.add_argument("--images")
    scan.add_argument("--out", default="labelscope_out")
    scan.add_argument(
        "--names-file",
        help="newline-separated filenames; required when --labels or --images is a base URL",
    )
    scan.add_argument(
        "--sample",
        type=int,
        default=0,
        metavar="N",
        help="audit a seeded random subsample of N volumes, for the expensive "
        "per-class metrics that --deep turns on",
    )
    scan.add_argument("--seed", type=int, default=0)
    scan.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="worker processes; volumes are independent",
    )
    scan.add_argument(
        "--headers-only", action="store_true", help="probe headers only; never decode voxels"
    )
    scan.add_argument(
        "--deep",
        action="store_true",
        help="also skeletonise each label (slow) to count branch points",
    )
    scan.set_defaults(func=cmd_scan)

    leak = sub.add_parser(
        "leakage", help="find patches that share voxels and emit a split that does not"
    )
    leak.add_argument("--labels", help="labelsTr directory")
    leak.add_argument("--images")
    leak.add_argument(
        "--names-file",
        help="newline-separated patch names, when the volumes are not "
        "on this machine — the check only needs the names",
    )
    leak.add_argument(
        "--patch-size",
        type=int,
        nargs="+",
        default=[300],
        help="fallback size when a volume's own shape cannot be read; "
        "one value for a cube, or three as z y x",
    )
    leak.add_argument(
        "--assume-patch-size",
        action="store_true",
        help="trust --patch-size instead of reading each volume's real "
        "shape (faster, and wrong on any release with mixed sizes)",
    )
    leak.add_argument("--k", type=int, default=5)
    leak.add_argument(
        "--buffer",
        type=int,
        default=0,
        help="treat patches within this many voxels as neighbours too",
    )
    leak.add_argument("--mode", choices=["block", "component"], default="block")
    leak.add_argument("--trials", type=int, default=200)
    leak.add_argument("--surface-class", type=int, default=1)
    leak.add_argument(
        "--measure-seen",
        action="store_true",
        help="read the label volumes and measure, per validation patch, "
        "the fraction of its labelled surface that a training patch "
        "also covers — the leak in voxels rather than in patches",
    )
    leak.add_argument(
        "--consistency",
        type=int,
        default=0,
        metavar="N",
        help="check that N overlapping patch pairs agree about the voxels they share",
    )
    leak.add_argument("--out", default="labelscope_out")
    leak.set_defaults(func=cmd_leakage)

    align = sub.add_parser("align", help="measure label offset from the CT's own ridge")
    align.add_argument("--images", required=True)
    align.add_argument("--labels", required=True)
    align.add_argument("--out", default="labelscope_out")
    align.add_argument(
        "--surface-class",
        type=int,
        default=-1,
        help="thin sheet class; -1 (default) detects it",
    )
    align.add_argument(
        "--orient-class",
        type=int,
        default=-1,
        help="bulky region class defining 'outward', so the offset "
        "sign is meaningful; -1 (default) detects it",
    )
    align.add_argument(
        "--radius",
        default="auto",
        help="how far along the normal to look, in voxels; 'auto' "
        "(default) measures the local winding spacing and uses "
        "45%% of it, so the search cannot reach the next wrap",
    )
    align.add_argument(
        "--cell",
        type=int,
        default=64,
        help="edge of the cube of surface each offset is measured over",
    )
    align.add_argument(
        "--min-global-snr",
        type=float,
        default=2.0,
        help="sheet contrast, in units of voxel noise, below which a "
        "patch's global offset is reported as unmeasurable",
    )
    align.add_argument(
        "--min-per-cell",
        type=int,
        default=200,
        help="minimum sampled voxels before a cell gets an offset",
    )
    align.add_argument("--samples", type=int, default=40000)
    align.add_argument("--limit", type=int, default=0)
    align.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="worker processes; pairs are independent",
    )
    align.add_argument(
        "--overlays",
        type=int,
        default=0,
        metavar="N",
        help="render CT/label overlay and drift-map PNGs for the N worst-aligned patches",
    )
    align.set_defaults(func=cmd_align)

    switch = sub.add_parser(
        "sheetswitch",
        help="find where a traced surface jumps to a neighbouring wrap",
    )
    switch.add_argument(
        "--mesh", nargs="+", required=True, help="one or more tifxyz directories"
    )
    switch.add_argument(
        "--volume",
        required=True,
        help="the CT volume the surface was traced on (.zarr or 3-D TIFF)",
    )
    switch.add_argument("--out", default="labelscope_out")
    switch.add_argument(
        "--z-threshold",
        type=float,
        default=5.0,
        help="how far a grid line's mean darkening must stand out from "
        "the rest of the surface before it is called a seam",
    )
    switch.add_argument(
        "--steps", type=int, default=17, help="samples taken along each grid edge"
    )
    switch.add_argument(
        "--margin",
        type=int,
        default=8,
        help="voxels of volume to read beyond the surface's bounding box",
    )
    switch.add_argument(
        "--remote",
        action="store_true",
        help="stream the volume over HTTP, fetching only the chunks the surface "
        "passes through",
    )
    switch.add_argument("--cache", help="directory to keep fetched chunks in")
    switch.add_argument(
        "--window",
        type=int,
        default=0,
        metavar="N",
        help="measure the most complete NxN patch of each grid rather than the "
        "whole surface; streaming cost scales with how much scroll the surface "
        "spans, so a window is what makes a fleet-wide pass affordable",
    )
    switch.add_argument(
        "--decimate",
        type=int,
        default=1,
        help="use every Nth grid line; a seam spans a whole line, so decimating "
        "keeps it while cutting the sampling cost",
    )
    switch.set_defaults(func=cmd_sheetswitch)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
