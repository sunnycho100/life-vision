"""Audit two pinned upstream code paths using extracted source and synthetic inputs only."""
import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path

COMMIT = "2dfd33d6da45ec795cd35e47a683ea48d9f59199"


def source_at(commit, path):
    return subprocess.run(["git", "show", f"{commit}:{path}"], check=True, capture_output=True, text=True).stdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/upstream-model-audit.json"))
    args = parser.parse_args()
    detect_path, timer_path = "model/detect_people_rfdetr_s.py", "model/detect_drowning.py"
    detect_src, timer_src = source_at(COMMIT, detect_path), source_at(COMMIT, timer_path)
    detect_tree, timer_tree = ast.parse(detect_src), ast.parse(timer_src)
    main_fn = next(n for n in detect_tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    start = next(i for i, n in enumerate(main_fn.body) if isinstance(n, ast.Assign) and
                 isinstance(n.targets[0], ast.Name) and n.targets[0].id == "order")
    end = next(i for i, n in enumerate(main_fn.body) if i > start and isinstance(n, ast.Assign) and
               isinstance(n.targets[0], ast.Name) and n.targets[0].id == "out")
    extracted = ast.fix_missing_locations(ast.Module(body=main_fn.body[start:end], type_ignores=[]))
    timer_class = next(n for n in timer_tree.body if isinstance(n, ast.ClassDef) and n.name == "Person")

    import numpy as np
    from collections import deque
    from types import SimpleNamespace

    env = {"np": np, "B": np.asarray([[0., 0., 100., 100.], [40., 40., 60., 60.],
                                       [200., 0., 220., 20.]], dtype=float),
           "S": np.asarray([.9, .8, .7])}
    exec(compile(extracted, f"{detect_path}:duplicate_filter", "exec"), env)
    kept = [int(i) for i in env["final"]]
    assert kept == [0, 2], kept
    # Nested 20x20 box inside 100x100 box: IoU=.04, below upstream NMS=.50.
    nested_iou = 400 / 10000
    assert nested_iou < .5

    person_env = {"deque": deque, "np": np, "fps": 10, "UNDER": 1, "PERSON_ONLY": True,
                  "args": SimpleNamespace(alarm=12, still=8, warn=5)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[timer_class], type_ignores=[])),
                 f"{timer_path}:Person", "exec"), person_env)
    person = person_env["Person"](1, 4., [0., 0., 10., 20.])
    person.under_since = 1.0  # Externally known submerged before this visible detection.
    person.update(5., [0., 0., 10., 20.], 0)
    assert person.under_since is None

    args_ast = next(n for n in detect_tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    parser_call = next(n for n in ast.walk(args_ast) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "add_argument" and n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value == "--thr")
    thr_default = next(k.value.value for k in parser_call.keywords if k.arg == "default")
    conf_call = next(n for n in ast.walk(timer_tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "add_argument" and n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value == "--conf")
    conf_default = next(k.value.value for k in conf_call.keywords if k.arg == "default")
    low_line = next(line.strip() for line in timer_src.splitlines() if "track_low_thresh: 0.1" in line)

    report = {
        "schema": "upstream-model-audit/1", "commit": COMMIT,
        "source_hash_format": "SHA-256 of UTF-8 source with normalized newlines",
        "sources": {detect_path: hashlib.sha256(detect_src.encode()).hexdigest(),
                    timer_path: hashlib.sha256(timer_src.encode()).hexdigest()},
        "scope": "Static source inspection plus synthetic code-path diagnostics only; not model accuracy, field performance, or real person outcomes.",
        "duplicate_filter": {"extracted": "main statements from order=np.argsort(-S) through before out=img.copy",
                             "nms_iou_cutoff": 0.5, "nested_iou": nested_iou,
                             "containment_overlap_of_smaller_cutoff": 0.8,
                             "fragment_min_max_area_ratio_cutoff": 0.35,
                             "fragment_overlap_of_smaller_cutoff": 0.4,
                             "synthetic_boxes": 3, "kept_indices": kept,
                             "finding": "A nested separate-person box survives IoU NMS at 0.5 but is removed by containment filtering; the disjoint box remains."},
        "person_timer": {"extracted": "Person class only", "person_only_update_clears_under_since": person.under_since is None,
                         "finding": "In PERSON_ONLY mode, every visible Person.update resets under_since, even when the synthetic initial timer represents externally known submersion."},
        "thresholds": {"rfdetr_cli_thr_default": thr_default, "tracker_low_threshold": low_line,
                       "yolo_cli_conf_default": conf_default,
                       "finding": "RF-DETR CLI threshold default and YOLO detector confidence default are distinct from ByteTrack track_low_thresh."}
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
