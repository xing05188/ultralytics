import os
import time
from typing import List, Dict, Any

import matplotlib.pyplot as plt
import pandas as pd

from ultralytics import YOLO


# 下面的字典以配置要比较的模型与验证参数
CONFIG = {
    # 支持：列表、JSON 数组字符串或逗号分隔字符串（内部会规范化）
    "models": [
        r"C:\homework\ultralytics\runs\detect\train\weights\best.pt",
        r"C:\homework\ultralytics\runs\detect\train2\weights\best.pt",
        r"C:\homework\ultralytics\runs\detect\train3\weights\best.pt",
        r"C:\homework\ultralytics\runs\detect\train4\weights\best.pt",
        r"C:\homework\ultralytics\runs\detect\train5\weights\best.pt",
    ],
    "data": r"C:\homework\ultralytics\datasets\Face detection.v1i.yolo26\data.yaml",  # 示例路径，可改为 None 或你的 dataset yaml
    "imgsz": 640,
    "batch": 16,
    "device": None,  # e.g., "0" or "cpu" or "cuda:0"
    "out": "runs/compare",
    "plots": True,
    "half": False,
    "conf": 0.001,
    "iou": 0.7,
}


def safe_get_metric(results, path: List[str]):
    # results may be a Results object or a list containing one Results
    try:
        r = results[0] if isinstance(results, (list, tuple)) and len(results) > 0 else results
        cur = r
        for p in path:
            cur = getattr(cur, p)
        return float(cur)
    except Exception:
        return None


def run_val(model_path: str, data: str, imgsz: int, batch: int, device: str, half: bool, conf: float, iou: float, outdir: str) -> Dict[str, Any]:
    print(f"Validating model: {model_path}")
    t0 = time.time()
    model = YOLO(model_path)
    results = model.val(data=data, imgsz=imgsz, batch=batch, device=device, half=half, conf=conf, iou=iou, plots=False)
    dt = time.time() - t0

    metrics = {
        "model": os.path.basename(model_path),
        "path": model_path,
        "time_s": round(dt, 2),
    }

    # Common boxed metrics: box.map50-95, box.map50, box.map75, box.precision, box.recall
    metrics["map"] = safe_get_metric(results, ["box", "map"]) or safe_get_metric(results, ["metrics", "box", "map"]) 
    metrics["map50"] = safe_get_metric(results, ["box", "map50"]) or safe_get_metric(results, ["metrics", "box", "map50"])
    metrics["map75"] = safe_get_metric(results, ["box", "map75"]) or safe_get_metric(results, ["metrics", "box", "map75"])
    metrics["precision"] = safe_get_metric(results, ["box", "p"]) or safe_get_metric(results, ["box", "precision"]) or safe_get_metric(results, ["metrics", "box", "p"]) or safe_get_metric(results, ["metrics", "box", "precision"]) 
    metrics["recall"] = safe_get_metric(results, ["box", "r"]) or safe_get_metric(results, ["box", "recall"]) or safe_get_metric(results, ["metrics", "box", "r"]) or safe_get_metric(results, ["metrics", "box", "recall"]) 

    # Save per-model results (if available)
    os.makedirs(outdir, exist_ok=True)
    try:
        # Attempt to save full results object exports if it supports export
        if hasattr(results, "to_json"):
            with open(os.path.join(outdir, f"{metrics['model']}_val.json"), "w", encoding="utf-8") as f:
                f.write(results.to_json())
    except Exception:
        pass

    return metrics


def plot_comparison(df: pd.DataFrame, outdir: str):
    df_plot = df.set_index("model")
    cols = [c for c in ["map", "map50", "map75", "precision", "recall"] if c in df_plot.columns]
    if not cols:
        print("No supported metrics to plot.")
        return

    os.makedirs(outdir, exist_ok=True)
    # Bar plot for each metric
    for col in cols:
        plt.figure(figsize=(8, 4))
        vals = df_plot[col].fillna(0) * 100  # convert to percentage if in [0,1]
        vals.plot(kind="bar", color="C0")
        plt.title(f"Model comparison: {col}")
        plt.ylabel("%" if vals.max() <= 100 else "value")
        plt.grid(axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()
        out_path = os.path.join(outdir, f"compare_{col}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved plot: {out_path}")

    # Combined metrics plot (if several metrics exist)
    if len(cols) >= 2:
        plt.figure(figsize=(10, 5))
        df_plot[cols].fillna(0).multiply(100).plot(kind="bar")
        plt.title("Model comparison")
        plt.ylabel("%")
        plt.legend(loc="best")
        plt.grid(axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()
        out_path = os.path.join(outdir, "compare_combined.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved combined plot: {out_path}")


def get_next_outdir(base_root: str = os.path.join("runs", "detect"), prefix: str = "compare") -> str:
    """Return next available directory like runs/detect/compare1, compare2, ..."""
    import re

    os.makedirs(base_root, exist_ok=True)
    entries = [d for d in os.listdir(base_root) if os.path.isdir(os.path.join(base_root, d))]
    nums = []
    for d in entries:
        m = re.match(rf'^{re.escape(prefix)}(\d+)$', d)
        if m:
            try:
                nums.append(int(m.group(1)))
            except Exception:
                pass
    n = max(nums) + 1 if nums else 1
    return os.path.join(base_root, f"{prefix}{n}")


def main():
    # 使用 CONFIG 作为唯一配置来源（避免 CLI）
    cfg = CONFIG
    # compute auto-incremented outdir under runs/detect
    outdir = get_next_outdir()
    os.makedirs(outdir, exist_ok=True)

    # Normalize models input: support list, JSON array string, or comma-separated single string
    models = cfg.get("models", [])
    if isinstance(models, (list, tuple)) and len(models) == 1:
        single = models[0]
        if isinstance(single, str) and single.strip().startswith("["):
            try:
                import json

                parsed = json.loads(single)
                if isinstance(parsed, list):
                    models = parsed
            except Exception:
                pass
        elif isinstance(single, str) and "," in single and not os.path.exists(single):
            models = [s.strip() for s in single.split(",") if s.strip()]

    records = []
    for m in models:
        rec = run_val(
            m,
            data=cfg.get("data"),
            imgsz=cfg.get("imgsz", 640),
            batch=cfg.get("batch", 16),
            device=cfg.get("device"),
            half=cfg.get("half", False),
            conf=cfg.get("conf", 0.001),
            iou=cfg.get("iou", 0.7),
            outdir=outdir,
        )
        records.append(rec)

    df = pd.DataFrame.from_records(records)
    csv_path = os.path.join(outdir, "compare_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved metrics CSV: {csv_path}")

    if cfg.get("plots", True):
        plot_comparison(df, outdir)

    print("Done.")


if __name__ == "__main__":
    main()
