"""Inference adapter: stub (demo targets) or Docker baseline."""

from __future__ import annotations

import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

from app.config import settings


def _confidence_path_for_mask(mask_path: Path) -> Path:
    return mask_path.parent / f"{mask_path.stem}_confidence.npy"


def list_demo_pairs() -> list[dict[str, str]]:
    images_dir = settings.demo_data_dir / "images"
    if not images_dir.exists():
        return []
    pairs: list[dict[str, str]] = []
    for pre in sorted(images_dir.glob("*_pre_disaster.png")):
        base = pre.stem.replace("_pre_disaster", "")
        post = images_dir / f"{base}_post_disaster.png"
        if not post.exists():
            continue
        disaster = base.split("_")[0]
        if "earthquake" in base:
            dtype = "earthquake"
        elif "flooding" in base or "flood" in base:
            dtype = "flood"
        elif "fire" in base:
            dtype = "wildfire"
        else:
            dtype = disaster
        pairs.append(
            {
                "id": base,
                "disaster_type": dtype,
                "pre_image": pre.name,
                "post_image": post.name,
            }
        )
    return pairs


def resolve_demo_target(post_image_path: Path) -> Path | None:
    stem = post_image_path.stem.replace("_post_disaster", "")
    target = settings.demo_data_dir / "targets" / f"{stem}_post_disaster_target.png"
    if target.exists():
        return target
    test_target = settings.test_data_dir / "targets" / f"{stem}_post_disaster_target.png"
    return test_target if test_target.exists() else None


def stub_diff_mask(pre_path: Path, post_path: Path, out_path: Path) -> Path:
    """Fallback stub when no ground-truth target exists."""
    pre = np.array(Image.open(pre_path).convert("L"), dtype=np.float32)
    post = np.array(Image.open(post_path).convert("L"), dtype=np.float32)
    if pre.shape != post.shape:
        post_img = Image.open(post_path).convert("L").resize((pre.shape[1], pre.shape[0]))
        post = np.array(post_img, dtype=np.float32)
    diff = np.abs(post - pre)
    thresh = max(15, float(np.percentile(diff, 92)))
    building = diff > thresh
    mask = np.zeros(pre.shape, dtype=np.uint8)
    band = diff[building]
    classes = np.full(band.shape, 2, dtype=np.uint8)
    classes[band > thresh * 1.5] = 3
    classes[band > thresh * 2.2] = 4
    mask[building] = classes
    Image.fromarray(mask, mode="L").save(out_path)
    return out_path


def run_stub_inference(pre_path: Path, post_path: Path, out_dir: Path) -> tuple[Path, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    target = resolve_demo_target(post_path)
    out_path = out_dir / f"{uuid.uuid4().hex}_mask.png"
    if target is not None:
        shutil.copy2(target, out_path)
        return out_path, "stub-groundtruth"
    return stub_diff_mask(pre_path, post_path, out_path), "stub-heuristic"


def run_docker_inference(pre_path: Path, post_path: Path, out_dir: Path) -> Path:
    submission_dir = out_dir / "submission"
    output_mount = out_dir / "docker_output"
    submission_dir.mkdir(parents=True, exist_ok=True)
    output_mount.mkdir(parents=True, exist_ok=True)

    loc_out = output_mount / "localization.png"
    cls_out = output_mount / "classification.png"

    shutil.copy2(pre_path, submission_dir / pre_path.name)
    shutil.copy2(post_path, submission_dir / post_path.name)

    pre_in_container = f"/submission/{pre_path.name}"
    post_in_container = f"/submission/{post_path.name}"
    loc_in_container = f"/output/{loc_out.name}"
    cls_in_container = f"/output/{cls_out.name}"

    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{submission_dir.resolve()}:/submission",
        "-v",
        f"{output_mount.resolve()}:/output",
        settings.xview2_docker_image,
        pre_in_container,
        post_in_container,
        loc_in_container,
        cls_in_container,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(
            f"Docker inference failed: {result.stderr or result.stdout}"
        )

    if cls_out.exists():
        return cls_out
    if loc_out.exists():
        return loc_out
    raise RuntimeError("Docker inference produced no output mask")


def run_pytorch_inference(pre_path: Path, post_path: Path, out_dir: Path) -> tuple[Path, Path | None]:
    """Run fine-tuned PyTorch checkpoint via ml/pytorch-inference/infer_pair.py."""
    if not settings.pytorch_checkpoint_path.exists():
        raise RuntimeError(
            f"PyTorch checkpoint not found: {settings.pytorch_checkpoint_path}. "
            "Train on AMD GPU first (ml/finetune/run_amd_pipeline.sh)."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_mask = out_dir / f"{uuid.uuid4().hex}_mask.png"
    infer_script = settings.pytorch_inference_dir / "infer_pair.py"
    if not infer_script.exists():
        raise RuntimeError(f"Missing inference script: {infer_script}")

    cmd = [
        sys.executable,
        str(infer_script),
        "--pre",
        str(pre_path.resolve()),
        "--post",
        str(post_path.resolve()),
        "--checkpoint",
        str(settings.pytorch_checkpoint_path.resolve()),
        "--out",
        str(out_mask.resolve()),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"PyTorch inference failed: {result.stderr or result.stdout}")
    if not out_mask.exists():
        raise RuntimeError("PyTorch inference produced no output mask")
    conf_path = _confidence_path_for_mask(out_mask)
    return out_mask, conf_path if conf_path.exists() else None


def run_inference(pre_path: Path, post_path: Path, out_dir: Path) -> tuple[Path, str, Path | None]:
    mode = settings.inference_mode.lower()
    if mode == "pytorch":
        try:
            mask_path, confidence_path = run_pytorch_inference(pre_path, post_path, out_dir)
            return mask_path, "pytorch", confidence_path
        except (RuntimeError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
            raise RuntimeError(f"PyTorch inference failed: {exc}") from exc
    if mode == "docker":
        try:
            mask_path = run_docker_inference(pre_path, post_path, out_dir)
            return mask_path, "docker", None
        except (RuntimeError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
            raise RuntimeError(f"Docker inference failed: {exc}") from exc
    mask_path, stub_mode = run_stub_inference(pre_path, post_path, out_dir)
    return mask_path, stub_mode, None
