#!/usr/bin/env python3
"""One-batch smoke test for damage training — surfaces loss/GPU errors in ~30s."""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
XVIEW2 = REPO / "ml" / "pytorch-xview2"
sys.path.insert(0, str(XVIEW2))

WORKING = Path(os.environ.get("KAGGLE_WORKING", "/kaggle/working"))
DATA = WORKING / "data" / "train_subset"
LOC_CKPT = WORKING / "results" / "loc" / "checkpoints" / "best.ckpt"
INDEX = XVIEW2 / "utils" / "index.csv"


def main() -> None:
    import torch
    from argparse import Namespace

    from data_loading.data_module import DataModule
    from model.plt import Model

    os.environ["XVIEW2_INDEX_CSV"] = str(INDEX)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for smoke test")

    args = Namespace(
        exec_mode="train",
        data=str(DATA),
        results=str(WORKING / "results" / "dmg"),
        gpus=1,
        num_workers=0,
        batch_size=2,
        val_batch_size=2,
        precision=32,
        epochs=1,
        patience=100,
        ckpt=None,
        logname="logs",
        ckpt_pre=str(LOC_CKPT),
        type="post",
        seed=1,
        interpolate=False,
        optimizer="adamw",
        dmg_model="siamese",
        encoder="resnet50",
        loss_str="focal+dice",
        use_scheduler=False,
        warmup=1,
        init_lr=1e-4,
        final_lr=1e-4,
        lr=3e-4,
        weight_decay=0,
        momentum=0.9,
        dilation=1,
        tta=False,
        ppm=False,
        aspp=False,
        no_skip=False,
        deep_supervision=True,
        attention=True,
        autoaugment=False,
        dec_interp=False,
    )

    os.makedirs(args.results, exist_ok=True)
    dm = DataModule(args)
    loader = dm.train_dataloader()
    batch = next(iter(loader))

    model = Model(args).cuda()
    if LOC_CKPT.is_file():
        state = torch.load(str(LOC_CKPT), map_location="cpu", weights_only=False)["state_dict"]
        keys = model.state_dict()
        for name, tensor in state.items():
            if "enc" in name and name in keys:
                model.state_dict()[name].copy_(tensor)

    img = batch["image"].cuda()
    lbl = batch["mask"].cuda()
    pred = model.model(img)
    loss = model.compute_loss(pred, lbl)
    loss.backward()
    print(f"SMOKE OK: loss={loss.item():.4f} batch={tuple(img.shape)}")


if __name__ == "__main__":
    main()
