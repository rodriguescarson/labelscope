"""Does regularising the labels give a better surface model?

The Vesuvius Open Problems post suggests the next gains in unwrapping may come
from better labels rather than larger models.  `labelscope regularise` produces
a corrected copy of a label set; this trains the same network twice on the two
copies, under an identical budget, and asks whether the corrected one produces a
model whose predictions sit more consistently on the scan's own surface.

The evaluation is deliberately **label-free**.  Comparing Dice against either
label set is circular -- each arm is favoured by the labels it trained on -- so
the primary metric never looks at a label.  It takes the model's own prediction
on a held-out patch, binarises it, and measures how consistently *that* surface
sits relative to the CT, using the same estimator the audit uses.  A model that
has learned a cleaner surface should produce a prediction whose cells agree with
each other, whichever labels it was taught from.

    python experiments/train_surface.py --labels <dir> --tag orig --seed 0

Every run writes its config, its seeds and its curve, so a pair of runs can be
compared without rerunning either.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
class Patches(torch.utils.data.Dataset):
    """Random ``crop`` cubes from nnU-Net style image/label pairs.

    The patches in Dataset059 are not one size -- 170, 172, 236, 300 and 364
    cubed all live in the same directory -- so nothing here may assume a shape.
    Each item reads its own volume and crops it, and a patch smaller than the
    crop is padded rather than skipped.
    """

    def __init__(
        self,
        names: List[str],
        images: str,
        labels: str,
        crop: int = 128,
        surface_class: int = 1,
        samples_per_epoch: int = 400,
        seed: int = 0,
    ):
        self.names = list(names)
        self.images, self.labels = images, labels
        self.crop, self.surface_class = crop, surface_class
        self.samples_per_epoch = samples_per_epoch
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _read(self, name: str) -> Tuple[np.ndarray, np.ndarray]:
        import tifffile

        image = tifffile.imread(os.path.join(self.images, f"{name}_0000.tif"))
        label = tifffile.imread(os.path.join(self.labels, f"{name}.tif"))
        return image, label

    def __getitem__(self, index: int):
        name = self.names[self.rng.integers(len(self.names))]
        image, label = self._read(name)
        c = self.crop
        pad = [(0, max(0, c - s)) for s in image.shape]
        if any(p[1] for p in pad):
            image = np.pad(image, pad)
            label = np.pad(label, pad)
        start = [int(self.rng.integers(0, s - c + 1)) for s in image.shape]
        sl = tuple(slice(s, s + c) for s in start)
        x = image[sl].astype(np.float32)
        y = (label[sl] == self.surface_class).astype(np.float32)
        # per-crop standardisation: the patches come from three scrolls and do
        # not share an intensity scale
        x = (x - x.mean()) / (x.std() + 1e-6)
        return torch.from_numpy(x)[None], torch.from_numpy(y)[None]


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def block(cin, cout):
    return nn.Sequential(
        nn.Conv3d(cin, cout, 3, padding=1, bias=False),
        nn.InstanceNorm3d(cout, affine=True),
        nn.LeakyReLU(0.01, inplace=True),
        nn.Conv3d(cout, cout, 3, padding=1, bias=False),
        nn.InstanceNorm3d(cout, affine=True),
        nn.LeakyReLU(0.01, inplace=True),
    )


class UNet3D(nn.Module):
    """A compact 3-D U-Net.  Not nnU-Net: its preprocessing would cost ~190 GB
    of disk here for no benefit to the question being asked, which is about the
    labels rather than the architecture."""

    def __init__(self, width=(16, 32, 64, 128)):
        super().__init__()
        self.downs = nn.ModuleList()
        cin = 1
        for w in width:
            self.downs.append(block(cin, w))
            cin = w
        self.ups = nn.ModuleList()
        self.reduce = nn.ModuleList()
        for i in range(len(width) - 1, 0, -1):
            self.reduce.append(nn.ConvTranspose3d(width[i], width[i - 1], 2, stride=2))
            self.ups.append(block(width[i - 1] * 2, width[i - 1]))
        self.head = nn.Conv3d(width[0], 1, 1)

    def forward(self, x):
        skips = []
        for i, down in enumerate(self.downs):
            x = down(x)
            if i < len(self.downs) - 1:
                skips.append(x)
                x = F.max_pool3d(x, 2)
        for reduce, up in zip(self.reduce, self.ups):
            x = reduce(x)
            skip = skips.pop()
            x = up(torch.cat([x, skip], 1))
        return self.head(x)


def dice_bce(logits, target, eps=1.0):
    """Soft dice on *probabilities* plus BCE on logits.

    Feeding raw logits to a soft dice is the bug ScrollPrize/villa#1488 reports
    and villa PR #1644 fixes; it is repeated here deliberately in the corrected
    form, since a loss that returns 1.2e+11 and flips sign would drown any
    effect the labels have.
    """
    probs = torch.sigmoid(logits)
    num = 2 * (probs * target).sum(dim=(2, 3, 4)) + eps
    den = probs.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4)) + eps
    return F.binary_cross_entropy_with_logits(logits, target) + (1 - num / den).mean()


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="/workspace/d059/imagesTr")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--splits", default="findings/full/d059_leakage/splits_final.json")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--crop", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="/workspace/runs")
    args = ap.parse_args(argv)

    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.splits) as handle:
        splits = json.load(handle)[args.fold]
    train_names, val_names = splits["train"], splits["val"]
    have = {f[: -len(".tif")] for f in os.listdir(args.labels) if f.endswith(".tif")}
    train_names = [n for n in train_names if n in have]
    val_names = [n for n in val_names if n in have]

    out = os.path.join(args.out, f"{args.tag}_seed{args.seed}")
    os.makedirs(out, exist_ok=True)
    config = vars(args) | {
        "device": device,
        "n_train": len(train_names),
        "n_val": len(val_names),
        "torch": torch.__version__,
    }
    with open(os.path.join(out, "config.json"), "w") as handle:
        json.dump(config, handle, indent=2)
    print(json.dumps(config, indent=2), flush=True)

    train = Patches(
        train_names,
        args.images,
        args.labels,
        args.crop,
        samples_per_epoch=args.steps * args.batch,
        seed=args.seed,
    )
    val = Patches(
        val_names,
        args.images,
        args.labels,
        args.crop,
        samples_per_epoch=64,
        seed=args.seed + 1000,
    )
    loader = torch.utils.data.DataLoader(
        train,
        batch_size=args.batch,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    val_loader = torch.utils.data.DataLoader(
        val, batch_size=args.batch, num_workers=max(2, args.workers // 2)
    )

    model = UNet3D().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    scaler = torch.amp.GradScaler(device)

    history = []
    best = float("inf")
    for epoch in range(args.epochs):
        model.train()
        started, total = time.time(), 0.0
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device):
                loss = dice_bce(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += float(loss)
        sched.step()

        model.eval()
        vloss, vdice, n = 0.0, 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                with torch.amp.autocast(device):
                    logits = model(x)
                    vloss += float(dice_bce(logits, y)) * x.shape[0]
                p = (torch.sigmoid(logits.float()) > 0.5).float()
                inter = (p * y).sum(dim=(1, 2, 3, 4))
                vdice += float(
                    (
                        2 * inter / (p.sum(dim=(1, 2, 3, 4)) + y.sum(dim=(1, 2, 3, 4)) + 1e-6)
                    ).sum()
                )
                n += x.shape[0]
        row = {
            "epoch": epoch,
            "train_loss": total / max(1, len(loader)),
            "val_loss": vloss / max(1, n),
            "val_dice": vdice / max(1, n),
            "seconds": round(time.time() - started, 1),
            "lr": sched.get_last_lr()[0],
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        # Rewritten every epoch, so it is written to a temp file and renamed:
        # opening the real path with "w" truncates it, and anything reading the
        # run while it trains sees an empty file rather than the last epoch.
        tmp = os.path.join(out, "history.json.tmp")
        with open(tmp, "w") as handle:
            json.dump(history, handle, indent=2)
        os.replace(tmp, os.path.join(out, "history.json"))
        if row["val_loss"] < best:
            best = row["val_loss"]
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "config": config},
                os.path.join(out, "best.pt"),
            )
    torch.save({"model": model.state_dict(), "config": config}, os.path.join(out, "last.pt"))
    print(f"done, best val_loss {best:.4f}, checkpoints in {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
