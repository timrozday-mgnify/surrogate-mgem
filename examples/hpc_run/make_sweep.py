#!/usr/bin/env python3
"""Write the M3b sweep samplesheet: the cross product of the axes, one row per cell.

Stdlib only -- it runs on the login node before nextflow, outside any container.

The samplesheet is `cell_id,arch,labels,args`, where `args` is the literal flag
string handed to `cfs train-value` (or `cfs baseline-rf` for `arch=rf`). Keeping the
flags literal is deliberate: the axes are not a clean cross product (`--phi-hidden`
and `--k-code` are deepset-only, `--n-estimators`/`--delta` are rf-only, and the "5x
rows" arm is a different label root), and a cell's exact command stays greppable.

    ./make_sweep.py --labels bands=/path/to/20hm_bands/labels \\
                    --arch icnn --width 128,512 --depth 3,4 --epochs 1500 > sweep.csv
"""

from __future__ import annotations

import argparse
import itertools
import sys

# Flags that apply only to some archs. `cfs train-value` rejects unknown flags and
# `baseline-rf` has neither --arch nor --width, so a shared cross product has to be
# filtered per row rather than per axis.
_DEEPSET_ONLY = {"--phi-hidden", "--k-code", "--emb-dim"}
_GROUPMAX_ONLY = {"--gm-group", "--gm-temp", "--gm-init", "--gm-reanchor"}
_RF_ONLY = {"--n-estimators", "--delta"}


def _csv(v: str) -> list[str]:
    return [t.strip() for t in v.split(",") if t.strip()]


def _labels(v: str) -> tuple[str, str]:
    """``name=path`` -- the name goes in the cell id, the path in the samplesheet."""
    name, _, path = v.partition("=")
    if not path:
        raise argparse.ArgumentTypeError(f"expected name=path, got {v!r}")
    return name, path


def cells(
    labels,
    archs,
    *,
    width,
    depth,
    epochs,
    batch,
    lr,
    w_grad,
    eps,
    emb_dim,
    gm_group,
    gm_temp,
    gm_init,
    gm_reanchor,
    phi_hidden,
    k_code,
    n_estimators,
    delta,
    seed,
):
    """Yield ``(cell_id, arch, labels_path, args)`` for the cross product.

    Deduplicated by cell id: an axis an arch ignores (``--width`` for ``rf``) would
    otherwise produce one identical cell per value of it.
    """
    seen = set()
    axes = {
        "--width": width,
        "--depth": depth,
        "--epochs": epochs,
        "--batch": batch,
        "--lr": lr,
        "--w-grad": w_grad,
        "--eps": eps,
        "--emb-dim": emb_dim,
        "--gm-group": gm_group,
        "--gm-temp": gm_temp,
        "--gm-init": gm_init,
        "--gm-reanchor": gm_reanchor,
        "--phi-hidden": phi_hidden,
        "--k-code": k_code,
        "--n-estimators": n_estimators,
        "--delta": delta,
        "--seed": seed,
    }
    # Short cell-id tag per axis, in the `__k<v>` style of the legacy sweep's cells.
    tags = {
        "--width": "w",
        "--depth": "d",
        "--epochs": "e",
        "--batch": "b",
        "--lr": "lr",
        "--w-grad": "g",
        "--eps": "eps",
        "--emb-dim": "emb",
        "--gm-group": "grp",
        "--gm-temp": "T",
        "--gm-init": "init",
        "--gm-reanchor": "ra",
        "--phi-hidden": "ph",
        "--k-code": "kc",
        "--n-estimators": "t",
        "--delta": "dl",
        "--seed": "s",
    }
    names, values = list(axes), [axes[k] for k in axes]
    # Only *swept* axes go in the cell id -- otherwise every id carries the same
    # dozen fixed knobs and the leaderboard's key column is unreadable. The full
    # flag string stays in the samplesheet's `args` column either way.
    varying = {k for k, v in axes.items() if len(v) > 1} | {"--arch"}
    for (lname, lpath), arch, combo in itertools.product(labels, archs, itertools.product(*values)):
        keep = {}
        for flag, val in zip(names, combo, strict=True):
            if val == "":  # an axis left at the CLI's own default
                continue
            if flag in _RF_ONLY and arch != "rf":
                continue
            if flag in _DEEPSET_ONLY and not arch.startswith("deepset"):
                continue
            if flag in _GROUPMAX_ONLY and arch != "groupmax-u":
                continue
            if arch == "rf" and flag not in _RF_ONLY | {"--eps", "--seed"}:
                continue
            keep[flag] = val
        cell = "__".join([lname, arch] + [f"{tags[f]}{v}" for f, v in keep.items() if f in varying])
        if cell in seen:
            continue
        seen.add(cell)
        yield cell, arch, lpath, " ".join(f"{f} {v}" for f, v in keep.items())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--labels",
        type=_labels,
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Label root, repeatable. NAME lands in the cell id; the rows "
        "arm of the sweep is two of these (4000 vs 20000 media).",
    )
    p.add_argument(
        "--arch",
        type=_csv,
        default=["icnn"],
        help="icnn,deepset,deepset-private,mlp,rf (rf -> `cfs baseline-rf`).",
    )
    p.add_argument("--width", type=_csv, default=["128"])
    p.add_argument("--depth", type=_csv, default=["3"])
    p.add_argument("--epochs", type=_csv, default=["1500"])
    p.add_argument("--batch", type=_csv, default=["512"])
    p.add_argument("--lr", type=_csv, default=["3e-3"])
    p.add_argument("--w-grad", type=_csv, default=["1"])
    p.add_argument("--eps", type=_csv, default=["1e-3"])
    p.add_argument("--emb-dim", type=_csv, default=["8"], help="deepset only.")
    p.add_argument("--gm-group", type=_csv, default=[""], help="groupmax-u only; '' = default.")
    p.add_argument("--gm-temp", type=_csv, default=[""], help="groupmax-u only; '' = default.")
    p.add_argument(
        "--gm-reanchor",
        type=_csv,
        default=[""],
        help="groupmax-u only: re-anchor passes. Seeding fixes initialisation; "
        "planes still go dead during training and cannot revive.",
    )
    p.add_argument("--gm-init", type=_csv, default=[""], help="groupmax-u only: random,labels.")
    p.add_argument("--phi-hidden", type=_csv, default=[""], help="deepset only; '' = default.")
    p.add_argument("--k-code", type=_csv, default=[""], help="deepset only; '' = default.")
    p.add_argument("--n-estimators", type=_csv, default=["100"], help="rf only.")
    p.add_argument("--delta", type=_csv, default=["0.05"], help="rf only.")
    p.add_argument("--seed", type=_csv, default=["0"])
    p.add_argument("-o", "--out", default="-", help="Output CSV ('-' = stdout).")
    a = p.parse_args(argv)

    rows = list(
        cells(
            a.labels,
            a.arch,
            width=a.width,
            depth=a.depth,
            epochs=a.epochs,
            batch=a.batch,
            lr=a.lr,
            w_grad=a.w_grad,
            eps=a.eps,
            emb_dim=a.emb_dim,
            gm_group=a.gm_group,
            gm_temp=a.gm_temp,
            gm_init=a.gm_init,
            gm_reanchor=a.gm_reanchor,
            phi_hidden=a.phi_hidden,
            k_code=a.k_code,
            n_estimators=a.n_estimators,
            delta=a.delta,
            seed=a.seed,
        )
    )
    out = sys.stdout if a.out == "-" else open(a.out, "w")
    with out:
        print("cell_id,arch,labels,args", file=out)
        for cell, arch, lpath, args in rows:
            print(f"{cell},{arch},{lpath},{args}", file=out)
    print(f"{len(rows)} cells", file=sys.stderr)
    return 0


def demo() -> None:
    """Self-check: the cross product, and that arch-only flags stay on their arch."""
    rows = list(
        cells(
            [("bands", "/labels")],
            ["icnn", "rf"],
            width=["128", "512"],
            depth=["3"],
            epochs=["1500"],
            batch=["512"],
            lr=["3e-3"],
            w_grad=["1"],
            eps=["1e-3"],
            emb_dim=["8"],
            gm_group=[""],
            gm_temp=[""],
            gm_init=[""],
            phi_hidden=[""],
            k_code=[""],
            n_estimators=["100"],
            delta=["0.05"],
            seed=["0"],
        )
    )
    # icnn takes --width (2 values); rf ignores it, so its two combos dedupe to one.
    icnn = [r for r in rows if r[1] == "icnn"]
    rf = [r for r in rows if r[1] == "rf"]
    assert len(icnn) == 2, icnn
    assert len(rf) == 1, rf
    # Only --width varies, so only it is tagged.
    assert {r[0] for r in icnn} == {"bands__icnn__w128", "bands__icnn__w512"}, icnn
    assert "--width" not in rf[0][3] and "--n-estimators 100" in rf[0][3], rf[0]
    assert "--n-estimators" not in icnn[0][3], icnn[0]
    assert "--phi-hidden" not in icnn[0][3], icnn[0]
    # groupmax-only flags route the same way: onto groupmax-u rows and nowhere else.
    gm = list(
        cells(
            [("bands", "/l")],
            ["groupmax-u", "icnn-u"],
            width=["128"],
            depth=["3"],
            epochs=["10"],
            batch=["512"],
            lr=["3e-3"],
            w_grad=["10"],
            eps=["1e-3"],
            emb_dim=["8"],
            gm_group=["8"],
            gm_temp=["0.1"],
            gm_init=["labels"],
            phi_hidden=[""],
            k_code=[""],
            n_estimators=["100"],
            delta=["0.05"],
            seed=["0"],
        )
    )
    gmax = [r for r in gm if r[1] == "groupmax-u"][0]
    icnnu = [r for r in gm if r[1] == "icnn-u"][0]
    assert "--gm-group 8" in gmax[3] and "--gm-temp 0.1" in gmax[3], gmax
    assert "--gm-init labels" in gmax[3], gmax
    assert "--gm-group" not in icnnu[3] and "--gm-temp" not in icnnu[3], icnnu
    assert "--gm-init" not in icnnu[3], icnnu
    # deepset-only flags reach a deepset row; an empty axis value is dropped, which
    # is how a knob is left at the CLI's own default.
    ds = list(
        cells(
            [("bands", "/l")],
            ["deepset"],
            width=["128"],
            depth=["3"],
            epochs=["10"],
            batch=["512"],
            lr=["3e-3"],
            w_grad=["1"],
            eps=["1e-3"],
            emb_dim=["8"],
            gm_group=[""],
            gm_temp=[""],
            gm_init=[""],
            phi_hidden=["64"],
            k_code=["64"],
            n_estimators=["100"],
            delta=["0.05"],
            seed=["0"],
        )
    )
    assert "--phi-hidden 64 --k-code 64" in ds[0][3], ds[0]
    print("demo ok")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        raise SystemExit(main())
