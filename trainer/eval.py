#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from rdkit import Chem
from rdkit import RDLogger
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")


def xyz_to_sdf(xyz: Path, sdf: Path, obabel: str = "obabel") -> bool:
    try:
        subprocess.run(
            [obabel, str(xyz), "-O", str(sdf)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return sdf.is_file() and sdf.stat().st_size > 0
    except Exception:
        return False


def is_valid_sdf(sdf: Path):
    try:
        mol = Chem.SDMolSupplier(str(sdf), removeHs=False)[0]
        if mol is None:
            return False, None
        smi = Chem.MolToSmiles(mol)
        if "." in smi:
            return False, None
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True, smi
    except Exception:
        return False, None


def evaluate(samples_dir: Path, n_samples: int = 100, obabel: str = "obabel") -> dict:
    mol_dirs = sorted([d for d in samples_dir.iterdir() if d.is_dir()], key=lambda p: p.name)
    valid_cnt = unique_cnt = total_cnt = recovery_cnt = 0
    invalid_true = 0

    for mol_dir in tqdm(mol_dirs, desc="eval"):
        true_xyz = mol_dir / "true_.xyz"
        true_sdf = mol_dir / "true_.sdf"
        if not true_xyz.is_file():
            continue
        if not true_sdf.is_file():
            xyz_to_sdf(true_xyz, true_sdf, obabel=obabel)
        true_mol = Chem.SDMolSupplier(str(true_sdf), removeHs=False)[0] if true_sdf.is_file() else None
        if true_mol is None:
            invalid_true += 1
            continue
        true_smi = Chem.MolToSmiles(true_mol)

        smi_group = []
        for i in range(n_samples):
            xyz = mol_dir / f"{i}_.xyz"
            sdf = mol_dir / f"{i}_.sdf"
            if not xyz.is_file():
                continue
            total_cnt += 1
            if not sdf.is_file():
                xyz_to_sdf(xyz, sdf, obabel=obabel)
            ok, smi = is_valid_sdf(sdf)
            if ok and smi is not None:
                valid_cnt += 1
                smi_group.append(smi)

        if true_smi in smi_group:
            recovery_cnt += 1
        unique_cnt += len(set(smi_group))

    n_mols = max(1, len(mol_dirs) - invalid_true)
    return {
        "samples_dir": str(samples_dir),
        "n_molecules": len(mol_dirs),
        "n_samples_per_mol": n_samples,
        "validity_pct": valid_cnt / max(1, total_cnt) * 100,
        "uniqueness_pct": unique_cnt / max(1, total_cnt) * 100,
        "recovery_pct": recovery_cnt / n_mols * 100,
        "valid_cnt": valid_cnt,
        "total_cnt": total_cnt,
        "recovery_cnt": recovery_cnt,
        "invalid_true": invalid_true,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples_dir", type=Path, required=True)
    p.add_argument("--n_samples", type=int, default=100)
    p.add_argument("--out_json", type=Path, required=True)
    p.add_argument("--obabel", type=str, default="obabel")
    args = p.parse_args()

    metrics = evaluate(args.samples_dir, n_samples=args.n_samples, obabel=args.obabel)
    print(f"samples_dir: {metrics['samples_dir']}")
    print(f"Validity:   {metrics['validity_pct']:.2f}%")
    print(f"Uniqueness: {metrics['uniqueness_pct']:.2f}%")
    print(f"Recovery:   {metrics['recovery_pct']:.2f}%")
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(metrics, indent=2))
    print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
