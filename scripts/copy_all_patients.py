#!/usr/bin/env python3
"""
Run copy_patient_all() for all patients that have either
- SAG_3D_DESS MRI, or
- X-Ray with comments_misc containing "OAI XRAY SCREENING KNEE".

Uses copy_patient_all() from copy_one_patient.py and the same
folder assumptions as that script.
"""

from pathlib import Path

import pandas as pd

from copy_one_patient import copy_patient_all


def get_all_relevant_subject_ids(image03_path: Path) -> list[str]:
    """Return sorted list of subject IDs (as strings) worth processing."""
    df = pd.read_csv(image03_path, sep="\t", low_memory=False)

    # Work on string version of src_subject_id, drop NaNs
    if "src_subject_id" not in df.columns:
        raise ValueError("Expected 'src_subject_id' column in image03.txt")

    df = df[df["src_subject_id"].notna()].copy()
    df["src_subject_id"] = df["src_subject_id"].astype(str)

    # Conditions for MRI DESS
    mri_mask = (
        (df.get("image_modality") == "MRI")
        & df.get("image_description", "")
             .astype(str)
             .str.contains("SAG_3D_DESS", na=False)
    )

    # Conditions for X-Ray OAI XRAY SCREENING KNEE
    xray_mask = (
        (df.get("scan_type") == "X-Ray")
        & df.get("comments_misc", "")
             .astype(str)
             .str.contains("OAI XRAY SCREENING KNEE", na=False)
    )

    mask = mri_mask | xray_mask
    df_sel = df[mask]

    subject_ids = sorted(df_sel["src_subject_id"].unique())
    return subject_ids


def main():
    data_root = Path(".").resolve()
    baseline_dir = data_root / "baseline"
    image03_path = baseline_dir / "image03.txt"

    if not image03_path.is_file():
        raise FileNotFoundError(f"image03.txt not found at {image03_path}")

    print(f"[ALL] Reading subject list from {image03_path}")
    subject_ids = get_all_relevant_subject_ids(image03_path)
    print(f"[ALL] Found {len(subject_ids)} subjects with relevant MRI/X-ray")


    for i, sid in enumerate(subject_ids, start=1):
        print(f"\n[ALL] ({i}/{len(subject_ids)}) Processing subject {sid}")
        copy_patient_all(
            subject_id=sid,
            data_root=data_root,
            baseline_name="baseline",
            output_root_name="cleaned_data_baseline",
            meta_filename="image03.txt",
        )


if __name__ == "__main__":
    main()
