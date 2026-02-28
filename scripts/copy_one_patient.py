#!/usr/bin/env python3
"""
Copy MRI (SAG_3D_DESS) + specific X-ray data for a single patient into a
cleaned dataset folder, organized by cohort and side.

Output structure (DESTINATION):
    cleaned_data_baseline/
        <cohort>/                # e.g. 0.E.1, 0.C.2
            <patient_id>/        # src_subject_id or subjectkey
                mri/
                    left/<file>.tar.gz
                    right/<file>.tar.gz
                    bilateral/<file>.tar.gz
                xray/
                    <file>.tar.gz

X-ray filter: only rows where
    scan_type      == "X-Ray"
    comments_misc contains "OAI XRAY SCREENING KNEE"
MRI filter: only rows where
    image_modality == "MRI"
    image_description contains "SAG_3D_DESS"

No metadata CSVs are written, only files are copied.
"""

from pathlib import Path
import argparse
import shutil

import pandas as pd


# ---------- helpers ----------

def infer_side(desc: str) -> str:
    """Infer left/right/bilateral from the image_description."""
    d = str(desc).upper()
    if "LEFT" in d and "RIGHT" in d:
        return "bilateral"
    if "LEFT" in d:
        return "left"
    if "RIGHT" in d:
        return "right"
    return "unknown"


def load_tabular(path: Path) -> pd.DataFrame:
    """Load a TAB-separated NDA-style txt file."""
    return pd.read_csv(path, sep="\t", low_memory=False)


def extract_cohort_and_rest(image_file: str):
    """
    From an S3-style path like:
        s3:/.../00m/0.E.1/9363408/data/20060810/11304209.tar.gz
    or
        s3:/.../00m/0.E.1/9363408/20060810/11304209.tar.gz

    return:
        cohort="0.E.1",
        rest=Path("data/20060810/11304209.tar.gz")  or Path("20060810/11304209.tar.gz")

    i.e. we:
      - take the part after "00m/"
      - treat first as cohort, second as subject_id
      - keep the rest exactly for the SOURCE path (we'll only use the filename for DEST)
    """
    s = image_file.strip()
    if "00m/" not in s:
        return None, None

    after = s.split("00m/", 1)[1]            # e.g. '0.E.1/9363408/data/20060810/1130.tar.gz'
    parts = after.split("/")
    if len(parts) < 3:
        return None, None

    cohort = parts[0]                        # '0.E.1'
    # parts[1] should be subject_id
    remainder = parts[2:]                    # ['data', '20060810', '1130.tar.gz'] or ['20060810', '1130.tar.gz']

    if not remainder:
        return None, None

    rest = Path(*remainder)                  # 'data/20060810/1130.tar.gz' or '20060810/1130.tar.gz'
    return cohort, rest


# ---------- MRI (SAG_3D_DESS) ----------

def copy_patient_mri(
    subject_id: str,
    baseline_dir: Path,
    output_root: Path,
    meta_filename: str = "image03.txt",
) -> int:
    """
    Copy all SAG_3D_DESS MRI files for one patient into
    cleaned_data_baseline/<cohort>/<patient_id>/mri/<side>/<file>.tar.gz

    Returns: number of files actually copied.
    """
    meta_path = baseline_dir / meta_filename
    if not meta_path.is_file():
        raise FileNotFoundError(f"MRI metadata file not found: {meta_path}")

    print(f"[MRI] Loading MRI metadata from {meta_path}")
    df = load_tabular(meta_path)
    print(f"[MRI] MRI metadata shape: {df.shape}")

    cols = set(df.columns)
    has_subjectkey = "subjectkey" in cols
    has_src_id = "src_subject_id" in cols

    if not has_subjectkey and not has_src_id:
        raise ValueError(
            "MRI metadata must contain 'subjectkey' or 'src_subject_id'. "
            f"Columns: {df.columns.tolist()}"
        )

    subj_str = str(subject_id)
    mask = False
    if has_subjectkey:
        mask = (df["subjectkey"].astype(str) == subj_str)
    if has_src_id:
        mask = mask | (df["src_subject_id"].astype(str) == subj_str)

    df_subj = df[mask].copy()
    if df_subj.empty:
        print(f"[MRI] No MRI/X-ray rows for subject {subject_id} in image03.txt")
        return 0

    print(f"[MRI] Found {len(df_subj)} rows for subject {subject_id} (all imaging types)")

    # Need these columns
    for col in ["image_modality", "image_description", "image_file"]:
        if col not in df_subj.columns:
            raise ValueError(f"MRI metadata missing '{col}' column.")

    # Keep only MRI + SAG_3D_DESS
    df_subj = df_subj[df_subj["image_modality"] == "MRI"].copy()
    df_subj = df_subj[
        df_subj["image_description"].astype(str).str.contains("SAG_3D_DESS", na=False)
    ].copy()

    if df_subj.empty:
        print(f"[MRI] Subject {subject_id} has no SAG_3D_DESS MRI rows")
        return 0

    print(f"[MRI] After SAG_3D_DESS filter: {len(df_subj)} rows")

    # Add side and cohort
    df_subj["side"] = df_subj["image_description"].map(infer_side)
    df_subj["cohort"] = df_subj["image_file"].astype(str).apply(
        lambda s: extract_cohort_and_rest(s)[0]
    )

    copied = 0

    for _, row in df_subj.iterrows():
        image_file = row["image_file"]
        side = row["side"] or "unknown"
        cohort, rest = extract_cohort_and_rest(image_file)

        if cohort is None or rest is None:
            print(f"[MRI][WARN] Could not parse cohort/rest from image_file: {image_file}")
            continue

        # local SOURCE path: baseline/image03/00m/<cohort>/<subject_id>/<rest>
        rel_src = Path("image03") / "00m" / cohort / str(subject_id) / rest
        src = baseline_dir / rel_src

        # DEST: cleaned_data_baseline/<cohort>/<patient>/mri/<side>/<filename>
        filename = rest.name
        patient_root = output_root / cohort / str(subject_id)
        dst = patient_root / "mri" / side / filename

        if not src.exists():
            print(f"[MRI][WARN] Source MRI file not found: {src}")
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
        print(f"[MRI][COPY] {src} -> {dst}")

    print(f"[MRI] Copied {copied} MRI files for subject {subject_id}")
    return copied


# ---------- X-ray from image03.txt (OAI XRAY SCREENING KNEE only) ----------

def copy_patient_xray_from_image03(
    subject_id: str,
    baseline_dir: Path,
    output_root: Path,
    meta_filename: str = "image03.txt",
) -> int:
    """
    Copy all X-ray files for one patient into
    cleaned_data_baseline/<cohort>/<patient_id>/xray/<file>.tar.gz

    Uses image03.txt and filters rows for this subject with:
        scan_type == "X-Ray"
        AND comments_misc contains "OAI XRAY SCREENING KNEE"
    """
    meta_path = baseline_dir / meta_filename
    if not meta_path.is_file():
        print(f"[XRAY] Metadata file not found: {meta_path} (skipping X-ray)")
        return 0

    print(f"[XRAY] Loading imaging metadata from {meta_path}")
    df = load_tabular(meta_path)
    print(f"[XRAY] Metadata shape: {df.shape}")

    cols = set(df.columns)
    has_subjectkey = "subjectkey" in cols
    has_src_id = "src_subject_id" in cols

    if not has_subjectkey and not has_src_id:
        raise ValueError(
            "Metadata must contain 'subjectkey' or 'src_subject_id'. "
            f"Columns: {df.columns.tolist()}"
        )

    subj_str = str(subject_id)
    mask = False
    if has_subjectkey:
        mask = (df["subjectkey"].astype(str) == subj_str)
    if has_src_id:
        mask = mask | (df["src_subject_id"].astype(str) == subj_str)

    df_subj = df[mask].copy()
    if df_subj.empty:
        print(f"[XRAY] No rows for subject {subject_id} in image03.txt")
        return 0

    print(f"[XRAY] Found {len(df_subj)} rows for subject {subject_id} (all imaging types)")

    # Require these columns
    for col in ["scan_type", "image_file", "comments_misc"]:
        if col not in df_subj.columns:
            raise ValueError(f"X-ray metadata missing '{col}' column.")

    # Filter to X-Ray + OAI XRAY SCREENING KNEE
    df_xray = df_subj[
        (df_subj["scan_type"] == "X-Ray")
        & (df_subj["comments_misc"].astype(str).str.contains("OAI XRAY SCREENING KNEE", na=False))
    ].copy()

    if df_xray.empty:
        print(f"[XRAY] Subject {subject_id} has no X-Ray rows with 'OAI XRAY SCREENING KNEE'")
        return 0

    print(f"[XRAY] After filters (X-Ray + OAI XRAY SCREENING KNEE): {len(df_xray)} rows")

    # Add cohort
    df_xray["cohort"] = df_xray["image_file"].astype(str).apply(
        lambda s: extract_cohort_and_rest(s)[0]
    )

    copied = 0

    for _, row in df_xray.iterrows():
        image_file = row["image_file"]
        cohort, rest = extract_cohort_and_rest(image_file)

        if cohort is None or rest is None:
            print(f"[XRAY][WARN] Could not parse cohort/rest from image_file: {image_file}")
            continue

        # SOURCE path: baseline/image03/00m/<cohort>/<subject_id>/<rest>
        rel_src = Path("image03") / "00m" / cohort / str(subject_id) / rest
        src = baseline_dir / rel_src

        # DEST: cleaned_data_baseline/<cohort>/<patient>/xray/<filename>
        filename = rest.name
        patient_root = output_root / cohort / str(subject_id)
        dst = patient_root / "xray" / filename

        if not src.exists():
            print(f"[XRAY][WARN] Source X-ray file not found: {src}")
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
        print(f"[XRAY][COPY] {src} -> {dst}")

    print(f"[XRAY] Copied {copied} X-ray files for subject {subject_id}")
    return copied


# ---------- main wrapper ----------

def copy_patient_all(
    subject_id: str,
    data_root: Path,
    baseline_name: str = "baseline",
    output_root_name: str = "cleaned_data_baseline",
    meta_filename: str = "image03.txt",
) -> None:
    """Copy both MRI + filtered X-ray for one patient into cohort/patient folders."""
    baseline_dir = data_root / baseline_name
    if not baseline_dir.is_dir():
        raise FileNotFoundError(f"Baseline folder not found: {baseline_dir}")

    output_root = data_root / output_root_name
    output_root.mkdir(parents=True, exist_ok=True)

    subj_str = str(subject_id)
    print(f"[INFO] Output root: {output_root}, subject: {subj_str}")

    mri_copied = copy_patient_mri(
        subject_id=subj_str,
        baseline_dir=baseline_dir,
        output_root=output_root,
        meta_filename=meta_filename,
    )
    xray_copied = copy_patient_xray_from_image03(
        subject_id=subj_str,
        baseline_dir=baseline_dir,
        output_root=output_root,
        meta_filename=meta_filename,
    )

    print(
        f"[INFO] Done for subject {subject_id}: "
        f"{mri_copied} MRI files, {xray_copied} X-ray files copied."
    )


# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser(
        description="Copy MRI (SAG_3D_DESS) and 'OAI XRAY SCREENING KNEE' X-rays for one patient."
    )
    parser.add_argument(
        "subject_id",
        help="Subject identifier (src_subject_id or subjectkey, e.g. 9363408 or NDAR_INV...).",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=".",
        help="Directory that contains 'baseline_images/' (default: current directory).",
    )
    parser.add_argument(
        "--baseline-name",
        type=str,
        default="baseline_images",
        help="Name of the baseline folder (default: 'baseline_images').",
    )
    parser.add_argument(
        "--output-root-name",
        type=str,
        default="cleaned_images_baseline",
        help="Name of output root folder (default: 'cleaned_images_baseline').",
    )
    parser.add_argument(
        "--meta-filename",
        type=str,
        default="image03.txt",
        help="Metadata filename inside baseline_images (default: 'image03.txt').",
    )

    args = parser.parse_args()
    data_root = Path(args.data_root).resolve()

    copy_patient_all(
        subject_id=args.subject_id,
        data_root=data_root,
        baseline_name=args.baseline_name,
        output_root_name=args.output_root_name,
        meta_filename=args.meta_filename,
    )


if __name__ == "__main__":
    main()
