import os
import pandas as pd
from util import get_image  # make sure util.py is importable
import sys
CLINICAL_CSV = "csv/clinical00_cleaned.csv"
CLUSTERS_CSV = "results/tabular_input_ae/csv/mri_clusters_left.csv"
OUT_CSV = "results/tabular_input_ae/csv/selected_patients_left_by_cluster_and_kl.csv"

ID_COL = "ID"
KL_COL = "V00XRKL_L"
CLUSTER_COL = "cluster"

GRADES = [0, 1, 2, 3, 4]
N_PER_GRADE = 2
SEED = 42


def coerce_id(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def create_csv():
    clinical = pd.read_csv(CLINICAL_CSV, sep=",", skipinitialspace=True, engine="python")
    clinical.columns = clinical.columns.astype(str).str.strip()

    clusters = pd.read_csv(CLUSTERS_CSV)
    clusters.columns = clusters.columns.astype(str).str.strip()

    for col in [ID_COL, KL_COL]:
        if col not in clinical.columns:
            raise ValueError(f"'{col}' not found in {CLINICAL_CSV}. Columns:\n{list(clinical.columns)}")
    for col in [ID_COL, CLUSTER_COL]:
        if col not in clusters.columns:
            raise ValueError(f"'{col}' not found in {CLUSTERS_CSV}. Columns:\n{list(clusters.columns)}")

    clinical = clinical[[ID_COL, KL_COL]].copy()
    clusters = clusters[[ID_COL, CLUSTER_COL]].copy()

    clinical[ID_COL] = coerce_id(clinical[ID_COL])
    clusters[ID_COL] = coerce_id(clusters[ID_COL])

    clinical[KL_COL] = pd.to_numeric(clinical[KL_COL], errors="coerce")
    clusters[CLUSTER_COL] = pd.to_numeric(clusters[CLUSTER_COL], errors="coerce").astype("Int64")

    merged = pd.merge(clinical, clusters, on=ID_COL, how="inner").dropna(subset=[ID_COL, KL_COL, CLUSTER_COL])
    cluster_values = sorted(merged[CLUSTER_COL].unique().tolist())

    rows = []
    warnings = []

    for cl in cluster_values:
        df_cl = merged[merged[CLUSTER_COL] == cl]
        for grade in GRADES:
            eligible = df_cl[df_cl[KL_COL] == grade][ID_COL].dropna().astype(int).unique().tolist()

            if len(eligible) == 0:
                warnings.append(f"[WARN] cluster={int(cl)} KL={grade}: 0 eligible.")
                chosen = []
            elif len(eligible) < N_PER_GRADE:
                warnings.append(f"[WARN] cluster={int(cl)} KL={grade}: only {len(eligible)} (<{N_PER_GRADE}); taking all.")
                chosen = eligible
            else:
                chosen = pd.Series(eligible).sample(n=N_PER_GRADE, random_state=SEED, replace=False).tolist()

            for pid in chosen:
                rows.append({"cluster": int(cl), "KL_left": int(grade), "ID": int(pid)})

    out = pd.DataFrame(rows).sort_values(["cluster", "KL_left", "ID"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"Saved: {OUT_CSV}")

    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(w)

    return out

def collect_images(df):
    DATASET_ROOT = "/vol/miltank/projects/practical_wise2526/knee-osteoarthritis-severity/data/cleaned_images_baseline"
    OUTPUT_ROOT = "/vol/miltank/users/foca/adlm-knee-osteoarthritis/results/MRIS"
    SIDE = "left"

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    saved = 0
    missing = []

    # iterate actual clusters present
    clusters = sorted(df["cluster"].dropna().unique().tolist())

    for cl in clusters:
        df_cl = df[df["cluster"] == cl]

        # iterate grades present in that cluster
        grades = sorted(df_cl["KL_left"].dropna().unique().tolist())

        for kl in grades:
            df_ck = df_cl[df_cl["KL_left"] == kl]
            ids = df_ck["ID"].dropna().astype(int).unique().tolist()

            out_dir = os.path.join(OUTPUT_ROOT, f"cluster_{int(cl)}", f"KL_{int(kl)}")
            os.makedirs(out_dir, exist_ok=True)

            print(f"[INFO] cluster {int(cl)} KL {int(kl)}: {len(ids)} IDs")

            for patient_id in ids:
                res = get_image(DATASET_ROOT, patient_id, side=SIDE)

                # supports util.get_image returning either img OR (pid,img)
                if isinstance(res, tuple) and len(res) == 2:
                    pid, img = res
                else:
                    pid, img = patient_id, res

                if img is None:
                    missing.append((int(cl), int(kl), int(patient_id)))
                    continue

                out_path = os.path.join(
                    out_dir,
                    f"{int(pid)}_{SIDE}_cluster{int(cl)}_KL{int(kl)}.png"
                )
                img.save(out_path)
                saved += 1

    print(f"\nDone. Saved {saved} images to {OUTPUT_ROOT}")
    if missing:
        print(f"Missing/failed for {len(missing)} patients. First 20:")
        print(missing[:20])

    return saved, missing


def main():
    df = create_csv()
    collect_images(df)


if __name__ == "__main__":
    main()
