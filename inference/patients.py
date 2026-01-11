import pandas as pd

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


def main():
    # Clinical: handle possible spaces after commas + messy headers
    clinical = pd.read_csv(CLINICAL_CSV, sep=",", skipinitialspace=True, engine="python")
    clinical.columns = clinical.columns.astype(str).str.strip()

    # Clusters: clean standard csv
    clusters = pd.read_csv(CLUSTERS_CSV)
    clusters.columns = clusters.columns.astype(str).str.strip()

    # Check required columns
    for col in [ID_COL, KL_COL]:
        if col not in clinical.columns:
            raise ValueError(
                f"'{col}' not found in {CLINICAL_CSV}. Columns are:\n{list(clinical.columns)}"
            )
    for col in [ID_COL, CLUSTER_COL]:
        if col not in clusters.columns:
            raise ValueError(
                f"'{col}' not found in {CLUSTERS_CSV}. Columns are:\n{list(clusters.columns)}"
            )

    # Normalize IDs + types
    clinical = clinical[[ID_COL, KL_COL]].copy()
    clusters = clusters[[ID_COL, CLUSTER_COL]].copy()

    clinical[ID_COL] = coerce_id(clinical[ID_COL])
    clusters[ID_COL] = coerce_id(clusters[ID_COL])

    clinical[KL_COL] = pd.to_numeric(clinical[KL_COL], errors="coerce")
    clusters[CLUSTER_COL] = pd.to_numeric(clusters[CLUSTER_COL], errors="coerce").astype("Int64")

    # Merge to keep only IDs present in both
    merged = pd.merge(clinical, clusters, on=ID_COL, how="inner")
    merged = merged.dropna(subset=[ID_COL, KL_COL, CLUSTER_COL])

    cluster_values = sorted(merged[CLUSTER_COL].unique().tolist())

    rows = []
    warnings = []

    for cl in cluster_values:
        df_cl = merged[merged[CLUSTER_COL] == cl]
        for grade in GRADES:
            eligible = df_cl[df_cl[KL_COL] == grade][ID_COL].dropna().astype(int).unique().tolist()

            if len(eligible) == 0:
                warnings.append(f"[WARN] cluster={int(cl)} KL={grade}: 0 eligible patients.")
                chosen = []
            elif len(eligible) < N_PER_GRADE:
                warnings.append(
                    f"[WARN] cluster={int(cl)} KL={grade}: only {len(eligible)} eligible (<{N_PER_GRADE}); taking all."
                )
                chosen = eligible
            else:
                chosen = (
                    pd.Series(eligible)
                    .sample(n=N_PER_GRADE, random_state=SEED, replace=False)
                    .tolist()
                )

            for pid in chosen:
                rows.append({"cluster": int(cl), "KL_left": int(grade), "ID": int(pid)})

    out = pd.DataFrame(rows).sort_values(["cluster", "KL_left", "ID"]).reset_index(drop=True)
    out.to_csv(OUT_CSV, index=False)

    print(f"Saved: {OUT_CSV}\n")

    if out.empty:
        print("No selections made (output is empty).")
    else:
        counts = out.groupby(["cluster", "KL_left"])["ID"].count().reset_index(name="n_selected")
        print("Counts per (cluster, KL_left):")
        print(counts.to_string(index=False))

        expected = len(GRADES) * N_PER_GRADE
        per_cluster = out.groupby("cluster")["ID"].count().to_dict()
        for cl in cluster_values:
            got = per_cluster.get(int(cl), 0)
            if got != expected:
                print(f"\n[NOTE] cluster={int(cl)}: selected {got}/{expected} IDs (some grades missing/short).")

    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(w)


if __name__ == "__main__":
    main()
