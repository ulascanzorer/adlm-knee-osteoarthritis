import json

import pandas as pd
from scipy import stats


# Load the JSON file with variable definitions
with open("variables.json", "r") as file:
    variables = json.load(file)


def compute_symptoms_score(patients_df, side: str):
    """
    Compute the mean Symptoms score for a set of patients in a cluster,
    for a given side ("left" or "right").
    """
    if side == "left":
        cols = [
            "V00WOMTSL",
            "V00KOOSKPL",
            "V00KOOSYML",
            "V00KOOSQOL",
        ]
    elif side == "right":
        cols = [
            "V00WOMTSR",
            "V00KOOSKPR",
            "V00KOOSYMR",
            "V00KOOSQOL",
        ]
    else:
        raise ValueError(f"Unknown side: {side}")

    vals = patients_df[cols].mean(axis=1)  # patient-level means
    cluster_score = 100 - vals.mean()      # invert and average (higher = worse)
    return cluster_score


def compute_structure_score(patients_df, side: str):
    """
    Compute the mean structural severity (0-2) for the cluster,
    using JSN + osteophyte variables for the given side.
    """
    jsn_cols = variables["VARIABLES"]["JOINT_SPACE_NARROWING"]
    ost_cols = variables["VARIABLES"]["OSTEOPHYTES"]

    if side == "left":
        cols = [
            jsn_cols[0],  # P01SVLKJSL
            jsn_cols[1],  # P01SVLKJSM
            ost_cols[0],  # P01SVLKOST
        ]
    elif side == "right":
        cols = [
            jsn_cols[2],  # P01SVRKJSL
            jsn_cols[3],  # P01SVRKJSM
            ost_cols[1],  # P01SVRKOST
        ]
    else:
        raise ValueError(f"Unknown side: {side}")

    # Average over variables and patients → scalar in [0, 2]
    return float(patients_df[cols].mean().mean())


def compute_surgery_percentages(patients_df, side: str):
    """
    Compute % of surgery yes/no for the given side.

    For each side we look at:
      - that side's surgery indicator
      - and the global replacement indicator (either knee)
    """
    surg_cols = variables["VARIABLES"]["SURGERY"]
    if side == "left":
        cols = [
            surg_cols[0],  # P01KSURGL
            surg_cols[2],  # P02KSURGCV
        ]
    elif side == "right":
        cols = [
            surg_cols[1],  # P01KSURGR
            surg_cols[2],  # P02KSURGCV
        ]
    else:
        raise ValueError(f"Unknown side: {side}")

    vals = patients_df[cols].to_numpy().flatten()

    s = pd.Series(vals)
    counts = s.value_counts(dropna=True)

    n_yes = counts.get(1, 0)
    n_no = counts.get(0, 0)
    total = n_yes + n_no

    if total == 0:
        return 0.0, 0.0

    pct_yes = n_yes / total * 100.0
    pct_no = n_no / total * 100.0
    return pct_yes, pct_no


def compute_kl_distribution(patients_df, side: str):
    """
    Return counts of KL grades 0-4 for the given knee side in this cluster.
    side: "left" or "right"
    """
    kl_cols = variables["VARIABLES"]["KL_GRADE"]

    if side == "left":
        col = kl_cols[0]  # P01OAGRDL
    elif side == "right":
        col = kl_cols[1]  # P01OAGRDR
    else:
        raise ValueError(f"Unknown side: {side}")

    vc = patients_df[col].value_counts(dropna=True)
    return vc.reindex([0, 1, 2, 3, 4], fill_value=0).astype(int)


def compute_all_statistics(patients_df, side: str):
    """
    Compute statistics for each subgroup (SYMPTOMS, STRUCTURE, SURGERY, KL_GRADE)
    inside ALL_L or ALL_R depending on the knee side.
    Returns a dictionary: subgroup -> {variable: score}
    """

    # Select correct group
    if side == 'left':
        lr_group = variables["VARIABLES"]["ALL_L"]
    elif side == 'right':
        lr_group = variables["VARIABLES"]["ALL_R"]
    else:
        raise ValueError(f"Unknown side: {side}")

    result = {}

    # Loop through subgroups
    for subgroup, cols in lr_group.items():
        subgroup_stats = {}

        # Only use valid columns present in dataframe
        valid_cols = [c for c in cols if c in patients_df.columns]

        for col in valid_cols:
            # All scoring logic is mean() as discussed
            subgroup_stats[col] = patients_df[col].mean()

        result[subgroup] = subgroup_stats

    return result

    