import json

import pandas as pd
from scipy import stats


with open("variables.json", "r") as file:
    variables = json.load(file)



def compute_surgery_percentages(patients_df, side: str):
    """
    Compute % of surgery yes/no for the given side.

    For each side we look at:
      - that side's surgery indicator
      - and the global replacement indicator (either knee)
    """
    if side == "left":
        surg_cols = variables["VARIABLES"]["ALL_L"]["SURGERY"]

        cols =  surg_cols[0]
            
    elif side == "right":
        surg_cols = variables["VARIABLES"]["ALL_R"]["SURGERY"]

        cols = surg_cols[0]
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

    if side == "left":

        kl_cols = variables["VARIABLES"]["ALL_L"]["KL_GRADE"]
        col = kl_cols[0]  

    elif side == "right":

        kl_cols = variables["VARIABLES"]["ALL_R"]["KL_GRADE"]
        col = kl_cols[0]  

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

    if side == 'left':
        lr_group = variables["VARIABLES"]["ALL_L"]
    elif side == 'right':
        lr_group = variables["VARIABLES"]["ALL_R"]
    else:
        raise ValueError(f"Unknown side: {side}")

    result = {}

    # Loop through only the subgroups of SYMPTOMS and STRUCTURE
    for subgroup, cols in lr_group.items():
        if subgroup not in ["SYMPTOMS", "STRUCTURE"]:
            continue
        subgroup_stats = {}

        valid_cols = [c for c in cols if c in patients_df.columns]

        for col in valid_cols:
            subgroup_stats[col] = patients_df[col].mean()

        result[subgroup] = subgroup_stats

    return result

    