import numpy as np
import pandas as pd
from scipy import stats
import json


# Load the JSON file
with open('variables.json', 'r') as file:
	variables = json.load(file)


def compute_symptoms_score(patients_df, variables=variables):
    """
    Compute the mean Symptoms score for a set of patients in a cluster.
    Inputs:
        patients_df: DataFrame containing patient data
        variables: dict containing 'WOMAC' and 'KOOS' lists
    Returns:
        float (cluster mean of symptoms, higher = worse)
    """
    cols = variables["VARIABLES"]["WOMAC"] + variables["VARIABLES"]["KOOS"]
    vals = patients_df[cols].mean(axis=1)  # patient-level means
    cluster_score = 100 - vals.mean()      # invert and average
    return cluster_score



def compute_structure_score(patients_df, variables=variables):
    """
    Compute the dominant (mode) Structure score (0–2) for the cluster.
    Inputs:
        patients_df: DataFrame
        variables: dict containing 'JOINT_SPACE_NARROWING' and 'OSTEOPHYTES'
    Returns:
        int (0, 1, or 2)
    """
    cols = variables["VARIABLES"]["JOINT_SPACE_NARROWING"] + variables["VARIABLES"]["OSTEOPHYTES"]
    # Flatten all values for all patients and variables
    flattened = patients_df[cols].to_numpy().flatten()
    mode_val = stats.mode(flattened, keepdims=True).mode[0]
    return int(mode_val)


def compute_surgery_score(patients_df, variables=variables):
    """
    Compute the dominant (mode) Surgery score (0 or 1) for the cluster.
    Inputs:
        patients_df: DataFrame
        variables: dict containing 'SURGERY' list
    Returns:
        int (0 or 1)
    """
    cols = variables["VARIABLES"]["SURGERY"]
    flattened = patients_df[cols].to_numpy().flatten()
    mode_val = stats.mode(flattened, keepdims=True).mode[0]
    return int(mode_val)

def compute_surgery_percentages(patients_df, variables=variables):
    """
    Compute percentage of 0/1 surgery values in the cluster.

    This looks at all surgery columns in VARIABLES["SURGERY"],
    flattens them, and computes:
        % of entries == 1 (yes)
        % of entries == 0 (no)
    """
    cols = variables["VARIABLES"]["SURGERY"]
    vals = patients_df[cols].to_numpy().flatten()

    s = pd.Series(vals)
    counts = s.value_counts(dropna=True)

    n_yes = counts.get(1, 0)
    n_no  = counts.get(0, 0)
    total = n_yes + n_no

    if total == 0:
        return 0.0, 0.0  

    pct_yes = n_yes / total * 100.0
    pct_no  = n_no  / total * 100.0
    return pct_yes, pct_no

def compute_kl_left_distribution(patients_df, variables=variables):
    """
    Return counts of KL grades 0-4 for the left knee in this cluster.
    """
    left_col = variables["VARIABLES"]["KL_GRADE"][0]
    vc = patients_df[left_col].value_counts(dropna=True)
    return vc.reindex([0, 1, 2, 3, 4], fill_value=0).astype(int)


def compute_kl_right_distribution(patients_df, variables=variables):
    """
    Return counts of KL grades 0-4 for the right knee in this cluster.
    """
    right_col = variables["VARIABLES"]["KL_GRADE"][1]
    vc = patients_df[right_col].value_counts(dropna=True)
    return vc.reindex([0, 1, 2, 3, 4], fill_value=0).astype(int)