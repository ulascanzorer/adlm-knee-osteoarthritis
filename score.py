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
