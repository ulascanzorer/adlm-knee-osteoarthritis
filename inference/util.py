import pandas as pd
import random
from collections import defaultdict
import os
import tempfile
import shutil
import tarfile
import pydicom
import numpy as np
from PIL import Image
import json
from score import compute_surgery_percentages, compute_all_statistics

# Load the JSON file with variable definitions
with open("variables.json", "r") as file:
    variables = json.load(file)


# SAMPLE PATIENTS PER CLUSTER
def extract_random_patients(df, num_clusters, patients_per_cluster=5):

    d = defaultdict(list)

    for cl in range(num_clusters):
        cluster_patients = df[df["cluster"] == cl]["ID"].tolist()

        if len(cluster_patients) == 0:
            print(f"⚠ Cluster {cl} contains no patients.")
            continue

        sample_size = min(len(cluster_patients), patients_per_cluster)
        d[cl] = random.sample(cluster_patients, sample_size)

    return dict(d)


def stack_slices_grid(pil_slices, grid_cols=12):
    """
    Takes a list of PIL slices and arranges them into a grid.
    """
    num_slices = len(pil_slices)
    cols = grid_cols
    rows = int(np.ceil(num_slices / cols))

    w, h = pil_slices[0].size
    grid = Image.new("L", (cols * w, rows * h))

    for i, img in enumerate(pil_slices):
        r = i // cols
        c = i % cols
        grid.paste(img, (c * w, r * h))

    return grid

# GET STACKED PNG PER PATIENT
def get_image(dataset_root, patient_id, side="left"):

    subset_dirs = [
        d for d in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, d))
    ]

    for subset in subset_dirs:

        subset_path = os.path.join(dataset_root, subset)
        patient_path = os.path.join(subset_path, str(patient_id))
        mri_side_dir = os.path.join(patient_path, "mri", side)

        if not os.path.isdir(mri_side_dir):
            continue

        # Look for tar.gz
        tar_files = [f for f in os.listdir(mri_side_dir) if f.endswith(".tar.gz")]
        if not tar_files:
            continue

        tar_path = os.path.join(mri_side_dir, tar_files[0])

        # TEMP extract location
        temp_dir = tempfile.mkdtemp()

        try:
            # Extract DICOMs
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=temp_dir)

            dicom_files = []
            for root, _, files in os.walk(temp_dir):
                for f in files:
                    dicom_files.append(os.path.join(root, f))

            dicoms = [pydicom.dcmread(f) for f in dicom_files]
            dicoms.sort(key=lambda d: getattr(d, "InstanceNumber",
                                              getattr(d, "SliceLocation", 0)))

            
            #if you want to see all the slices
            # # Convert slices ONE BY ONE (no volume array)
            # pil_slices = []
            # for dcm in dicoms:
            #     sl = dcm.pixel_array.astype(np.float32)

            #     sl -= sl.min()
            #     if sl.max() > 0:
            #         sl /= sl.max()

            #     sl_img = Image.fromarray((sl * 255).astype(np.uint8))
            #     pil_slices.append(sl_img)

            # final_png = stack_slices_grid(pil_slices, grid_cols=12)

            
            #if you want to see the MRI
            volume_3d = np.stack([d.pixel_array.astype(np.float32) for d in dicoms], axis=0)

            # Normalize to [0,1]
            volume_3d -= volume_3d.min()
            if volume_3d.max() > 0:
                volume_3d /= volume_3d.max()

            
            # Choose ONE of the following 

            # 1) Maximum Intensity Projection (MIP)
            unique_slice = volume_3d.max(axis=0)

            # 2) OR: Mean projection
            #unique_slice = volume_3d.mean(axis=0)

            # 3) OR: Median projection
            #unique_slice = np.median(volume_3d, axis=0)

            # Convert to uint8 PIL image
            final_png = Image.fromarray((unique_slice * 255).astype(np.uint8))

            return patient_id, final_png

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    return None, None



# SAVE SAMPLED MRIs
def save_sampled_mris(sampled_dict, dataset_root, output_root, side="left"):

    os.makedirs(output_root, exist_ok=True)

    for cluster_id, patient_list in sampled_dict.items():

        print(f"\n=== Processing cluster {cluster_id} ===")

        cluster_dir = os.path.join(output_root, f"cluster_{cluster_id}")
        os.makedirs(cluster_dir, exist_ok=True)

        for pid in patient_list:
            print(f" → Loading patient {pid}")

            try:
                patient_id, final_png = get_image(dataset_root, pid, side)

                if final_png is None:
                    print(f"    No MRI for patient {pid}")
                    continue

                out_path = os.path.join(cluster_dir, f"{patient_id}.png")
                final_png.save(out_path)

                print(f"    Saved {out_path}")

            except Exception as e:
                print(f"   Failed to load {pid}: {e}")


def compute_all_stats_by_kl(df_clinical, side="left", output_path=None):

    # Clean column names
    df_clinical.columns = df_clinical.columns.str.strip()

    # Determine KL column
    kl_col = f"V00XRKL_{'L' if side == 'left' else 'R'}"
    print(f"Using KL column: {kl_col}")

    if kl_col not in df_clinical.columns:
        raise KeyError(f"KL column {kl_col} not found in clinical dataframe!")

    # Ensure KL values are integers
    df_clinical[kl_col] = df_clinical[kl_col].astype(float).astype("Int64")

    # Group patients by KL
    grouped = df_clinical.groupby(kl_col)

    all_rows = []

    for kl_value, patients_df in grouped:

        print(f"\n=== Processing KL grade {kl_value}, n={len(patients_df)} ===")

        # Surgery percentages
        pct_yes, pct_no = compute_surgery_percentages(patients_df, side)

        # Mean stats for symptoms and structure
        stats = compute_all_statistics(patients_df, side)

        # Flatten stats dictionary into one row
        row = {
            "KL_GRADE": kl_value,
            "NUM_PATIENTS": len(patients_df),
            "SURGERY_YES_%": pct_yes,
            "SURGERY_NO_%": pct_no,
        }

        for subgroup, measures in stats.items():
            for varname, value in measures.items():
                row[f"{varname}"] = value

        all_rows.append(row)

    # Create final DataFrame
    result = pd.DataFrame(all_rows).sort_values("KL_GRADE")

    # Save
    if output_path is not None:
        result.to_csv(output_path, index=False)
        print(f"✔ Saved KL statistics to {output_path}")

    return result






def main():

    #DATASET_ROOT = "/vol/miltank/projects/practical_wise2526/knee-osteoarthritis-severity/data/cleaned_images_baseline"
    #OUTPUT_ROOT = "/vol/miltank/users/foca/adlm-knee-osteoarthritis/results/sampled_patientsMIP"
    SIDE = "left"

    # file_path = "/vol/miltank/users/foca/adlm-knee-osteoarthritis/results/autoencoder/csv/mri_clusters_left.csv"
    # clinical="/vol/miltank/users/foca/adlm-knee-osteoarthritis/csv/clinical00_cleaned.csv"
    clinical_local= "/Users/filippofocaccia/Desktop/adlm-knee-osteoarthritis/csv/clinical00_cleaned.csv"
    df_clinical = pd.read_csv(clinical_local, sep = ',')
    #pat_clust = pd.read_csv(file_path)

    #num_clusters = pat_clust["cluster"].nunique()
    # sampled = extract_random_patients(pat_clust, num_clusters)

   
    
    compute_all_stats_by_kl(
        df_clinical,
        side="left",
        output_path="/Users/filippofocaccia/Desktop/adlm-knee-osteoarthritis/csv/kl_clusters_left.csv"
)


if __name__ == "__main__":
    main()

