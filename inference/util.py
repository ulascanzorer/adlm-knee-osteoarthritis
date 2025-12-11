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


def compute_koos_pain_by_kl(
        df_clinical: pd.DataFrame,
        side: str = "left",
        output_path: str = None
    ) -> pd.DataFrame:
    """
    Compute mean KOOS pain per KL grade cluster directly from clinical data.
    Optionally save results to CSV.
    """

    # Choose KL column
    kl_col = variables["VARIABLES"]["KL_GRADE"][0] if side == "left" else variables["VARIABLES"]["KL_GRADE"][1]

    # Choose KOOS pain column
    koos_col = "V00KOOSKPL" if side == "left" else "V00KOOSKPR"

    # Filter only rows with valid KL grade & KOOS pain
    df = df_clinical.dropna(subset=[kl_col, koos_col]).copy()

    df[kl_col] = df[kl_col].astype(int)

    # Compute mean KOOS pain per KL grade
    result = (
        df.groupby(kl_col)[koos_col]
        .mean()
        .reset_index()
        .rename(columns={
            kl_col: "KL_GRADE",
            koos_col: "KOOS_PAIN_MEAN"
        })
    )

    # Save to CSV if path given
    if output_path is not None:
        result.to_csv(output_path, index=False)
        print(f"✔ Saved KOOS pain stats to {output_path}")

    return result



def main():

    DATASET_ROOT = "/vol/miltank/projects/practical_wise2526/knee-osteoarthritis-severity/data/cleaned_images_baseline"
    OUTPUT_ROOT = "/vol/miltank/users/foca/adlm-knee-osteoarthritis/results/sampled_patientsMIP"
    SIDE = "left"

    file_path = "/vol/miltank/users/foca/adlm-knee-osteoarthritis/results/autoencoder/csv/mri_clusters_left.csv"
    clinical="/vol/miltank/users/foca/adlm-knee-osteoarthritis/csv/clinical00_cleaned.csv"
    df_clinical = pd.read_csv(clinical, sep = ',')
    pat_clust = pd.read_csv(file_path)

    num_clusters = pat_clust["cluster"].nunique()
    # sampled = extract_random_patients(pat_clust, num_clusters)

   
    
    df_stats_left = compute_koos_pain_by_kl(
        df_clinical,
        side="left",
        output_path="/vol/miltank/users/foca/adlm-knee-osteoarthritis/csv/kl_clusters_left.csv"
)


if __name__ == "__main__":
    main()

