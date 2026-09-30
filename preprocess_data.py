import glob
import os

import cv2
import numpy as np
import pandas as pd
import pydicom
from dataclasses import dataclass
from multiprocessing import Pool
from tqdm import tqdm


N_SLICES = 9
IMG_SIZE = 224

slot_mapping = {
    ("Sagittal", 0): 0,
    ("Sagittal", 1): 1,
    ("Coronal", 0): 2,
    ("Coronal", 1): 3,
    ("Axial", 0): 4,
    ("Axial", 1): 5,
}


#----------------------------------------------------------------------------------------------------------------------#
# Configuration                                                                                                        #
#----------------------------------------------------------------------------------------------------------------------#
@dataclass
class Configuration:
    series_csv: str = "./data/train_series.csv"
    dicom_path: str = "./data/train_images"
    output_path: str = "./data/npy_study"
    meta_path: str = "./data/train_meta.csv"
    n_processes: int = max(1, (os.cpu_count() or 2) - 1)


def series_files(root, study, series):
    patterns = [
        os.path.join(root, str(study), str(series), "*.dcm"),
        os.path.join(root, str(study), str(series), "*"),
        os.path.join(root, str(series), "*.dcm"),
    ]
    for pattern in patterns:
        files = sorted(glob.glob(pattern))
        if files:
            return [path for path in files if os.path.isfile(path)]
    return []


def slice_position(dicom):
    if "ImageOrientationPatient" in dicom and "ImagePositionPatient" in dicom:
        orientation = np.asarray(dicom.ImageOrientationPatient, dtype=np.float64)
        row, col = orientation[:3], orientation[3:]
        normal = np.cross(row, col)
        return float(np.dot(np.asarray(dicom.ImagePositionPatient, dtype=np.float64), normal))
    return float(getattr(dicom, "InstanceNumber", 0))


def load_series(files):
    slices = []
    for path in files:
        try:
            dicom = pydicom.dcmread(path)
            slices.append((slice_position(dicom), dicom.pixel_array.astype(np.float32)))
        except Exception:
            continue
    slices.sort(key=lambda item: item[0])
    if not slices:
        return None
    volume = np.stack([item[1] for item in slices])
    p1, p99 = np.percentile(volume, [1, 99])
    volume = np.clip(volume, p1, p99)
    volume = (volume - p1) / (p99 - p1 + 1e-6)
    return (volume * 255).round().astype(np.uint8)


def sample_series(volume):
    low = int(round((len(volume) - 1) * 0.05))
    high = int(round((len(volume) - 1) * 0.95))
    indices = np.linspace(low, max(low, high), N_SLICES).round().astype(int)
    return np.stack([cv2.resize(volume[i], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR) for i in indices])


def choose_series(df_study, dicom_path):
    selected = {}
    for _, row in df_study.iterrows():
        key = (str(row.Anatomical_Plane), int(row.Fluid_Sensitive))
        if key not in slot_mapping:
            continue
        files = series_files(dicom_path, row.StudyInstanceUID, row.SeriesInstanceUID)
        candidate = (len(files), str(row.SeriesInstanceUID), files)
        slot = slot_mapping[key]
        if slot not in selected or candidate[:2] > selected[slot][:2]:
            selected[slot] = candidate
    return selected


def process_study(args):
    study, records, config = args
    df_study = pd.DataFrame(records)
    selected = choose_series(df_study, config.dicom_path)
    images = np.zeros((6, N_SLICES, IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    slot_mask = np.zeros(6, dtype=np.uint8)
    meta = {"StudyInstanceUID": study}

    for slot in range(6):
        meta[f"slot_{slot}_series"] = ""
        if slot not in selected:
            continue
        _, series, files = selected[slot]
        volume = load_series(files)
        if volume is None:
            continue
        images[slot] = sample_series(volume)
        slot_mask[slot] = 1
        meta[f"slot_{slot}_series"] = series

    os.makedirs(config.output_path, exist_ok=True)
    np.savez_compressed(os.path.join(config.output_path, str(study) + ".npz"),
                        images=images, slot_mask=slot_mask)
    return meta


def main():
    config = Configuration()
    df = pd.read_csv(config.series_csv)
    jobs = [(study, group.to_dict("records"), config) for study, group in df.groupby("StudyInstanceUID", sort=True)]
    rows = []
    with Pool(processes=config.n_processes) as pool:
        for result in tqdm(pool.imap(process_study, jobs), total=len(jobs)):
            rows.append(result)
    pd.DataFrame(rows).sort_values("StudyInstanceUID").to_csv(config.meta_path, index=False)
    print("Saved:", config.meta_path)


if __name__ == "__main__":
    main()
