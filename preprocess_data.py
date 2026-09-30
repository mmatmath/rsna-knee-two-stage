import glob
import hashlib
import os
from zipfile import BadZipFile

import cv2
import numpy as np
import pandas as pd
import pydicom
from dataclasses import dataclass
from multiprocessing import Pool
from tqdm import tqdm


N_SLICES = 24
IMG_SIZE = 224
CACHE_VERSION = 1

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
    dicom_path: str = "./data/train_series"
    output_path: str = "./data/npy_study"
    meta_path: str = "./data/train_meta.csv"
    n_processes: int = min(4, max(1, (os.cpu_count() or 2) - 1))


def series_files(root, study, series):
    return sorted(glob.glob(os.path.join(root, str(study), str(series), "*.dcm")))


def slice_position(dicom):
    if "ImageOrientationPatient" in dicom and "ImagePositionPatient" in dicom:
        orientation = np.asarray(dicom.ImageOrientationPatient, dtype=np.float64)
        row, col = orientation[:3], orientation[3:]
        normal = np.cross(row, col)
        # Increasing patient-space coordinate gives a consistent direction within each plane.
        if normal[np.argmax(np.abs(normal))] < 0:
            normal = -normal
        return float(np.dot(np.asarray(dicom.ImagePositionPatient, dtype=np.float64), normal))
    return float(getattr(dicom, "InstanceNumber", 0))


def load_series(files):
    slices = []
    for path in files:
        try:
            dicom = pydicom.dcmread(path)
            pixels = np.asarray(dicom.pixel_array, dtype=np.float32)
            if pixels.ndim != 2:
                continue
            slices.append((slice_position(dicom), pixels))
        except Exception:
            continue
    slices.sort(key=lambda item: item[0])
    if not slices:
        return None
    volume = np.stack([item[1] for item in slices])
    volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)
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
        if pd.isna(row.Fluid_Sensitive):
            continue
        plane = str(row.Anatomical_Plane).strip().capitalize()
        fluid = int(row.Fluid_Sensitive)
        key = (plane, fluid)
        if key not in slot_mapping:
            continue
        files = series_files(dicom_path, row.StudyInstanceUID, row.SeriesInstanceUID)
        candidate = (len(files), str(row.SeriesInstanceUID), files)
        slot = slot_mapping[key]
        if slot not in selected or candidate[:2] > selected[slot][:2]:
            selected[slot] = candidate
    return selected


def source_stamp(selected):
    """Track the chosen DICOM files without reading their pixel data."""
    digest = hashlib.sha256()
    for slot, (_, series, files) in sorted(selected.items()):
        digest.update(f"{slot}:{series}\n".encode())
        for path in files:
            stat = os.stat(path)
            digest.update(f"{path}:{stat.st_size}:{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def process_study(args):
    study, records, config = args
    df_study = pd.DataFrame(records)
    selected = choose_series(df_study, config.dicom_path)
    stamp = source_stamp(selected)
    cache_path = os.path.join(config.output_path, str(study) + ".npz")
    if os.path.exists(cache_path):
        try:
            with np.load(cache_path) as data:
                valid = (int(data["cache_version"]) == CACHE_VERSION
                         and str(data["source_stamp"]) == stamp
                         and data["slot_mask"].shape == (6,)
                         and "images" in data.files)
                if valid:
                    slot_mask = data["slot_mask"]
                    meta = {"StudyInstanceUID": study}
                    for slot in range(6):
                        meta[f"slot_{slot}_series"] = selected[slot][1] if slot_mask[slot] else ""
                    return meta, True
        except (OSError, ValueError, KeyError, BadZipFile):
            pass

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

    if not slot_mask.any():
        return None, False
    os.makedirs(config.output_path, exist_ok=True)
    temp_path = cache_path + ".tmp.npz"
    np.savez_compressed(temp_path, images=images, slot_mask=slot_mask,
                        cache_version=CACHE_VERSION, source_stamp=stamp)
    os.replace(temp_path, cache_path)
    return meta, False


def main():
    config = Configuration()
    df = pd.read_csv(config.series_csv)
    jobs = [(study, group.to_dict("records"), config) for study, group in df.groupby("StudyInstanceUID", sort=True)]
    rows = []
    reused = 0
    with Pool(processes=config.n_processes) as pool:
        for row, cached in tqdm(pool.imap(process_study, jobs), total=len(jobs)):
            if row is not None:
                rows.append(row)
                reused += cached
    if not rows:
        raise ValueError("No usable studies; check the train_series DICOM directory and pixel decoders")
    df_meta = pd.DataFrame(rows).sort_values("StudyInstanceUID")
    df_meta.to_csv(config.meta_path, index=False)
    print("Usable studies:", len(df_meta), "of", len(jobs))
    print("Reused study caches:", reused)
    print("Slot coverage:", df_meta[[f"slot_{i}_series" for i in range(6)]].notna().sum().to_dict())
    df_slots = df.dropna(subset=["Fluid_Sensitive"]).copy()
    df_slots["slot"] = [slot_mapping.get((str(plane).strip().capitalize(), int(fluid)))
                        for plane, fluid in zip(df_slots.Anatomical_Plane, df_slots.Fluid_Sensitive)]
    duplicates = df_slots.dropna(subset=["slot"]).groupby(["StudyInstanceUID", "slot"]).size()
    print("Studies with duplicate slot candidates:", int(duplicates.gt(1).groupby(level=0).any().sum()))
    print("Saved:", config.meta_path)


if __name__ == "__main__":
    main()
