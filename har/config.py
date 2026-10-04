"""Shared constants. OWNER: Person 1. Frozen after the Day-1 kickoff merge.

Change this file only through a PR that BOTH people approve.
"""
import os
from pathlib import Path

SEED = 229            # one seed for every split, fold and model
TEST_FRACTION = 0.15  # report: 85% train / 15% test
N_FOLDS = 5           # report: 5-fold CV on the training set only

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = Path(os.environ.get("HAR_RAW_DIR", REPO_ROOT / "data/raw/PAMAP2_Dataset/Protocol"))
PROCESSED_DIR = Path(os.environ.get("HAR_PROCESSED_DIR", REPO_ROOT / "data/processed"))
RESULTS_DIR = REPO_ROOT / "results"
DOCS_DIR = REPO_ROOT / "docs"

# The 12 protocol activities named in the report (Section III).
ACTIVITY_NAMES = {
    1: "lying", 2: "sitting", 3: "standing", 4: "walking", 5: "running",
    6: "cycling", 7: "Nordic walking", 12: "ascending stairs",
    13: "descending stairs", 16: "vacuum cleaning", 17: "ironing",
    24: "rope jumping",
}
LABEL_ORDER = sorted(ACTIVITY_NAMES)   # [1, 2, 3, ..., 24] (12 labels)
TRANSIENT_LABEL = 0                    # "other / transient" -> removed

# PAMAP2 .dat layout: 54 space-separated columns (see the dataset readme).
IMU_FIELDS = [
    "temp",
    "acc16_x", "acc16_y", "acc16_z",   # +-16 g accelerometer
    "acc6_x", "acc6_y", "acc6_z",      # +-6 g accelerometer (saturates)
    "gyro_x", "gyro_y", "gyro_z",
    "mag_x", "mag_y", "mag_z",
    "ori_0", "ori_1", "ori_2", "ori_3",  # orientation: readme says invalid
]
IMU_LOCATIONS = ["hand", "chest", "ankle"]
RAW_COLUMNS = ["timestamp", "activity_id", "heart_rate"] + [
    f"{loc}_{f}" for loc in IMU_LOCATIONS for f in IMU_FIELDS
]
assert len(RAW_COLUMNS) == 54

# Per-IMU columns used by the authors' public code (feature_indices in parent_class.py):
# temperature, +-16 g accel, gyro, magnetometer = 10 per IMU.
USED_IMU_FIELDS = ["temp", "acc16_x", "acc16_y", "acc16_z",
                   "gyro_x", "gyro_y", "gyro_z", "mag_x", "mag_y", "mag_z"]

FEATURE_SETS = {
    # "reduced" / "limited" = hand IMU + heart rate -> 11 columns
    "reduced": ["heart_rate"] + [f"hand_{f}" for f in USED_IMU_FIELDS],
    # "full" = hand + chest + ankle IMUs + heart rate -> 31 columns
    "full": ["heart_rate"] + [f"{loc}_{f}" for loc in IMU_LOCATIONS for f in USED_IMU_FIELDS],
}

# Columns that must NEVER be model inputs.
FORBIDDEN_FEATURES = {"timestamp", "activity_id", "subject_id", "row_id"}
