CHANNELS_TO_DROP = [
    "X4", "X5", "X6", "DC3", "X7", "X8", "SaO2", "DC8",
    "ROC-A1", "F3-A2", "C3-A2", "O1-A2", "F4-A1",
    "O2-A1", "X2", "X3",
]

ARTIFACT_THRESHOLD = 60
CONFIDENCE_THRESHOLD = 0.95

COMBINED_CLASS_NAMES = ["Wake", "N1", "N2", "N3", "REM"]
COMBINED_CLASS_LABELS = [0, 1, 2, 3, 4]

MULTICLASS_CLASS_NAMES = ["Wake", "N1, N2", "N3", "REM"]
MULTICLASS_CLASS_LABELS = [0, 1, 2, 3]

BINARY_CLASS_NAMES = ["N1", "N2"]
BINARY_CLASS_LABELS = [0, 1]

train_patients = 7
