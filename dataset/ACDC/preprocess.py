import re
import json
from pathlib import Path

import SimpleITK as sitk
from const import BASE_PATH

sitk.ProcessObject.SetGlobalWarningDisplay(False)


def preprocess_acdc_directory(root_path: Path, output_json_path: Path) -> None:
    """
    Scan ACDC patient folders under `root_path`, locate ED/ES volumes and their segmentations,
    and write a summary JSON for downstream loading.

    Parameters
    ----------
    root_path : Path
        Root directory of the ACDC dataset (e.g. BASE_PATH / 'train').
    output_json_path : Path
        File path where the resulting JSON metadata will be saved.
    """
    records = []

    for patient_dir in sorted(root_path.iterdir()):
        if not patient_dir.is_dir() or patient_dir.name.startswith('.'):
            continue

        cfg_file = patient_dir / "Info.cfg"
        if not cfg_file.exists():
            print(f"Warning: missing Info.cfg in {patient_dir.name}")
            continue

        ed_frame = es_frame = None
        for line in cfg_file.read_text().splitlines():
            if "ED" in line:
                nums = re.findall(r"\d+", line)
                ed_frame = int(nums[0]) if nums else None
            elif "ES" in line:
                nums = re.findall(r"\d+", line)
                es_frame = int(nums[0]) if nums else None

        if ed_frame is None or es_frame is None:
            print(f"Warning: could not parse ED/ES frames in {patient_dir.name}")
            continue

        frames = {}
        for nifti in patient_dir.glob("*.nii.gz"):
            name = nifti.name
            rel_path = str(Path(patient_dir.name) / name)
            if f"frame{ed_frame:02}" in name:
                frames["ED_gt" if "_gt" in name else "ED"] = rel_path
            elif f"frame{es_frame:02}" in name:
                frames["ES_gt" if "_gt" in name else "ES"] = rel_path

        if len(frames) == 4:
            records.append({
                "patient_id": patient_dir.name,
                "frames": frames
            })
        else:
            print(f"Warning: incomplete frames for {patient_dir.name}")

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with output_json_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=4)

    print(f"Processed metadata saved to {output_json_path}")
