#!/usr/bin/env python3
import subprocess
from pathlib import Path


GLOBAL_SCRIPT = Path("/home/pi/village/village/scripts/global_csv_for_subject.py")
REMOTE_PYTHON = "~/miniconda3/envs/report/bin/python"
REMOTE_SESSIONS = "/storage/training_village/COT_cannula_data/sessions"

script = GLOBAL_SCRIPT.read_bytes()

for i in range(1, 13):
    subject = f"NUO{i:03d}"
    print(f"Running {subject}")
    subprocess.run(
        [
            "ssh",
            "mini",
            f"{REMOTE_PYTHON} - {subject} {REMOTE_SESSIONS}",
        ],
        input=script,
        check=True,
    )
