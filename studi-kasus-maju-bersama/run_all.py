"""Jalankan seluruh studi kasus secara berurutan."""
import subprocess, sys
from pathlib import Path

src = Path(__file__).parent / "src"
for s in ["01_generate_data.py", "02_data_quality.py", "03_late_payment.py", "04_anomaly_detection.py"]:
    print(f"\n{'=' * 70}\n>>> {s}\n{'=' * 70}")
    subprocess.run([sys.executable, s], cwd=src, check=True)
