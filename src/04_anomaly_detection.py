"""Deteksi anomali transaksi: rule-based vs Isolation Forest vs hybrid.

Hasil = REKOMENDASI pemeriksaan, bukan vonis bahwa transaksi pasti curang.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from common import load_raw, clean, RESULTS

df = clean(load_raw())

# ---------- Fitur turunan ----------
df["harga_median_produk"] = df.groupby("kode_produk")["harga_satuan"].transform("median")
df["rasio_harga"] = df["harga_satuan"] / df["harga_median_produk"]
df["total_hitung"] = df["qty"] * df["harga_satuan"] * (1 - df["diskon_persen"] / 100)
df["selisih_total"] = (df["total"] - df["total_hitung"]).abs() / df["total_hitung"].clip(lower=1)
df["log_qty"] = np.log1p(df["qty"])
df["log_total"] = np.log1p(df["total"])

# ---------- 1. Rule-based ----------
aturan = {
    "diskon > 20%": df["diskon_persen"] > 20,
    "qty > 100": df["qty"] > 100,
    "harga menyimpang >50% dari median produk": (df["rasio_harga"] > 1.5) | (df["rasio_harga"] < 0.5),
    "total != qty x harga x (1-diskon)": df["selisih_total"] > 0.01,
}
alasan = pd.DataFrame(aturan)
df["flag_rule"] = alasan.any(axis=1).astype(int)
df["alasan_rule"] = alasan.apply(lambda r: "; ".join(r.index[r]), axis=1)

# ---------- 2. Isolation Forest ----------
fitur = ["log_qty", "harga_satuan", "diskon_persen", "rasio_harga", "selisih_total", "log_total"]
iso = IsolationForest(n_estimators=300, contamination=0.04, random_state=42).fit(df[fitur])
df["skor_anomali"] = -iso.score_samples(df[fitur])  # makin besar = makin tidak wajar
df["flag_iforest"] = (iso.predict(df[fitur]) == -1).astype(int)

# ---------- 3. Gabungan ----------
df["flag_hybrid"] = ((df["flag_rule"] == 1) | (df["flag_iforest"] == 1)).astype(int)
df["flag_konsensus"] = ((df["flag_rule"] == 1) & (df["flag_iforest"] == 1)).astype(int)

# ---------- Evaluasi terhadap label sintetis ----------
y = df["label_anomali"]
rows = []
for nama, kol in [("Rule-based", "flag_rule"), ("Isolation Forest", "flag_iforest"),
                  ("Hybrid (rule ATAU iForest)", "flag_hybrid"), ("Konsensus (rule DAN iForest)", "flag_konsensus")]:
    rows.append({"metode": nama, "ditandai": int(df[kol].sum()),
                 "precision": precision_score(y, df[kol]), "recall": recall_score(y, df[kol]), "f1": f1_score(y, df[kol])})
hasil = pd.DataFrame(rows).round(3)
print(f"Anomali sebenarnya (label sintetis): {int(y.sum())} dari {len(df)} transaksi\n")
print(hasil.to_string(index=False))
print(f"\nAverage precision (skor Isolation Forest): {average_precision_score(y, df['skor_anomali']):.3f}")
per_jenis = df[y == 1].groupby("jenis_anomali")[["flag_rule", "flag_iforest", "flag_hybrid"]].mean().round(3)
print("\nRecall per jenis anomali:\n", per_jenis.to_string())

# ---------- Luaran ----------
hasil.to_csv(RESULTS / "hasil_deteksi_anomali.csv", index=False)
per_jenis.to_csv(RESULTS / "recall_per_jenis_anomali.csv")
kolom = ["id_transaksi", "tanggal", "kode_pelanggan", "kode_produk", "qty", "harga_satuan",
         "diskon_persen", "total", "skor_anomali", "flag_rule", "flag_iforest", "alasan_rule"]
df[df.flag_hybrid == 1].sort_values("skor_anomali", ascending=False)[kolom].to_csv(
    RESULTS / "daftar_transaksi_perlu_diperiksa.csv", index=False)

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(df.loc[y == 0, "skor_anomali"], bins=50, alpha=.7, label="Normal")
ax.hist(df.loc[y == 1, "skor_anomali"], bins=50, alpha=.7, label="Anomali (label sintetis)")
ax.set(xlabel="Skor anomali Isolation Forest", ylabel="Jumlah transaksi", title="Distribusi skor anomali")
ax.legend(); fig.tight_layout(); fig.savefig(RESULTS / "distribusi_skor_anomali.png", dpi=150)
