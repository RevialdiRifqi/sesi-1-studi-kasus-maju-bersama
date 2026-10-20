"""Membuat data SINTETIS transaksi PT Maju Bersama (distributor fiktif).

Data sengaja dibuat 'kotor' (duplikat, kategori tidak konsisten, nilai kosong)
dan disisipi anomali dengan label kebenaran (ground truth) agar deteksi dapat dievaluasi.
"""
import numpy as np
import pandas as pd
from common import DATA

rng = np.random.default_rng(42)
N = 6000

produk = pd.DataFrame({
    "kode_produk": [f"P{i:03d}" for i in range(1, 21)],
    "kategori": rng.choice(["Makanan", "Minuman", "Kebersihan", "Atk"], 20),
    "harga_dasar": rng.integers(5, 120, 20) * 1000,
})
risiko_pelanggan = {f"C{i:03d}": rng.beta(1.2, 8) for i in range(1, 81)}  # peluang telat dasar

tanggal = pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, N), unit="D")
pel = rng.choice(list(risiko_pelanggan), N)
prod = produk.sample(N, replace=True, random_state=1).reset_index(drop=True)
qty = np.clip(rng.poisson(12, N), 1, None)
diskon = rng.choice([0, 0, 2, 5, 5, 10], N).astype(float)
harga = (prod["harga_dasar"] * rng.normal(1.0, 0.03, N)).round(-2)

df = pd.DataFrame({
    "id_transaksi": [f"T{i:05d}" for i in range(N)],
    "tanggal": tanggal, "kode_pelanggan": pel, "kode_produk": prod["kode_produk"],
    "kategori": prod["kategori"], "qty": qty, "harga_satuan": harga, "diskon_persen": diskon,
})
df["total"] = (df.qty * df.harga_satuan * (1 - df.diskon_persen / 100)).round(0)

# ---- Suntik anomali (3,5%) ----
df["label_anomali"] = 0
df["jenis_anomali"] = ""
idx = rng.choice(N, int(N * 0.035), replace=False)
for k, i in enumerate(idx):
    jenis = ["diskon_berlebihan", "qty_tidak_wajar", "harga_tidak_konsisten", "total_tidak_sesuai"][k % 4]
    if jenis == "diskon_berlebihan":
        df.loc[i, "diskon_persen"] = rng.choice([35, 45, 60])
    elif jenis == "qty_tidak_wajar":
        df.loc[i, "qty"] = int(rng.integers(150, 400))
    elif jenis == "harga_tidak_konsisten":
        df.loc[i, "harga_satuan"] = round(df.loc[i, "harga_satuan"] * rng.choice([0.3, 2.5, 4.0]), -2)
    if jenis != "total_tidak_sesuai":
        df.loc[i, "total"] = round(df.loc[i, "qty"] * df.loc[i, "harga_satuan"] * (1 - df.loc[i, "diskon_persen"] / 100))
    else:
        df.loc[i, "total"] = round(df.loc[i, "total"] * rng.choice([0.5, 1.8, 3.0]))
    df.loc[i, ["label_anomali", "jenis_anomali"]] = [1, jenis]

# ---- Data pembayaran invoice (untuk contoh akurasi vs recall & data leakage) ----
df["jatuh_tempo"] = df["tanggal"] + pd.Timedelta(days=30)
p_telat = df["kode_pelanggan"].map(risiko_pelanggan) + (df["total"] > df["total"].quantile(0.9)) * 0.08
terlambat = rng.random(N) < np.clip(p_telat, 0, 0.9)
selisih = np.where(terlambat, rng.integers(1, 30, N), -rng.integers(0, 20, N))
df["tanggal_bayar"] = df["jatuh_tempo"] + pd.to_timedelta(selisih, unit="D")
df["terlambat"] = terlambat.astype(int)

# ---- Kotori data ----
df.loc[rng.choice(N, 120, replace=False), "diskon_persen"] = np.nan
kotor = rng.choice(N, 150, replace=False)
df.loc[kotor, "kategori"] = df.loc[kotor, "kategori"].str.upper()
df = pd.concat([df, df.sample(40, random_state=3)], ignore_index=True)

df.to_csv(DATA / "transaksi_raw.csv", index=False)
print(f"Data dibuat: {len(df)} baris | proporsi terlambat = {df.terlambat.mean():.1%} | anomali = {df.label_anomali.mean():.1%}")
