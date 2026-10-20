"""Data understanding: audit kualitas data (garbage in, garbage out)."""
from common import load_raw, clean, RESULTS

raw = load_raw()
rep = {
    "baris_awal": len(raw),
    "duplikasi_id": int(raw.duplicated("id_transaksi").sum()),
    "nilai_kosong_diskon": int(raw["diskon_persen"].isna().sum()),
    "variasi_kategori_sebelum": sorted(raw["kategori"].unique().tolist()),
}
c = clean(raw)
rep["baris_setelah_bersih"] = len(c)
rep["variasi_kategori_sesudah"] = sorted(c["kategori"].unique().tolist())
for k, v in rep.items():
    print(f"{k}: {v}")
c.to_csv(RESULTS / "transaksi_bersih.csv", index=False)
