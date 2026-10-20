# Studi Kasus: Make Sense of Data with Analysis and AI
### Analisis Data Transaksi PT Maju Bersama (Distributor Fiktif)

**Penulis:** [Revialdi Rifqi Ramadhan Permana] | **NPM:** [G1A023066]

Studi kasus ini dibuat dari materi **Informatics Practitioner Lecture Series – Sesi 1** (mata kuliah DEEPL),
pemateri **Andria Arisal (BRIN)**, topik *Make Sense of Data with Analysis and AI*.
Seluruh data adalah **data sintetis** yang dibangkitkan oleh kode di repo ini; tidak ada data perusahaan nyata.

> **Catatan:** angka pada materi (akurasi ±89%, recall 0%, ROC ±0,6) berasal dari ringkasan transkrip otomatis.
> Studi kasus ini **tidak mengklaim mereplikasi angka tersebut**, melainkan mendemonstrasikan fenomena yang sama dengan data buatan sendiri.

## 1. Skenario
PT Maju Bersama adalah distributor yang pendapatannya naik, tetapi laba bersih turun, piutang membengkak, dan arus kas melemah.
Manajemen ingin menjawab dua pertanyaan berbasis data:

1. **Invoice mana yang berisiko dibayar terlambat?** (prediksi keterlambatan pembayaran)
2. **Transaksi mana yang tidak wajar dan perlu diperiksa?** (deteksi anomali; ide pengembangan skripsi)

## 2. Pemetaan Materi ke Kode

| Konsep dari pemateri | Di mana didemonstrasikan |
|---|---|
| CRISP-DM (business/data understanding, preparation, modeling, evaluation) | Seluruh alur `src/01`–`04` |
| Data dunia nyata kotor, *garbage in garbage out* | `01_generate_data.py` (duplikat, kosong, kategori tidak konsisten), `02_data_quality.py` |
| Analitik descriptive/diagnostic/predictive/prescriptive | Audit data (descriptive), alasan flag (diagnostic), prediksi telat (predictive), daftar pemeriksaan (prescriptive) |
| Akurasi tinggi belum tentu model baik; class imbalance | `03_late_payment.py` |
| Precision, recall, F1, ROC | `03` dan `04` |
| Data leakage | Skenario "Decision Tree DENGAN LEAKAGE" di `03` |
| Metode kompleks tidak selalu terbaik | Rule-based vs Isolation Forest di `04` |
| AI/hasil model harus diverifikasi manusia | Output berupa *rekomendasi pemeriksaan* beserta alasan |

## 3. Cara Menjalankan
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_all.py
```
Hasil tersimpan di folder `results/`. Data mentah dibuat ulang di `data/transaksi_raw.csv` (seed tetap). Angka model berbasis pohon (Random Forest) dapat bergeser sedikit tergantung versi library; tabel di bawah dihasilkan dengan scikit-learn 1.9.1.

## 4. Hasil Utama (seed 42)

### 4.1 Kualitas data
6.040 baris awal → 6.000 setelah menghapus 40 duplikat; 121 diskon kosong diisi 0; 8 variasi penulisan kategori disatukan menjadi 4.

### 4.2 Prediksi keterlambatan: jebakan akurasi
Hanya ±14,5% invoice yang terlambat.

| Skenario | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Baseline: selalu "tidak terlambat" | 0,855 | 0,000 | 0,000 | 0,000 | 0,500 |
| Logistic Regression | 0,863 | 0,739 | 0,078 | 0,142 | 0,724 |
| Random Forest | 0,854 | 0,478 | 0,101 | 0,167 | 0,689 |
| Logistic Reg + class_weight | 0,633 | 0,239 | 0,705 | 0,357 | 0,724 |
| Random Forest + class_weight | 0,794 | 0,338 | 0,442 | 0,383 | 0,721 |
| **Decision Tree + leakage** | **1,000** | **1,000** | **1,000** | **1,000** | **1,000** |

**Pembelajaran**
- Model dummy yang tidak berguna sama sekali sudah mencapai akurasi 85,5%. Model "terbaik" menurut akurasi (Logistic Regression, 86,3%) hanya menangkap ±8% invoice yang benar-benar telat.
- Menggunakan `class_weight` menurunkan akurasi, tetapi recall dan F1 naik jauh. Untuk penagihan, melewatkan invoice berisiko biasanya lebih mahal daripada memeriksa beberapa invoice yang ternyata aman.
- Sempurna (100%) pada skenario leakage adalah **tanda bahaya**: fitur `hari_bayar_dari_tempo` baru diketahui setelah pembayaran terjadi, sehingga tidak tersedia saat prediksi dibutuhkan.

### 4.3 Deteksi anomali (210 anomali tersuntik dari 6.000 transaksi)

| Metode | Ditandai | Precision | Recall | F1 |
|---|---|---|---|---|
| Rule-based | 280 | 0,750 | 1,000 | 0,857 |
| Isolation Forest | 240 | 0,833 | 0,952 | 0,889 |
| Hybrid (rule ATAU iForest) | 314 | 0,669 | 1,000 | 0,802 |
| Konsensus (rule DAN iForest) | 206 | 0,971 | 0,952 | 0,962 |

**Pembelajaran**
- Rule-based sangat baik untuk pelanggaran yang sudah diketahui, dan hasilnya mudah dijelaskan (kolom `alasan_rule`).
- Isolation Forest tanpa aturan tetap menangkap ±95% anomali, tetapi paling lemah pada harga tidak konsisten (recall 0,83) karena tidak memiliki aturan eksplisit "harga vs median produk".
- Menggabungkan keduanya memberi pilihan operasional: ATAU untuk cakupan maksimal, DAN untuk daftar pendek berprioritas tinggi.
- **Temuan menarik:** sebagian false positive rule-based berasal dari 121 diskon kosong yang diisi 0, padahal total aslinya dihitung dengan diskon bukan nol. Jadi *imputasi data dapat menciptakan anomali baru*. Ini contoh nyata mengapa persiapan data tidak sepenuhnya diserahkan ke AI.

## 5. Keterbatasan
- Data sintetis dan anomali disuntik dengan pola sederhana, sehingga hasilnya lebih mudah daripada data nyata. Jangan dibaca sebagai klaim performa pada transaksi sungguhan.
- Sinyal keterlambatan dalam data sintetis sengaja lemah; ROC-AUC ±0,72 bukan patokan untuk kasus nyata.
- Evaluasi anomali memakai label buatan sendiri; pada data nyata diperlukan validasi oleh pihak yang memahami proses transaksi.
- Hasil deteksi adalah **rekomendasi pemeriksaan**, bukan keputusan bahwa transaksi pasti salah atau curang.

## 6. Pengembangan Lanjutan
- Tuning `contamination` dan ambang aturan; uji pada dataset anomali publik atau data mitra yang telah dianonimkan.
- Dashboard web (Laravel/Streamlit) berisi daftar transaksi terindikasi, alasan, dan status pemeriksaan.
- Replikasi alur yang sama di KNIME/Orange untuk membandingkan pendekatan kode vs workflow visual.

## Struktur Repo
```
├── README.md
├── requirements.txt
├── run_all.py
├── src/
│   ├── common.py               # path & pembersihan data
│   ├── 01_generate_data.py     # data sintetis (kotor + anomali berlabel)
│   ├── 02_data_quality.py      # audit kualitas data
│   ├── 03_late_payment.py      # akurasi vs recall, class imbalance, leakage
│   └── 04_anomaly_detection.py # rule-based, Isolation Forest, hybrid
├── data/                       # transaksi_raw.csv (hasil generate)
└── results/                    # tabel metrik, daftar pemeriksaan, grafik
```

## Lisensi & Sumber
Kode: MIT. Materi konsep: ringkasan sesi Informatics Practitioner Lecture Series (A. Arisal, BRIN).
