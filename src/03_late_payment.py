"""Prediksi keterlambatan pembayaran invoice.

Tiga pelajaran dari materi:
 1. Akurasi tinggi != model bagus (class imbalance -> recall rendah).
 2. Metrik yang tepat: precision, recall, F1, ROC-AUC.
 3. Data leakage: fitur yang baru diketahui setelah pembayaran membuat model 'terlalu sempurna'.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from common import load_raw, clean, RESULTS

df = clean(load_raw())
df["bulan"] = df["tanggal"].dt.month
df["hari_bayar_dari_tempo"] = (df["tanggal_bayar"] - df["jatuh_tempo"]).dt.days  # LEAKAGE: baru ada setelah bayar

AMAN = ["qty", "harga_satuan", "diskon_persen", "total", "bulan"]
KAT = ["kategori", "kode_pelanggan"]


def pipeline(model, num):
    pre = ColumnTransformer([("num", StandardScaler(), num), ("kat", OneHotEncoder(handle_unknown="ignore"), KAT)])
    return Pipeline([("pre", pre), ("model", model)])


def evaluasi(nama, model, num=AMAN):
    X, y = df[num + KAT], df["terlambat"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    pipe = pipeline(model, num).fit(Xtr, ytr)
    pred, proba = pipe.predict(Xte), pipe.predict_proba(Xte)[:, 1]
    return {"skenario": nama, "accuracy": accuracy_score(yte, pred),
            "precision": precision_score(yte, pred, zero_division=0),
            "recall": recall_score(yte, pred, zero_division=0),
            "f1": f1_score(yte, pred, zero_division=0), "roc_auc": roc_auc_score(yte, proba)}


hasil = [
    evaluasi("Baseline: selalu 'tidak terlambat'", DummyClassifier(strategy="most_frequent")),
    evaluasi("Logistic Regression", LogisticRegression(max_iter=1000)),
    evaluasi("Decision Tree (depth=4)", DecisionTreeClassifier(max_depth=4, random_state=0)),
    evaluasi("Random Forest", RandomForestClassifier(200, random_state=0, n_jobs=-1)),
    evaluasi("Logistic Reg + class_weight", LogisticRegression(max_iter=1000, class_weight="balanced")),
    evaluasi("Random Forest + class_weight", RandomForestClassifier(200, class_weight="balanced_subsample", min_samples_leaf=5, random_state=0, n_jobs=-1)),
    evaluasi("!! Decision Tree DENGAN LEAKAGE", DecisionTreeClassifier(max_depth=4, random_state=0), AMAN + ["hari_bayar_dari_tempo"]),
]
out = pd.DataFrame(hasil).round(3)
out.to_csv(RESULTS / "hasil_prediksi_keterlambatan.csv", index=False)
print(f"Proporsi kelas terlambat: {df['terlambat'].mean():.1%} (kelas tidak seimbang)\n")
print(out.to_string(index=False))
