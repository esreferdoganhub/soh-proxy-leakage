"""
config.py — Proje Yapılandırma Dosyası
HAT-Net: Li-ion Batarya SOH+RUL Multi-task Tahmini
"""

import torch
from pathlib import Path

# ─── Dizinler ────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
FEAT_DIR = DATA_DIR / "features"
RESULTS_DIR = BASE_DIR / "results"
CKPT_DIR = RESULTS_DIR / "checkpoints"
FIG_DIR = RESULTS_DIR / "figures"
ABLATION_DIR = RESULTS_DIR / "ablation"

# Dizinleri otomatik oluştur
for d in [
    DATA_DIR,
    RAW_DIR,
    PROC_DIR,
    FEAT_DIR,
    RESULTS_DIR,
    CKPT_DIR,
    FIG_DIR,
    ABLATION_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)


# ─── Apple Silicon MPS Cihaz Seçimi ─────────────────────────
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


DEVICE = get_device()

# ─── Veri Seti Parametreleri ─────────────────────────────────
CALCE_CELLS = [
    "CS2_35",
    "CS2_36",
    "CS2_37",
    "CS2_38",
    "CX2_16",
    "CX2_33",
    "CX2_34",
    "CX2_37",
]

SOH_EOL = 0.80  # End-of-life eşiği (%80)
NOMINAL_CAP = {  # Anma kapasitansları (Ah)
    "CALCE": 1.1,
    "MIT": 1.1,
    "HUST": 3.0,
}

# ─── CALCE Pouch Koşul-Bloklu Değerlendirme Protokolü ─────────
# Aynı çalışma koşuluna ait hücreler bir fold içinde karıştırılmaz.
CONDITION_FOLDS = {
    "F0": {
        "train": ("PL03", "PL04", "PL09", "PL11", "PL12", "PL17", "PL19", "PL21"),
        "val": ("PL10", "PL25", "PL14", "PL23"),
        "test": ("PL05", "PL13", "PL18", "PL24"),
    },
    "F1": {
        "train": ("PL03", "PL04", "PL09", "PL11", "PL12", "PL17", "PL19", "PL21"),
        "val": ("PL05", "PL13", "PL18", "PL24"),
        "test": ("PL10", "PL25", "PL14", "PL23"),
    },
    "F2": {
        "train": ("PL10", "PL05", "PL25", "PL13", "PL14", "PL18", "PL24", "PL23"),
        "val": ("PL03", "PL09", "PL12", "PL21"),
        "test": ("PL04", "PL11", "PL17", "PL19"),
    },
    "F3": {
        "train": ("PL10", "PL05", "PL25", "PL13", "PL14", "PL18", "PL24", "PL23"),
        "val": ("PL04", "PL11", "PL17", "PL19"),
        "test": ("PL03", "PL09", "PL12", "PL21"),
    },
}

EXPERIMENT_SEEDS = [0, 1, 2, 3, 4]


# ─── Sliding Window ──────────────────────────────────────────
WINDOW_SIZE = 30  # Giriş penceresi (döngü sayısı)
STRIDE = 1  # Adım büyüklüğü
PRED_HORIZON = 1  # Kaç döngü sonrası tahmin

# ─── Model Hiperparametreleri ────────────────────────────────
MODEL_CFG = {
    # CNN dalı
    "cnn_channels": [1, 32, 64],
    "cnn_kernel": 3,
    # BiLSTM dalı
    "lstm_hidden": 256,
    "lstm_layers": 2,
    "lstm_dropout": 0.3,
    # Transformer
    "d_model": 256,
    "n_heads": 8,
    "n_enc_layers": 3,
    "dim_feedforward": 512,
    "tf_dropout": 0.1,
    # Çıkış kafaları
    "soh_head_dims": [128, 64, 1],
    "rul_head_dims": [128, 64, 1],
    # Multi-task ağırlıkları
    "lambda_soh": 0.6,
    "lambda_rul": 0.4,
    "loss_learnable": True,
    # ── Tier 2: iTransformer Variate-Attention ────────────────
    # Zha et al. (2024) iTransformer: Inverted Transformers Are Effective for Time Series
    # Her fiziksel değişken (F=13 özellik) bir "token" olarak işlenir.
    # Değişkenler ARASI korelasyonu öğrenir; zaman adımları arasındaki dikkatten farklıdır.
    "use_variate_attn": True,  # Ablasyon: False → sadece zaman-bazlı dikkat
    "variate_attn_heads": 4,  # Değişken-dikkat kafaları (F=13 için 4 uygun)
    "variate_attn_layers": 2,  # İstif sayısı
    "variate_attn_dim": 64,  # Variate token boyutu (F'den bağımsız projeksiyon)
    "variate_dropout": 0.1,
}

# ─── True-PINN Fizik Kaybı (SPM difüzyon rezidüeli) ──────────
PINN_CFG = {
    "enabled": True,
    "chemistry": "LCO",           # "LCO" (Marquis2019) veya "LFP" (Chen2020)
    "hidden_dim": 64,
    "n_layers": 4,
    "collocation_points": 256,    # her batch için (t̂, r̂) örnek sayısı
    "lambda_pde": 0.05,           # PDE + sınır koşulu rezidüel ağırlığı
    "lambda_consistency": 0.1,    # HATNet-SOH <-> fizik-SOH tutarlılık ağırlığı
    "warmup_epochs": 10,          # lambda'ların 0 -> hedef değere tavlanma süresi
}

# ─── Eğitim Hiperparametreleri ───────────────────────────────
TRAIN_CFG = {
    "epochs": 150,
    "batch_size": 64,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "patience": 20,  # Early stopping
    "scheduler": "cosine",  # CosineAnnealingLR
    "mc_dropout_samples": 50,  # Belirsizlik tahmini
}

# ─── Göreve Özel Özellik Kayıtları ───────────────────────────
# SOH girdileri gözlem anındaki kapasiteyi içermez. Böylece hedef kapasitenin
# aynı satırdan girdiye sızması engellenir.
SOH_FEATURE_NAMES = [
    "voltage_mean",
    "voltage_std",
    "voltage_min",
    "temp_mean",
    "temp_max",
    "current_mean",
    "charge_time",
    "discharge_time",
    "ic_peak1_amp",
    "ic_peak1_pos",
    "ic_peak2_amp",
    "ic_area",
    "internal_resistance",
    "lower_soc",
    "upper_soc",
    "local_overpotential_mean",
    "local_overpotential_min",
    "sei_thickness",
    "loss_of_lithium_inventory",
    "temp_internal_max",
]

# Referans kapasitenin yalnızca geçmişten taşınmış hali RUL girdisidir.
RUL_FEATURE_NAMES = [
    *SOH_FEATURE_NAMES,
    "capacity_causal",
    "capacity_causal_missing",
    "cap_slope_5",
]
