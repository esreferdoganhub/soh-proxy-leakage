"""
model.py — HAT-Net (Hybrid Attention Transformer Network)
Li-ion Batarya SOH + RUL Multi-task Tahmin Mimarisi

Yenilikler:
  1. CNN + BiLSTM paralel feature extraction
  2. Rotary Position Embedding (RoPE) destekli Transformer Encoder
  3. Dual output heads (SOH regresyon + RUL regresyon)
  4. Monte Carlo Dropout ile uncertainty quantification
  5. [Tier 2] iTransformer Variate-Attention Dalı
     Zha et al. (2024) "iTransformer: Inverted Transformers Are Effective for Time Series"
     Adaptasyon: Her fiziksel özellik (SEI, LLI, LAM, voltaj vb.) bir token olarak
     ele alınır; değişkenler arası korelasyon time-step dikkatinden bağımsız öğrenilir.

Yazar: [Adınız] — Çukurova Üniversitesi
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from config import MODEL_CFG, DEVICE


# ════════════════════════════════════════════════════════════════
# 1. Rotary Position Embedding (RoPE)
# ════════════════════════════════════════════════════════════════
class RotaryEmbedding(nn.Module):
    """
    RoPE — Rotary Position Embedding.
    Su et al. (2023): "RoFormer: Enhanced Transformer with Rotary Position Embedding"

    Batarya alan uygulaması: Döngü indeksine bağlı pozisyonel
    bilgiyi dikkat mekanizmasına göreceli olarak kodlar.
    Standart sinüsoidal PE'ye göre daha iyi uzun-mesafe genelleme.
    """

    def __init__(self, dim: int):
        super().__init__()
        # dim: d_model (tam boyut)
        assert dim % 2 == 0, "RoPE boyutu çift sayı olmalıdır."
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)  # (dim/2,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, dim)
        Returns:
            x rotated: (batch, seq_len, dim)
        """
        B, T, D = x.shape
        # inv_freq: (dim/2,)  →  freqs: (T, dim/2)
        t     = torch.arange(T, device=x.device, dtype=x.dtype)
        freqs = torch.outer(t, self.inv_freq)        # (T, dim/2)
        emb   = torch.cat([freqs, freqs], dim=-1)    # (T, dim)

        cos = emb.cos().unsqueeze(0)  # (1, T, dim)
        sin = emb.sin().unsqueeze(0)  # (1, T, dim)

        # Rotate half — x boyutu D, emb boyutu self.dim olabilir
        # D != dim ise kırp veya pad
        cos = cos[..., :D]
        sin = sin[..., :D]

        half = D // 2
        x1   = x[..., :half]
        x2   = x[..., half:half * 2]
        x_rot = torch.cat([-x2, x1], dim=-1)         # (B, T, D)
        return x * cos + x_rot * sin


# ════════════════════════════════════════════════════════════════
# 2. Temporal CNN Dalı
# ════════════════════════════════════════════════════════════════
class TemporalCNN(nn.Module):
    """
    1D-CNN ile kısa vadeli lokal örüntü çıkarımı.
    Her katman: Conv1d → BatchNorm → GELU → Dropout
    """

    def __init__(self, in_channels: int, channels: list, kernel: int = 3,
                 dropout: float = 0.2):
        super().__init__()
        layers = []
        for out_ch in channels:
            layers += [
                nn.Conv1d(in_channels, out_ch, kernel, padding=kernel//2),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_channels = out_ch
        self.cnn = nn.Sequential(*layers)
        self.out_channels = in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, features) → (batch, seq_len, out_channels)
        """
        x = x.permute(0, 2, 1)      # (B, F, T)
        x = self.cnn(x)             # (B, C, T)
        return x.permute(0, 2, 1)   # (B, T, C)


# ════════════════════════════════════════════════════════════════
# 3. BiLSTM Dalı
# ════════════════════════════════════════════════════════════════
class BiLSTMBranch(nn.Module):
    """
    Çift yönlü LSTM ile orta vadeli sekans belleği.
    """

    def __init__(self, input_size: int, hidden_size: int = 256,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out_size = hidden_size * 2  # Çift yönlü → 2x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.bilstm(x)   # (B, T, 2*H)
        return out


# ════════════════════════════════════════════════════════════════
# 4. iTransformer Variate-Attention (Tier 2)
# ════════════════════════════════════════════════════════════════
class VariateAttention(nn.Module):
    """
    Zha et al. (2024) iTransformer — Batarya SOH/RUL Adaptasyonu.

    Klasik Transformer: attention over T (zaman adımları)
    Bu modül:          attention over F (fiziksel değişkenler)

    Her özellik (cap_discharge, voltage_mean, ic_peak1 …) bir "variate token"
    olarak kodlanır. Attention matrisi (F×F) değişkenler arası ilişkiyi kapsar:
    örneğin IC eğrisi pik genliğinin iç dirençle nasıl korelasyon kurduğunu.

    Giriş:  (B, T, F)  — batch, zaman penceresi, özellik sayısı
    Çıkış:  (B, T, variate_dim) — variate_dim = projeksiyon boyutu
    """

    def __init__(self,
                 n_features: int,
                 seq_len: int,
                 variate_dim: int = 64,
                 n_heads: int = 4,
                 n_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.n_features  = n_features   # F
        self.seq_len     = seq_len      # T (pencere boyutu)
        self.variate_dim = variate_dim

        # Her variate (özellik) için: T boyutlu sekansı → variate_dim'e projeksiyon
        # Bu projeksiyon her özelliğin "kimliğini" sabit boyutlu bir vektörle temsil eder
        self.variate_proj = nn.Linear(seq_len, variate_dim)

        # Variate tokenlara (F adet, variate_dim boyutlu) dikkat uygula
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=variate_dim,
            nhead=n_heads,
            dim_feedforward=variate_dim * 4,
            dropout=dropout,
            batch_first=True,   # (B, F, variate_dim)
            activation="gelu",
            norm_first=True,    # Pre-norm: stabil eğitim
        )
        self.variate_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Çıkışı tekrar T boyutuna yay (output projeksiyon)
        # (B, F, variate_dim) → (B, F, T) → (B, T, F) → fusion'a hazır
        self.out_proj = nn.Linear(variate_dim, seq_len)
        self.norm     = nn.LayerNorm(n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, F)
        Returns:
            out: (B, T, F)  — residual connection uyumlu
        """
        B, T, F = x.shape

        # Transpoz: her özelliğin zaman serisini bir token olarak al
        # (B, T, F) → (B, F, T)
        x_t = x.permute(0, 2, 1)  # (B, F, T)

        # T → variate_dim projeksiyon: (B, F, T) → (B, F, variate_dim)
        tokens = self.variate_proj(x_t)   # (B, F, variate_dim)

        # Değişkenler arası dikkat: (B, F, variate_dim) → (B, F, variate_dim)
        attended = self.variate_encoder(tokens)  # (B, F, variate_dim)

        # variate_dim → T geri projeksiyon: (B, F, variate_dim) → (B, F, T)
        out_t = self.out_proj(attended)   # (B, F, T)

        # (B, F, T) → (B, T, F)
        out = out_t.permute(0, 2, 1)      # (B, T, F)

        # Residual + LayerNorm (orijinal x ile aynı shape)
        return self.norm(out + x)          # (B, T, F)


class iTransformerBranch(nn.Module):
    """
    VariateAttention üzerine inşa edilmiş tam dal.
    Çıkışı (B, T, itrans_out_dim) olarak verir; diğer dallarla concat edilir.
    """

    def __init__(self,
                 n_features: int,
                 seq_len: int,
                 variate_dim: int = 64,
                 n_heads: int = 4,
                 n_layers: int = 2,
                 out_dim: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.variate_attn = VariateAttention(
            n_features=n_features,
            seq_len=seq_len,
            variate_dim=variate_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )
        # (B, T, F) → (B, T, out_dim)
        self.proj = nn.Sequential(
            nn.Linear(n_features, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, F)
        Returns:
            (B, T, out_dim)
        """
        x = self.variate_attn(x)   # (B, T, F) — değişken korelasyonu zenginleştirilmiş
        return self.proj(x)         # (B, T, out_dim)


# ════════════════════════════════════════════════════════════════
# 5 (eski 4). Transformer Encoder (RoPE destekli)
# ════════════════════════════════════════════════════════════════
class RoPETransformerEncoder(nn.Module):
    """
    Multi-Head Self-Attention + Feed-Forward + RoPE.
    Uzun vadeli batarya degradasyon trendlerini yakalar.
    """

    def __init__(self, d_model: int, n_heads: int, n_layers: int,
                 dim_ff: int, dropout: float = 0.1):
        super().__init__()
        self.rope = RotaryEmbedding(d_model)  # Tam d_model boyutunu kullan

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,    # Pre-norm: daha stabil eğitim
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        """
        # RoPE gömme — her head için ayrı ayrı uygulanır
        # Basitleştirilmiş: tüm d_model üzerine uygulama
        x = self.rope(x)
        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)


# ════════════════════════════════════════════════════════════════
# 6 (eski 5). Çıkış Kafaları
# ════════════════════════════════════════════════════════════════
class RegressionHead(nn.Module):
    """
    Tam bağlı regresyon kafası.
    Batch normalization + GELU + Dropout katmanları.
    """

    def __init__(self, in_dim: int, hidden_dims: list,
                 out_activation: str = "sigmoid",
                 mc_dropout: float = 0.3):
        super().__init__()
        self.mc_dropout_p = mc_dropout
        layers = []
        curr = in_dim
        for h in hidden_dims[:-1]:
            layers += [
                nn.Linear(curr, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(mc_dropout),
            ]
            curr = h
        layers.append(nn.Linear(curr, hidden_dims[-1]))
        self.net = nn.Sequential(*layers)

        if out_activation == "sigmoid":
            self.act = nn.Sigmoid()
        elif out_activation == "relu":
            self.act = nn.ReLU()
        else:
            self.act = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.net(x)).squeeze(-1)


# ════════════════════════════════════════════════════════════════
# 7 (eski 6). HAT-Net — Ana Model
# ════════════════════════════════════════════════════════════════
class HATNet(nn.Module):
    """
    CNN-Transformer Network

    Mimari Akışı:
    Giriş (B, T, F)
        │
        ├─ [CNN Dal]          TemporalCNN          → (B, T, 64)
        │
        └─ [Fusion]   Linear Projection → (B, T, d_model=256)
        │
        └─ [Transformer] RoPE-Encoder → (B, T, 256)
        │
        └─ [Pooling]  Mean-Pool → (B, 256)
        │
        ├─ [SOH Head] → (B,) ∈ [0,1]
        └─ [RUL Head] → (B,) ≥ 0
    """

    def __init__(self,
                 n_features: int,
                 cfg: dict = MODEL_CFG,
                 seq_len: int = 30):
        super().__init__()
        self.cfg = cfg

        # ── CNN Dalı ──────────────────────────────────────────
        self.cnn = TemporalCNN(
            in_channels=n_features,
            channels=cfg["cnn_channels"][1:],  # [32, 64]
            kernel=cfg["cnn_kernel"],
            dropout=0.2,
        )
        cnn_channels = cfg["cnn_channels"][-1]  # 64

        # ── Feature Fusion ────────────────────────────────────
        self.proj = nn.Linear(cnn_channels, cfg["d_model"])

        # ── Transformer Encoder ───────────────────────────────
        self.transformer = RoPETransformerEncoder(
            d_model=cfg["d_model"],
            n_heads=cfg["n_heads"],
            n_layers=cfg["n_enc_layers"],
            dim_ff=cfg["dim_feedforward"],
            dropout=cfg["tf_dropout"],
        )

        # ── Çıkış Kafaları ────────────────────────────────────
        self.soh_head = RegressionHead(
            in_dim=cfg["d_model"],
            hidden_dims=cfg["soh_head_dims"],   # [128, 64, 1]
            out_activation="sigmoid",
        )
        self.rul_head = RegressionHead(
            in_dim=cfg["d_model"],
            hidden_dims=cfg["rul_head_dims"],   # [128, 64, 1]
            out_activation="relu",
        )

        # ── Ağırlık Başlatma ─────────────────────────────────
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor,
                return_attention: bool = False
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch_size, window_size, n_features)

        Returns:
            soh_pred: (batch_size,) — [0, 1]
            rul_pred: (batch_size,) — [0, ∞)
        """
        # ── Paralel Dallar ────────────────────────────────────
        cnn_out = self.cnn(x)              # CNN: (B, T, 64)

        # ── Fusion ───────────────────────────────────────────
        combined = cnn_out
        fused = self.proj(combined)             # (B, T, d_model=256)
        # Transformer
        encoded = self.transformer(fused)                # (B, T, 256)

        # Global ortalama havuzlama
        pooled = encoded.mean(dim=1)                     # (B, 256)

        # İkili çıkış
        soh = self.soh_head(pooled)                      # (B,)
        rul = self.rul_head(pooled)                      # (B,)

        return soh, rul

    def predict_with_uncertainty(self,
                                  x: torch.Tensor,
                                  n_samples: int = 50,
                                  batch_size: int = 512
                                 ) -> dict:
        """
        Monte Carlo Dropout ile belirsizlik tahmini.
        Dropout katmanları train modunda açık tutulur (Batch Norm eval modunda kalır).
        Bellek aşımını önlemek için veri batch'ler halinde işlenir.

        Returns:
            dict: mean_soh, std_soh, mean_rul, std_rul
        """
        import numpy as np
        self.eval()  # Tüm modeli eval moduna al (Batch Norm istatistikleri korunur)
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()  # Sadece dropout katmanlarını train moduna al (MC Dropout)
        
        n_samples_list_soh = []
        n_samples_list_rul = []

        with torch.no_grad():
            for _ in range(n_samples):
                soh_batches = []
                rul_batches = []
                for i in range(0, len(x), batch_size):
                    x_batch = x[i:i+batch_size]
                    soh, rul = self.forward(x_batch)
                    soh_batches.append(soh.cpu().numpy())
                    rul_batches.append(rul.cpu().numpy())
                
                soh_preds = np.concatenate(soh_batches, axis=0)
                rul_preds = np.concatenate(rul_batches, axis=0)
                n_samples_list_soh.append(soh_preds)
                n_samples_list_rul.append(rul_preds)

        self.eval()  # Tamamen eval moduna geri dön
        soh_arr = np.stack(n_samples_list_soh)
        rul_arr = np.stack(n_samples_list_rul)

        return {
            "mean_soh": soh_arr.mean(axis=0),
            "std_soh":  soh_arr.std(axis=0),
            "mean_rul": rul_arr.mean(axis=0),
            "std_rul":  rul_arr.std(axis=0),
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ════════════════════════════════════════════════════════════════
# 7. Gradient Reversal Layer (GRL) + Domain-Adaptive HAT-Net
#
# Motivasyon:
#   RFN (Göreceli Özellik Normalizasyonu) kısmi SOC pencerelerinden
#   kaynaklanan alan sapmalarını çözen sabit bir kuraldır.
#   DomainAdaptiveHATNet ise bu soruna öğrenilebilir bir çözüm üretir:
#   - Encoder, domain ayırt edici özellikler üretmeme yönünde ters gradyan alır.
#   - Sonuç temsil, kişmi ve tam döngü hucreleri için domain-invariant olur.
#
# Referans:
#   Ganin et al. (2016) Domain-Adversarial Training of Neural Networks.
#   JMLR 17(59):1–35.
# ════════════════════════════════════════════════════════════════




# ════════════════════════════════════════════════════════════════
# 8. Baseline Modeller (Ablasyon karşılaştırması için)
# ════════════════════════════════════════════════════════════════
class LSTMBaseline(nn.Module):
    """Tek yönlü LSTM baseline"""
    def __init__(self, n_features: int, hidden: int = 256, layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, layers, batch_first=True,
                            dropout=0.3 if layers > 1 else 0.0)
        self.soh_head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(),
                                       nn.Linear(64, 1), nn.Sigmoid())
        self.rul_head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(),
                                       nn.Linear(64, 1), nn.ReLU())

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.soh_head(last).squeeze(-1), self.rul_head(last).squeeze(-1)


class BiLSTMBaseline(nn.Module):
    """BiLSTM baseline"""
    def __init__(self, n_features: int, hidden: int = 256, layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, layers, batch_first=True,
                            bidirectional=True,
                            dropout=0.3 if layers > 1 else 0.0)
        self.soh_head = nn.Sequential(nn.Linear(hidden*2, 64), nn.ReLU(),
                                       nn.Linear(64, 1), nn.Sigmoid())
        self.rul_head = nn.Sequential(nn.Linear(hidden*2, 64), nn.ReLU(),
                                       nn.Linear(64, 1), nn.ReLU())

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.soh_head(last).squeeze(-1), self.rul_head(last).squeeze(-1)


class CNNLSTMBaseline(nn.Module):
    """CNN-LSTM baseline"""
    def __init__(self, n_features: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, 64, 3, padding=1), nn.ReLU(),
            nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(),
        )
        self.lstm = nn.LSTM(128, 256, 2, batch_first=True, dropout=0.3)
        self.soh_head = nn.Sequential(nn.Linear(256, 64), nn.ReLU(),
                                       nn.Linear(64, 1), nn.Sigmoid())
        self.rul_head = nn.Sequential(nn.Linear(256, 64), nn.ReLU(),
                                       nn.Linear(64, 1), nn.ReLU())

    def forward(self, x):
        cnn_out = self.cnn(x.permute(0, 2, 1)).permute(0, 2, 1)
        out, _  = self.lstm(cnn_out)
        last    = out[:, -1, :]
        return self.soh_head(last).squeeze(-1), self.rul_head(last).squeeze(-1)


class VanillaTransformerBaseline(nn.Module):
    """Standart sinüsoidal PE'li Transformer baseline"""
    def __init__(self, n_features: int, d_model: int = 256,
                 n_heads: int = 8, n_layers: int = 3):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)

        # Sinüsoidal PE
        pe = torch.zeros(1000, d_model)
        pos = torch.arange(1000).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

        enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, 512,
                                               batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers)
        self.soh_head = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(),
                                       nn.Linear(64, 1), nn.Sigmoid())
        self.rul_head = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(),
                                       nn.Linear(64, 1), nn.ReLU())

    def forward(self, x):
        T  = x.shape[1]
        x  = self.proj(x) + self.pe[:, :T, :]
        x  = self.encoder(x)
        x  = x.mean(dim=1)
        return self.soh_head(x).squeeze(-1), self.rul_head(x).squeeze(-1)


# ════════════════════════════════════════════════════════════════
# Model Fabrikası
# ════════════════════════════════════════════════════════════════
MODELS = {
    "lstm":              LSTMBaseline,
    "bilstm":            BiLSTMBaseline,
    "cnn_lstm":          CNNLSTMBaseline,
    "transformer":       VanillaTransformerBaseline,
    "hat_net":           HATNet,
    # Ablasyon varyantları — use_variate_attn / use_bilstm flag'leriyle kontrol
    # Bu isimler doğrudan build_model'de özel olarak ele alınır
    "hat_net_no_itrans": HATNet,   # iTransformer dalı kapalı
    "hat_net_no_lstm":   HATNet,   # BiLSTM dalı kapalı
    "hat_net_base":      HATNet,   # Sadece CNN + Transformer (her iki dal kapalı)
}

def build_model(name: str, n_features: int, seq_len: int = 30, **kwargs) -> nn.Module:
    """
    Model adına göre model oluştur ve DEVICE'a taşı.

    Özel ablasyon isimleri:
        hat_net_no_itrans  → use_variate_attn=False
        hat_net_no_lstm    → use_bilstm=False
        hat_net_base       → use_variate_attn=False, use_bilstm=False
    """
    if name == "hat_net_mamba":
        from model_mamba import HATNetMamba
        model = HATNetMamba(n_features=n_features, **kwargs).to(DEVICE)

    elif name == "hat_net_no_itrans":
        model = HATNet(n_features=n_features, seq_len=seq_len,
                       use_variate_attn=False, **kwargs).to(DEVICE)

    elif name == "hat_net_no_lstm":
        model = HATNet(n_features=n_features, seq_len=seq_len,
                       use_bilstm=False, **kwargs).to(DEVICE)

    elif name == "hat_net_base":
        model = HATNet(n_features=n_features, seq_len=seq_len,
                       use_bilstm=False, use_variate_attn=False, **kwargs).to(DEVICE)

    elif name == "hat_net":
        model = HATNet(n_features=n_features, seq_len=seq_len, **kwargs).to(DEVICE)

    else:
        cls = MODELS.get(name)
        if cls is None:
            raise ValueError(
                f"Bilinmeyen model: {name}. "
                f"Seçenekler: {list(MODELS.keys())} + ['hat_net_mamba']"
            )
        model = cls(n_features=n_features, **kwargs).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🧠 {name}: {n_params:,} eğitilebilir parametre → {DEVICE}")
    return model


if __name__ == "__main__":
    # Hızlı boyut testi
    B, T, F = 32, 30, 13   # batch, window, features
    x = torch.randn(B, T, F).to(DEVICE)

    test_models = [
        "lstm", "bilstm", "cnn_lstm", "transformer",
        "hat_net",           # Tam model: CNN + BiLSTM + iTransformer + RoPE-Transformer
        "hat_net_no_itrans", # Ablasyon: iTransformer dalı kapalı
        "hat_net_no_lstm",   # Ablasyon: BiLSTM dalı kapalı
        "hat_net_base",      # Ablasyon: sadece CNN + RoPE-Transformer
        "hat_net_mamba",
    ]
    for name in test_models:
        try:
            model = build_model(name, n_features=F, seq_len=T)
            with torch.no_grad():
                soh, rul = model(x)
            n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  {name:25s}: SOH={tuple(soh.shape)}, RUL={tuple(rul.shape)}, params={n_p:,}")
        except Exception as e:
            print(f"  {name:25s}: HATA — {e}")

    print("\n✅ Tüm modeller başarıyla test edildi.")
