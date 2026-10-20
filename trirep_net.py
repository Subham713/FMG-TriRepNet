import math
import torch
import torch.nn as nn


TUNED_C = dict(d_model=32, n_tx=2, heads=4, mlp_ratio=4.0, head_hidden=64,
               dropout=0.1042183620548366, tx_dropout=0.0886633389945999,
               modality_dropout=0.0867571133960082)


class ConvBNReLU(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ResidualCNNBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block1 = ConvBNReLU(ch)
        self.block2 = ConvBNReLU(ch)

    def forward(self, x):
        b1 = self.block1(x)
        return self.block2(b1) + b1


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, heads, key_dim):
        super().__init__()
        self.heads = heads
        self.key_dim = key_dim
        inner = heads * key_dim
        self.wq = nn.Linear(d_model, inner)
        self.wk = nn.Linear(d_model, inner)
        self.wv = nn.Linear(d_model, inner)
        self.wo = nn.Linear(inner, d_model)
        self.scale = key_dim ** -0.5

    def forward(self, x):
        b, n, _ = x.shape
        q = self.wq(x).view(b, n, self.heads, self.key_dim).transpose(1, 2)
        k = self.wk(x).view(b, n, self.heads, self.key_dim).transpose(1, 2)
        v = self.wv(x).view(b, n, self.heads, self.key_dim).transpose(1, 2)
        attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, n, self.heads * self.key_dim)
        return self.wo(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, heads, key_dim, dropout, mlp_ratio):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, heads, key_dim)
        self.norm2 = nn.LayerNorm(d_model)
        hiden = max(int(round(d_model * mlp_ratio)), 1)
        self.fc1 = nn.Linear(d_model, hiden)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hiden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.mha(self.norm1(x)))
        h = self.fc2(self.act(self.fc1(self.norm2(x))))
        return x + self.dropout(h)


class LN1d(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.ln = nn.LayerNorm(ch)

    def forward(self, x):
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


class Branch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(2, 16, kernel_size=7, padding=3)
        self.ln1 = LN1d(16)
        self.act1 = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.ln2 = LN1d(32)
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.act1(self.ln1(self.conv1(x)))
        x = self.pool(x)
        return self.act2(self.ln2(self.conv2(x)))


class TriRepNet(nn.Module):
    def __init__(self, num_classes=5, d_model=32, n_tx=2, heads=4, head_hidden=64,
                 dropout=0.1, tx_dropout=0.1, modality_dropout=0.1, mlp_ratio=4.0):
        super().__init__()
        self.modality_dropout = modality_dropout
        self.stft_stem_ch = 16

        self.stft_cnn = ResidualCNNBlock(self.stft_stem_ch)
        self.stft_patchify = nn.Conv2d(self.stft_stem_ch, d_model, kernel_size=4, stride=4)
        self.stft_norm = nn.LayerNorm(d_model)

        self.branch1 = Branch()
        self.branch2 = Branch()
        self.fuse_conv = nn.Conv1d(64, 64, kernel_size=5, padding=2)
        self.fuse_ln = LN1d(64)
        self.fuse_act = nn.ReLU(inplace=True)
        self.rf_patchify = nn.Conv1d(64, d_model, kernel_size=4, stride=4)
        self.rf_norm = nn.LayerNorm(d_model)

        self.mod_emb = nn.Parameter(torch.zeros(2, d_model))
        nn.init.normal_(self.mod_emb, std=0.02)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, heads, d_model, tx_dropout, mlp_ratio)
                                     for _ in range(n_tx)])

        self.head_drop = nn.Dropout(dropout)
        self.fc1 = nn.Linear(d_model, head_hidden)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(head_hidden, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def tokens_stft(self, x):
        b = x.shape[0]
        x = x.reshape(b, self.stft_stem_ch, x.shape[-2], x.shape[-1])
        x = self.stft_patchify(self.stft_cnn(x))
        bb, d, h, w = x.shape
        return self.stft_norm(x.reshape(bb, d, h * w).permute(0, 2, 1))

    def tokens_rawfft(self, x):
        f = torch.cat([self.branch1(x[:, 0]), self.branch2(x[:, 1])], dim=1)
        f = self.fuse_act(self.fuse_ln(self.fuse_conv(f)))
        f = self.rf_patchify(f)
        return self.rf_norm(f.permute(0, 2, 1))

    def _modality_mask(self, b, n_a, n_b, device):
        if not self.training or self.modality_dropout <= 0:
            return torch.ones(b, n_a + n_b, 1, device=device)
        p = self.modality_dropout
        drop_a = torch.rand(b, device=device) < p
        drop_b = torch.rand(b, device=device) < p
        both = drop_a & drop_b
        coin = torch.rand(b, device=device) < 0.5
        drop_a = drop_a & ~(both & coin)
        drop_b = drop_b & ~(both & ~coin)
        keep_a = (~drop_a).float().view(b, 1, 1).expand(b, n_a, 1)
        keep_b = (~drop_b).float().view(b, 1, 1).expand(b, n_b, 1)
        return torch.cat([keep_a, keep_b], dim=1)

    def forward(self, x_stft, x_rawfft):
        tok_a = self.tokens_stft(x_stft) + self.mod_emb[0]
        tok_b = self.tokens_rawfft(x_rawfft) + self.mod_emb[1]
        tokens = torch.cat([tok_a, tok_b], dim=1)

        mask = self._modality_mask(tokens.shape[0], tok_a.shape[1], tok_b.shape[1], tokens.device)
        tokens = tokens * mask
        for block in self.blocks:
            tokens = block(tokens)

        pooled = (tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        h = self.act(self.fc1(self.head_drop(pooled)))
        return self.fc2(h)


def build_trirep_net(num_classes=5, **overrides):
    cfg = dict(TUNED_C)
    cfg.update(overrides)
    return TriRepNet(num_classes=num_classes, **cfg)
