import random

import numpy as np
import torch
import torch.nn as nn

from trirep_net import build_trirep_net

FROZEN = ('stft_cnn', "branch1", "branch2")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_tunedC(checkpoint, num_classes=5, device="cpu"):
    model = build_trirep_net(num_classes, modality_dropout=0.0)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return model.to(device)


def freeze_low_level(model):
    for name, p in model.named_parameters():
        p.requires_grad_(not name.startswith(FROZEN))


def keep_frozen_bn_in_eval(model):
    for m in model.stft_cnn.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


@torch.no_grad()
def accuracy(model, x_stft, x_rawfft, y, batch_size=256):
    model.eval()
    correct = 0
    for i in range(0, len(y), batch_size):
        pred = model(x_stft[i:i + batch_size], x_rawfft[i:i + batch_size]).argmax(1)
        correct += int((pred == y[i:i + batch_size]).sum())
    return correct / len(y)


def fusion_up(model, x_stft, x_rawfft, y, lr=1e-4, epochs=100, batch_size=16, seed=44):
    set_seed(seed)
    freeze_low_level(model)
    trainable_parmas = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_parmas, lr=lr)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    n = len(y)
    bs = min(batch_size, n)
    g = torch.Generator(device=y.device)
    g.manual_seed(seed)
    for _ in range(epochs):
        model.train()
        keep_frozen_bn_in_eval(model)
        perm = torch.randperm(n, device=y.device, generator=g)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            crit(model(x_stft[b], x_rawfft[b]), y[b]).backward()
            opt.step()
    return model.eval()
