# TriRep-Net

Code and demo video accompanying our manuscript submission (under review).'

## Contents

| File | Description |
|---|---|
| [`trirep_net.py`](trirep_net.py) | TriRep-Net model: a residual CNN stem over STFT inputs and twin 1-D convolutional branches over raw/FFT inputs, fused by a shared transformer encoder with modality embeddings and modality dropout. `build_trirep_net()` returns the tuned configuration (`TUNED_C`). |
| [`personalize_fusion_up.py`](personalize_fusion_up.py) | "Fusion-up" personalization: loads a trained checkpoint, freezes the low-level feature extractors (`stft_cnn`, `branch1`, `branch2`), and fine-tunes the fusion transformer and classification head on a small user-specific set. |
| [`media/demo.mp4`](demo.mp4) | Real-time demonstration of gesture recognition with the wearable sensor (≈40 s, 1080p). |

## Requirements

```bash
pip install -r requirements.txt
```

Python ≥ 3.9, PyTorch ≥ 2.0, NumPy.

## Usage

```python
import torch
from trirep_net import build_trirep_net

model = build_trirep_net(num_classes=5).eval()   # ~100.8k parameters

x_stft   = torch.randn(8, 16, 32, 32)   # (batch, 16, H, W); H and W divisible by 4
x_rawfft = torch.randn(8, 2, 2, 256)    # (batch, 2 branches, 2 channels, L); L divisible by 8
logits = model(x_stft, x_rawfft)         # (batch, num_classes)
```

Personalizing a trained model on a new user's data:

```python
from personalize_fusion_up import load_tunedC, fusion_up, accuracy

model = load_tunedC("checkpoint.pt", num_classes=5)
model = fusion_up(model, x_stft_user, x_rawfft_user, y_user, lr=1e-4, epochs=100, batch_size=16)
print(accuracy(model, x_stft_test, x_rawfft_test, y_test))
```

Only the transformer fusion layers and the classification head are updated (~90.3k trainable parameters). The BatchNorm layers of the frozen STFT stem stay in eval mode.

## Demo video

The video shows the trained classifier recognising hand gestures live from the wearable sensor. Open [`media/demo.mp4`](demo.mp4) to play it in the browser, or download it.
