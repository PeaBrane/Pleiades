# Pleiades

Paper: [Building Temporal Kernels with Orthogonal Polynomials](https://arxiv.org/abs/2405.12179)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/building-temporal-kernels-with-orthogonal/gesture-recognition-on-dvs128-gesture)](https://paperswithcode.com/sota/gesture-recognition-on-dvs128-gesture?p=building-temporal-kernels-with-orthogonal)

## Quickstart

This is a self-contained repo for using temporal kernels parameterized by orthogonal polynomials. For example, it can be used as a drop-in replacement for convolutional layers (only supporting `nn.Conv3d` layers for now), where the last dimension (assumed to be temporal) will be parameterized by orthogonal polynomials up to a given degree.

```python
from model import PleiadesLayer

layer = PleiadesLayer(2, 8, kernel_size=(3, 3, 20), degrees=4)
```

The structured temporal kernels can also easily be resampled into different kernel sizes without needing to retrain the network.

```python
layer.resample(10)  # downsample the kernel size from 20 to 10
```
