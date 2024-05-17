This is a self-contained repo for using temporal kernels parameterized by orthogonal polynomials. For example, it can be used as a drop-in replacement for convolutional layers (only supporting `nn.Conv3d` layers for now), where the last dimension (assumed to be temporal) will be parameterized by orthogonal polynomials up to a given degree.

```python
from model import OrthoPolyLayer

layer = OrthoPolyLayer(2, 8, kernel_size=(3, 3, 20), degrees=4)
```

The structured temporal kernels can also easily be resampled into different kernel sizes without needing to retrain the network.

```python
layer.resample(10)  # downsample the kernel size from 20 to 10
```
