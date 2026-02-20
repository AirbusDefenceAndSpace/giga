### Model: Simple 3x3 Convolution
A basic 2D convolution with a $3 \times 3$ kernel

```python
import torch.nn as nn

class Model_Simple3x3Conv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)