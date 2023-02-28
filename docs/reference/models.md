# Clifford models

We provide exemplary 2D and 3D Clifford models as used in the paper.

All these modules are available for different algebras.

## 2D models

The following code snippet initializes a 2D Clifford ResNet.

```python
import torch.nn.functional as F

from cliffordlayers.models.models_2d import (
    CliffordNet2d,
    CliffordBasicBlock2d,
)

model = CliffordNet2d(
        g = [-1, -1],
        block = CliffordBasicBlock2d,
        num_blocks = [2, 2, 2, 2],
        in_channels = in_channels,
        out_channels = out_channels,
        hidden_channels = 32,
        activation = F.gelu,
        norm = True,
        rotation = False,
    )
```

The following code snippet initializes a 2D rotational Clifford ResNet.

```python
import torch.nn.functional as F

from cliffordlayers.models.models_2d import (
    CliffordNet2d,
    CliffordBasicBlock2d,
)

model = CliffordNet2d(
        g = [-1, -1],
        block = CliffordBasicBlock2d,
        num_blocks = [2, 2, 2, 2],
        in_channels = in_channels,
        out_channels = out_channels,
        hidden_channels = 32,
        activation = F.gelu,
        norm = True,
        rotation = True,
    )
```

The following code snippet initializes a 2D Clifford FNO.

```python
import torch.nn.functional as F

from cliffordlayers.models.utils import partialclass
from cliffordlayers.models.models_2d import (
    CliffordNet2d,
    CliffordFourierBasicBlock2d,
)

model = CliffordNet2d(
        g = [-1, -1],
        block = partialclass(
                "CliffordFourierBasicBlock2d", CliffordFourierBasicBlock2d, modes1=32, modes2=32
            ),
        num_blocks = [1, 1, 1, 1],
        in_channels = in_channels,
        out_channels = out_channels,
        hidden_channels = 32,
        activation = F.gelu,
        norm = False,
        rotation = False,
    )
```

::: cliffordlayers.models.models_2d

## 3D models

The following code snippet initializes a 3D Clifford FNO.

```python
import torch.nn.functional as F

from cliffordlayers.models.models_3d import (
    CliffordNet3d,
    CliffordFourierBasicBlock3d,
)
model = CliffordNet3d(
        g = [1, 1, 1],
        block = CliffordFourierBasicBlock3d,
        num_blocks = [1, 1, 1, 1],
        in_channels = 4,
        out_channels = 1,
        hidden_channels = 16,
        activation = F.gelu,
        norm = False,
    )
```

::: cliffordlayers.models.models_3d
