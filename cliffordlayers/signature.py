# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Union

import torch


class CliffordSignature:
    def __init__(self, g: Union[tuple, list, torch.Tensor]):
        super().__init__()
        self.g = self._g_tensor(g)
        self.dim = self.g.numel()
        if self.dim == 1:
            self.n_blades = 2
        elif self.dim == 2:
            self.n_blades = 4
        elif self.dim == 3:
            self.n_blades = 8
        else:
            raise NotImplementedError(f"Wrong Clifford signature.")

    def _g_tensor(self, g: Union[tuple, list, torch.Tensor]) -> torch.Tensor:
        """Convert Clifford signature to tensor.

        Args:
            g (Union[tuple, list, torch.Tensor]): Clifford signature.

        Raises:
            ValueError: Unknown metric.

        Returns:
            torch.Tensor: Clifford signature as torch.Tensor.
        """
        if type(g) in (tuple, list):
            g = torch.as_tensor(g, dtype=torch.float32)
        elif isinstance(g, torch.Tensor):
            pass
        else:
            raise ValueError(f"Unknown signature.")

        if not torch.any(abs(g) == 1.0):
            raise ValueError(f"Clifford signature should have at least one element as 1.")
        return g
