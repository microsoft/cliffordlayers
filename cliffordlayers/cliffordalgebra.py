from typing import Tuple
import torch

from .basisbladeorder import ShortLexBasisBladeOrder, construct_gmt


class CliffordAlgebra:
    def __init__(self, metric):
        self.metric = torch.as_tensor(metric, dtype=torch.float32)
        self.num_bases = len(metric)
        self.bbo = ShortLexBasisBladeOrder(self.num_bases)
        self.dim = len(self.metric)
        self.n_blades = len(self.bbo.grades)
        self.cayley = construct_gmt(self.bbo.index_to_bitmap, self.bbo.bitmap_to_index, self.metric).to_dense()

    def to(self, device: torch.device):
        self.bbo.grades = self.bbo.grades.to(device)
        self.metric = self.metric.to(device)
        self.cayley = self.cayley.to(device)
        return self

    def geometric_product(self, a, b):
        return torch.einsum(
            "bfi,ijk,bfk->bfj",
            a.view(len(a), -1, len(self.cayley)),
            self.cayley,
            b.view(len(b), -1, len(self.cayley)),
        ).view(*a.size())

    def reverse(self, mv, blades=None):
        """Perform the reversion operation on multivectors, an operation specific to geometric algebra.

        In Geometric Algebra, the reverse of a multivector is formed by reversing the order of the vectors in each blade.

        Args:
            mv (torch.Tensor): Input multivectors.
            blades (Union[tuple, list, torch.Tensor], optional): Specify which blades are present in the multivector.

        Returns:
            torch.Tensor: The reversed multivector.
        """
        grades = self.bbo.grades.to(mv.device)
        if blades is not None:
            grades = grades[torch.as_tensor(blades, dtype=int)]
        signs = torch.pow(-1, torch.floor(grades * (grades - 1) / 2))
        return signs * mv.clone()

    def embed(self, tensor: torch.Tensor, tensor_index: Tuple[int]) -> torch.Tensor:
        """
        Embeds the input tensor into a multivector.

        This method takes a tensor and embeds it into a multivector, with the tensor elements assigned to the basis blades
        specified by the tensor_index.

        Args:
            tensor (torch.Tensor): The input tensor to be embedded.
            tensor_index (tuple[int]): A tuple of integers specifying the blades where tensor elements are to be placed.

        Returns:
            torch.Tensor: The multivector embedding of the input tensor.

        Raises:
            AssertionError: If the last dimension of tensor does not match the length of tensor_index.
        """
        assert tensor.size(-1) == len(tensor_index)
        blades = tuple(range(self.n_blades))
        mv = torch.zeros(*tensor.shape[:-1], len(blades), device=tensor.device)
        tensor_index = torch.as_tensor(tensor_index, device=tensor.device, dtype=int)
        shape = *(1,) * (mv.dim() - 1), -1
        tensor_index = tensor_index.view(shape).repeat(*tensor.size()[:-1], 1)
        mv = torch.scatter(mv, -1, tensor_index, tensor)
        return mv

    def get(self, mv: torch.Tensor, blade_index: Tuple[int]) -> torch.Tensor:
        """
        Extracts the components of a multivector corresponding to the specified blade indices.

        This method takes a multivector and a tuple of blade indices, and returns the components of the multivector
        that correspond to those indices.

        Args:
            mv (torch.Tensor): The input multivector.
            blade_index (tuple[int]): A tuple of integers specifying the blades to be extracted.

        Returns:
            torch.Tensor: The tensor of components corresponding to the specified blade indices.
        """
        blade_index = tuple(blade_index)
        return mv[..., blade_index]

    def mag2(self, mv):
        return self.geometric_product(self.reverse(mv), mv)

    def norm(self, mv):
        mag2 = self.mag2(mv)[..., :1]
        mag2.abs_().sqrt_()
        return mag2

    def sandwich(self, a, b):
        """aba'"""
        return self.geometric_product(self.geometric_product(a, b), self.reverse(a))
