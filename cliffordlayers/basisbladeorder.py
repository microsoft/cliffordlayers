"""Inspired by https://github.com/pygae/clifford"""
import functools
import itertools
import operator

import torch


# copied from the itertools docs
def _powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s) + 1))


class ShortLexBasisBladeOrder:
    def __init__(self, n_vectors):
        self.index_to_bitmap = torch.empty(2**n_vectors, dtype=int)
        self.grades = torch.empty(2**n_vectors, dtype=int)
        self.bitmap_to_index = torch.empty(2**n_vectors, dtype=int)

        for i, t in enumerate(_powerset([1 << i for i in range(n_vectors)])):
            bitmap = functools.reduce(operator.or_, t, 0)
            self.index_to_bitmap[i] = bitmap
            self.grades[i] = len(t)
            self.bitmap_to_index[bitmap] = i
            del t  # enables an optimization inside itertools.combinations


def set_bit_indices(x: int):
    """Iterate over the indices of bits set to 1 in `x`, in ascending order"""
    n = 0
    while x > 0:
        if x & 1:
            yield n
        x = x >> 1
        n = n + 1


def count_set_bits(bitmap: int) -> int:
    """Counts the number of bits set to 1 in bitmap"""
    count = 0
    for i in set_bit_indices(bitmap):
        count += 1
    return count


def canonical_reordering_sign_euclidean(bitmap_a, bitmap_b):
    """
    Computes the sign for the product of bitmap_a and bitmap_b
    assuming a euclidean metric
    """
    a = bitmap_a >> 1
    sum_value = 0
    while a != 0:
        sum_value = sum_value + count_set_bits(a & bitmap_b)
        a = a >> 1
    if (sum_value & 1) == 0:
        return 1
    else:
        return -1


def canonical_reordering_sign(bitmap_a, bitmap_b, metric):
    """
    Computes the sign for the product of bitmap_a and bitmap_b
    given the supplied metric
    """
    bitmap = bitmap_a & bitmap_b
    output_sign = canonical_reordering_sign_euclidean(bitmap_a, bitmap_b)
    i = 0
    while bitmap != 0:
        if (bitmap & 1) != 0:
            output_sign *= metric[i]
        i = i + 1
        bitmap = bitmap >> 1
    return output_sign


def gmt_element(bitmap_a, bitmap_b, sig_array):
    """
    Element of the geometric multiplication table given blades a, b.
    The implementation used here is described in :cite:`ga4cs` chapter 19.
    """
    output_sign = canonical_reordering_sign(bitmap_a, bitmap_b, sig_array)
    output_bitmap = bitmap_a ^ bitmap_b
    return output_bitmap, output_sign


def construct_gmt(index_to_bitmap, bitmap_to_index, signature):
    n = len(index_to_bitmap)
    array_length = int(n * n)
    coords = torch.zeros((3, array_length), dtype=torch.uint8)
    k_list = coords[0, :]
    l_list = coords[1, :]
    m_list = coords[2, :]

    # use as small a type as possible to minimize type promotion
    mult_table_vals = torch.zeros(array_length)

    for i in range(n):
        bitmap_i = index_to_bitmap[i]

        for j in range(n):
            bitmap_j = index_to_bitmap[j]
            bitmap_v, mul = gmt_element(bitmap_i, bitmap_j, signature)
            v = bitmap_to_index[bitmap_v]

            list_ind = i * n + j
            k_list[list_ind] = i
            l_list[list_ind] = v
            m_list[list_ind] = j

            mult_table_vals[list_ind] = mul

    return torch.sparse_coo_tensor(indices=coords, values=mult_table_vals, size=(n, n, n))
