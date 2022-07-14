"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

# Borrowed from https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/transforms/random_rotate.py
# with changes to keep track of the rotation / inverse rotation matrices.

import math
import numbers
import random
from itertools import repeat

import torch
import torch_geometric
from torch_geometric.transforms import LinearTransformation


class RandomRotate(object):
    r"""Rotates node positions around a specific axis by a randomly sampled
    factor within a given interval.

    Args:
        degrees (tuple or float): Rotation interval from which the rotation
            angle is sampled. If `degrees` is a number instead of a
            tuple, the interval is given by :math:`[-\mathrm{degrees},
            \mathrm{degrees}]`.
        axes (int, optional): The rotation axes. (default: `[0, 1, 2]`)
    """

    def __init__(self, degrees, axes=[0, 1, 2]):
        if isinstance(degrees, numbers.Number):
            degrees = (-abs(degrees), abs(degrees))
        assert isinstance(degrees, (tuple, list)) and len(degrees) == 2
        self.degrees = degrees
        self.axes = axes

    def __call__(self, data):
        if data.pos.size(-1) == 2:
            degree = math.pi * random.uniform(*self.degrees) / 180.0
            sin, cos = math.sin(degree), math.cos(degree)
            matrix = [[cos, sin], [-sin, cos]]
        else:
            m1, m2, m3 = torch.eye(3), torch.eye(3), torch.eye(3)
            if 0 in self.axes:
                degree = math.pi * random.uniform(*self.degrees) / 180.0
                sin, cos = math.sin(degree), math.cos(degree)
                m1 = torch.tensor([[1, 0, 0], [0, cos, sin], [0, -sin, cos]])
            if 1 in self.axes:
                degree = math.pi * random.uniform(*self.degrees) / 180.0
                sin, cos = math.sin(degree), math.cos(degree)
                m2 = torch.tensor([[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]])
            if 2 in self.axes:
                degree = math.pi * random.uniform(*self.degrees) / 180.0
                sin, cos = math.sin(degree), math.cos(degree)
                m3 = torch.tensor([[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]])

            matrix = torch.mm(torch.mm(m1, m2), m3)

        data_rotated = LinearTransformation(matrix)(data)
        if torch_geometric.__version__.startswith("2."):
            matrix = matrix.T

        # LinearTransformation only rotates `.pos`; need to rotate `.cell` too.
        if hasattr(data_rotated, "cell"):
            data_rotated.cell = torch.matmul(data_rotated.cell, matrix)

        return (
            data_rotated,
            matrix,
            torch.inverse(matrix),
        )

    def __repr__(self):
        return "{}({}, axis={})".format(
            self.__class__.__name__, self.degrees, self.axis
        )


class RandomJitter(object):
    r"""With a certain probability translates node positions by randomly sampled translation values
    within a given interval (functional name: :obj:`random_jitter`).
    In contrast to other random transformations,
    translation is applied separately at each position

    Args:
        translate (sequence or float or int): Maximum translation in each
            dimension, defining the range
            :math:`(-\mathrm{translate}, +\mathrm{translate})` to sample from.
            If :obj:`translate` is a number instead of a sequence, the same
            range is used for each dimension.
        prob (float): Probability (0 to 1) of translating positions
    """

    def __init__(self, config):
        self.translate = config.get("max_translation", 0.1)
        self.prob = config.get("translation_probability", 1.0)
        assert (
            self.prob <= 1.0
        ), "Probabiltiy of RandomJitter needs to be maximum 1.0"

    def __call__(self, data):
        if random.random() < self.prob:
            non_fixed_elements = ~data.fixed.bool()
            (n, dim), t = data.pos[non_fixed_elements].size(), self.translate
            if isinstance(t, numbers.Number):
                t = list(repeat(t, times=dim))
            assert len(t) == dim

            ts = []
            for d in range(dim):
                ts.append(
                    data.pos[non_fixed_elements]
                    .new_empty(n)
                    .uniform_(-abs(t[d]), abs(t[d]))
                )

            data.pos[non_fixed_elements] = data.pos[
                non_fixed_elements
            ] + torch.stack(ts, dim=-1)
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({{"max_translation": {self.translate}, "translation_probability": {self.prob}}})'
