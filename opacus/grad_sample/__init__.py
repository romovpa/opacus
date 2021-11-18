#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from .grad_sample_module import GradSampleModule, create_or_accumulate_grad_sample
from .utils import register_grad_sampler


def register_grad_samplers():
    """Registers all grad samplers available in the library.

    Importing all submodules applies @register_grad_sampler to all decorated functions.
    """
    from .conv import compute_conv_grad_sample  # noqa
    from .dp_multihead_attention import compute_sequence_bias_grad_sample  # noqa
    from .embedding import compute_embedding_grad_sample  # noqa
    from .group_norm import compute_group_norm_grad_sample  # noqa
    from .instance_norm import compute_instance_norm_grad_sample  # noqa
    from .layer_norm import compute_layer_norm_grad_sample  # noqa
    from .linear import compute_linear_grad_sample  # noqa


register_grad_samplers()

__all__ = [
    "GradSampleModule",
    "register_grad_sampler",
    "create_or_accumulate_grad_sample",
]
