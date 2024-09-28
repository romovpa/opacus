#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from __future__ import annotations

import logging
from functools import partial
from typing import List, Tuple

import torch
import torch.nn as nn
from opacus.layers.dp_rnn import DPRNNBase, DPRNNCellBase
from opacus.utils.module_utils import requires_grad, trainable_modules


logger = logging.getLogger(__name__)


def create_or_accumulate_grad_sample(
    param: torch.Tensor, grad_sample: torch.Tensor, layer: nn.Module
) -> None:
    """
    Creates a ``_current_grad_sample`` attribute in the given parameter, or adds to it
    if the ``_current_grad_sample`` attribute already exists.



    Args:
        param: Parameter to which ``grad_sample`` will be added
        grad_sample: Per-sample gradients tensor. Must be of the same
            shape as ``param`` with extra batch dimension
    """

    if hasattr(param, "_current_grad_sample"):
        param._current_grad_sample[: grad_sample.shape[0]] += grad_sample
    else:
        # TODO: maybe set max_batch_len on a parameter level, so
        # you don't need to pass layer here?
        max_batch_len = layer.max_batch_len
        param._current_grad_sample = torch.zeros(
            torch.Size([max_batch_len]) + grad_sample.shape[1:],
            device=grad_sample.device,
            dtype=grad_sample.dtype,
        )
        param._current_grad_sample[: grad_sample.shape[0]] = grad_sample


def promote_current_grad_sample(p: nn.Parameter) -> None:
    if p.requires_grad:
        if hasattr(p, "grad_sample"):
            if isinstance(p.grad_sample, list):
                p.grad_sample.append(p._current_grad_sample)
            else:
                p.grad_sample = [p.grad_sample, p._current_grad_sample]
        else:
            p.grad_sample = p._current_grad_sample

        del p._current_grad_sample


class GradSampleModule(nn.Module):
    r"""
    Extends nn.Module so that its parameter tensors have an extra field called .grad_sample.
    """
    GRAD_SAMPLERS = {}

    def __init__(
        self,
        m: nn.Module,
        *,
        batch_first=True,
        loss_reduction="mean",
        strict: bool = True,
    ):
        super().__init__()

        errors = self.validate(module=m, raise_if_error=strict)
        if errors and not strict:
            logger.info(
                f"GradSampleModule found the following errors: {errors}."
                "Using non-strict mode, continuing"
            )

        # TODO: accessing weights via _module is inconveniet, override _getattr_
        self._module = m  # TODO: it's not 100% certain that this should stay private
        self.hooks_enabled = False
        self.batch_first = batch_first
        self.loss_reduction = loss_reduction
        self.add_hooks(loss_reduction=loss_reduction, batch_first=batch_first)

        # TODO: Do we want to initialize empty grad_sample attribures here?
        # e.g. p.grad_sample = torch.zeros([0, p.shape[1:]])

    # TODO: few other common methods needs to be forwarded (e.g. name())
    # I think there's a way to intercept calls to all unknown attributes and
    # forward it to the self._module - is it a good idea?

    def forward(self, x, *args, **kwargs):
        # TODO: check to forbid double forward
        # TODO: also check to force zero_grad
        return self._module(x, *args, **kwargs)

    # TODO: match nn.Module signature
    def zero_grad(self):
        self.del_grad_sample()
        super().zero_grad()

    def del_grad_sample(self):
        """
        Deletes ``.grad_sample`` from this module's parameters.

        Why del? Normally, ``zero_grad()`` would do ``p.grad.zero_()`` and keep the allocation.
        Normal grads can do this, because their shape is always the same.
        Grad samples do not behave like this, because they accumulate over the batch dim.
        If you have ``batch_size=32`` and size (12, 16) and you backprop twice, you should
        expect to have grad_samples of size [64, 12, 16]. If you backprop once more,
        then you'll get size [96, 12, 16] and so on.
        So when you zero out, you should be left with nothing so you can start over.
        """
        for p in self.parameters():
            if hasattr(p, "grad_sample") and p.grad_sample is not None:
                if isinstance(p.grad_sample, list):
                    grad_samples = p.grad_sample
                else:
                    grad_samples = [p.grad_sample]

                for grad_sample in grad_samples:
                    if grad_sample.grad_fn is not None:
                        grad_sample.detach_()
                    else:
                        grad_sample.requires_grad_(False)

                del p.grad_sample

    def to_standard_module(self) -> nn.Module:
        """
        Returns the standard nn.Module wrapped by this, eliminating all traces
        of grad samples and hooks

        Returns:
            The wrapped module
        """
        self._close()
        return self._module

    def add_hooks(self, loss_reduction: str = "mean", batch_first: bool = True) -> None:
        """
        Adds hooks to model to save activations and backprop values.
        The hooks will
        1. save activations into param.activations during forward pass
        2. compute per-sample gradients in params.grad_sample during backward pass.
        Call ``remove_hooks(model)`` to disable this.

        Args:
            model: the model to which hooks are added
            loss_type: either "mean" or "sum" depending on whether backpropped
                loss was averaged or summed over batch (default: "mean")
            batch_dim: the batch dimension (default: 0)
        """
        if hasattr(self._module, "autograd_grad_sample_hooks"):
            raise ValueError("Trying to add hooks twice to the same model")
        else:
            self._module.autograd_grad_sample_hooks = []
            self.autograd_grad_sample_hooks = self._module.autograd_grad_sample_hooks

        for module in trainable_modules(self._module):
            if type(module) in self.GRAD_SAMPLERS:
                self.autograd_grad_sample_hooks.append(
                    module.register_forward_hook(self.capture_activations_hook)
                )

                self.autograd_grad_sample_hooks.append(
                    module.register_backward_hook(
                        partial(
                            self.capture_backprops_hook,
                            loss_reduction=loss_reduction,
                            batch_first=batch_first,
                        )
                    )
                )
        self.enable_hooks()

    def remove_hooks(self) -> None:
        """
        Removes hooks added by ``add_hooks()``
        """
        self.disable_hooks()

        for p in self.parameters():
            if hasattr(p, "ddp_hooks"):
                while p.ddp_hooks:
                    handle = p.ddp_hooks.pop()
                    handle.remove()
                delattr(p, "ddp_hooks")

        if not hasattr(self, "autograd_grad_sample_hooks"):
            raise ValueError("Asked to remove hooks, but no hooks found")
        else:
            while self.autograd_grad_sample_hooks:
                handle = self.autograd_grad_sample_hooks.pop()
                handle.remove()
            delattr(self, "autograd_grad_sample_hooks")
            delattr(self._module, "autograd_grad_sample_hooks")

    def disable_hooks(self) -> None:
        r"""
        Globally disable all hooks installed by this library.
        Why is this needed? As per https://github.com/pytorch/pytorch/issues/25723, there is
        a bug in Autograd that makes removing hooks do nothing if the graph was already
        constructed. For this reason, we have this method to at least turn them off.
        """
        self.hooks_enabled = False

    def enable_hooks(self) -> None:
        r"""
        The opposite of ``disable_hooks()``. Hooks are always enabled unless you explicitly
        disable them so you don't need to call this unless you want to re-enable them.
        """
        self.hooks_enabled = True

    def __repr__(self):
        return f"GradSampleModule({self._module.__repr__()})"

    def _close(self):
        self.del_grad_sample()
        self.remove_hooks()

    def capture_activations_hook(
        self,
        module: nn.Module,
        forward_input: List[torch.Tensor],
        _forward_output: torch.Tensor,
    ):
        if (
            not requires_grad(module)
            or not module.training
            or not torch.is_grad_enabled()
        ):
            return

        if not self.hooks_enabled:
            return

        if not hasattr(module, "activations"):
            module.activations = []
        module.activations.append(forward_input[0].detach())  # pyre-ignore

    def capture_backprops_hook(
        self,
        module: nn.Module,
        _forward_input: torch.Tensor,
        forward_output: torch.Tensor,
        loss_reduction: str,
        batch_first: bool,
    ):
        """
        Computes per sample gradients given the current backprops and activations
        stored by the associated forward hook. Computed per sample gradients are
        stored in ``grad_sample`` field in each parameter.

        For non-recurrent layers the process is straightforward: for each
        ``loss.backward()`` call this hook will be called exactly one. For recurrent
        layers, however, this is more complicated and the hook will be called multiple
        times, while still processing the same batch of data.

        For this reason we first accumulate the gradients from *the same batch* in
        ``p._current_grad_sample`` and then, when we detect the end of a full backward
        pass - we store accumulated result on ``p.grad_sample``.

        From there, ``p.grad_sample`` could be either a Tensor or a list of Tensors,
        if accumulated over multiple batches

        Args:
            module: nn.Module,
            _forward_input: torch.Tensor,
            forward_output: torch.Tensor,
            loss_reduction: str,
            batch_first: bool,
        """
        if not self.hooks_enabled:
            return

        backprops = forward_output[0].detach()
        activations, backprops = self.rearrange_grad_samples(
            module, backprops, loss_reduction, batch_first
        )
        grad_sampler_fn = self.GRAD_SAMPLERS[type(module)]
        grad_samples = grad_sampler_fn(module, activations, backprops)
        for param, gs in grad_samples.items():
            create_or_accumulate_grad_sample(param, gs, module)

        if len(module.activations) == 0:
            if hasattr(module, "max_batch_len"):
                del module.max_batch_len

            for p in module.parameters():
                promote_current_grad_sample(p)

    def rearrange_grad_samples(
        self,
        module: nn.Module,
        backprops: torch.Tensor,
        loss_reduction: str,
        batch_first: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Rearrange activations and grad_samples based on loss reduction and batch dim

        Args:
            module: the module for which per-sample gradients are computed
            backprops: the captured backprops
            loss_reduction: either "mean" or "sum" depending on whether backpropped
                loss was averaged or summed over batch
            batch_first: True is batch dimension is first
        """
        if not hasattr(module, "activations"):
            raise ValueError(
                f"No activations detected for {type(module)},"
                " run forward after add_hooks(model)"
            )

        # TODO(remove before merge): Fix this
        batch_dim = 0 if batch_first or type(module) is nn.Linear else 1

        activations = module.activations.pop()

        if not hasattr(module, "max_batch_len"):
            # For packed sequences, max_batch_len is set in the forward of the model (e.g. the LSTM)
            # Otherwise we infer it here
            module.max_batch_len = _get_batch_size(module, activations, batch_dim)

        n = module.max_batch_len
        if loss_reduction == "mean":
            backprops = backprops * n
        elif loss_reduction == "sum":
            backprops = backprops
        else:
            raise ValueError(
                f"loss_reduction = {loss_reduction}. Only 'sum' and 'mean' losses are supported"
            )

        # No matter where the batch dimension was, .grad_samples will *always* put it in the first dim
        if batch_dim != 0:
            activations = activations.permute(
                [batch_dim] + [x for x in range(activations.dim()) if x != batch_dim]
            )
            backprops = backprops.permute(
                [batch_dim] + [x for x in range(backprops.dim()) if x != batch_dim]
            )

        return activations, backprops

    @classmethod
    def is_supported(cls, module: nn.Module) -> bool:
        """Check if this individual module is supported"""
        return type(module) in cls.GRAD_SAMPLERS or isinstance(
            module, (DPRNNBase, DPRNNCellBase)
        )

    @classmethod
    def validate(
        cls, module: nn.Module, raise_if_error: bool = False
    ) -> List[NotImplementedError]:
        """Validate support for module being wrapped"""
        errors = []
        errors.extend(
            [
                NotImplementedError(f"grad sampler is not yet implemented for {m}")
                for m in trainable_modules(module)
                if not GradSampleModule.is_supported(m)
            ]
        )
        # raise or return errors as needed
        if raise_if_error and len(errors) > 0:
            raise NotImplementedError(errors)
        else:
            return errors


def _get_batch_size(
    module: nn.Module, grad_sample: torch.Tensor, batch_dim: int
) -> int:
    r"""
    Computes and returns the maximum batch size which is the maximum of the dimension values
    along 'batch_dim' axis over module.activations + [grad_sample], where module.activations is
    a list. If module.activations is a not a list, then return grad_sample.shape[batch_dim].
    """

    max_batch_len = 0
    for out in module.activations:
        if out.shape[batch_dim] > max_batch_len:
            max_batch_len = out.shape[batch_dim]

    max_batch_len = max(max_batch_len, grad_sample.shape[batch_dim])
    return max_batch_len
