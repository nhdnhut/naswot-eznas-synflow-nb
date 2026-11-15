"""Educational implementations of two zero-cost architecture signals.

Scope
-----
- NASWOT: activation-overlap kernel and log-determinant score.
- SynFlow: one-pass, initialization-time aggregate synaptic-flow saliency.

These functions are intended for controlled experiments on small PyTorch models.
They are not a drop-in reproduction of every detail in the official paper code.
In particular, ``synflow_score`` computes an aggregate score once; it does not run
SynFlow's iterative pruning procedure.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
from torch import Tensor, nn


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters, optionally including frozen parameters."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if (parameter.requires_grad or not trainable_only)
    )


def naswot_score(
    model: nn.Module,
    inputs: Tensor,
    *,
    jitter: float = 1e-6,
    return_kernel: bool = False,
) -> dict[str, Any]:
    """Compute the NASWOT activation-overlap log-determinant score.

    For each ReLU output, the function records a binary activation pattern for
    every input in the batch. If C is the concatenated binary pattern matrix,
    the overlap kernel is

        K[i, j] = sum_k 1[C[i, k] == C[j, k]]
                = (C @ C.T + (1-C) @ (1-C).T)[i, j].

    The reported score uses log|det(K + jitter * I)|. ``raw_logdet`` is
    calculated without jitter so singularity remains visible in the results.
    A non-zero jitter is a numerical stabilization choice and changes the
    evaluated matrix; record it whenever scores are reported.

    The model's previous train/eval mode is restored after the forward pass.
    This implementation requires ReLU modules (``nn.ReLU``), not functional
    calls such as ``torch.nn.functional.relu``.
    """
    if inputs.ndim < 2:
        raise ValueError("inputs must have a batch dimension and feature dimensions")
    if jitter < 0:
        raise ValueError("jitter must be non-negative")

    activations: list[Tensor] = []
    handles = []

    def save_relu_output(_module: nn.Module, _module_inputs: tuple[Tensor, ...], output: Tensor) -> None:
        if not isinstance(output, Tensor):
            raise TypeError("A hooked ReLU module returned a non-Tensor output")
        if output.shape[0] != inputs.shape[0]:
            raise ValueError("ReLU output batch dimension changed unexpectedly")
        activations.append((output.detach() > 0).reshape(output.shape[0], -1))

    for module in model.modules():
        if isinstance(module, nn.ReLU):
            handles.append(module.register_forward_hook(save_relu_output))

    if not handles:
        raise ValueError(
            "No nn.ReLU modules were found. Replace functional ReLUs with nn.ReLU "
            "modules or add a suitable activation hook."
        )

    previous_mode = model.training
    model.eval()
    try:
        with torch.no_grad():
            _ = model(inputs)
    finally:
        for handle in handles:
            handle.remove()
        model.train(previous_mode)

    if not activations:
        raise RuntimeError("No ReLU activations were captured")

    # Concatenate every ReLU unit / spatial activation position into one code.
    codes = torch.cat(activations, dim=1).to(dtype=torch.float64)
    batch_size, num_units = codes.shape
    kernel = codes @ codes.T + (1.0 - codes) @ (1.0 - codes).T
    kernel = 0.5 * (kernel + kernel.T)  # remove tiny asymmetry from arithmetic

    raw_sign, raw_logabsdet = torch.linalg.slogdet(kernel)
    stable_kernel = kernel
    if jitter > 0:
        stable_kernel = kernel + jitter * torch.eye(
            batch_size, dtype=kernel.dtype, device=kernel.device
        )
    sign, logabsdet = torch.linalg.slogdet(stable_kernel)

    # For a PSD overlap kernel, the sign should be non-negative. A zero sign
    # indicates a singular matrix and therefore an undefined finite logdet.
    score = float(logabsdet.item()) if sign.item() > 0 else float("-inf")
    result: dict[str, Any] = {
        "score": score,
        "sign": int(sign.item()),
        "raw_sign": int(raw_sign.item()),
        "raw_logdet": float(raw_logabsdet.item()),
        "batch_size": int(batch_size),
        "num_relu_units": int(num_units),
        "jitter": float(jitter),
    }
    if return_kernel:
        result["kernel"] = kernel.detach().cpu()
    return result


def synflow_score(
    model: nn.Module,
    input_shape: tuple[int, ...],
    *,
    device: torch.device | str | None = None,
    use_double: bool = False,
    return_parameter_scores: bool = False,
) -> dict[str, Any]:
    """Compute a one-pass aggregate SynFlow saliency score at initialization.

    A deep copy is used so the caller's model parameters and gradients are not
    modified. Parameters are replaced by their element-wise absolute values,
    an all-ones synthetic input is propagated in eval mode, and the sum of the
    outputs is backpropagated. The aggregate score is

        sum_p sum(abs(p * grad_p)).

    This is a compact pedagogical implementation for sequential/small CNN
    experiments. It is not the full iterative pruning algorithm (which
    repeatedly re-evaluates saliency and updates a binary mask).
    """
    if not input_shape or any(int(dim) <= 0 for dim in input_shape):
        raise ValueError("input_shape must contain positive dimensions")

    model_copy = copy.deepcopy(model)
    if device is None:
        try:
            target_device = next(model_copy.parameters()).device
        except StopIteration:
            target_device = torch.device("cpu")
    else:
        target_device = torch.device(device)
    model_copy = model_copy.to(target_device)
    if use_double:
        model_copy = model_copy.double()
    model_copy.eval()

    # Linearize the network with |theta_0|. This is deliberately done only on
    # the copy so the original initialization remains available for NASWOT.
    with torch.no_grad():
        for parameter in model_copy.parameters():
            parameter.abs_()

    model_copy.zero_grad(set_to_none=True)
    first_parameter = next(model_copy.parameters(), None)
    parameter_dtype = first_parameter.dtype if first_parameter is not None else torch.float32
    synthetic_input = torch.ones(input_shape, device=target_device, dtype=parameter_dtype)

    output = model_copy(synthetic_input)
    if not isinstance(output, Tensor):
        raise TypeError("Model output must be a Tensor")
    if not torch.isfinite(output).all():
        raise FloatingPointError("Non-finite output encountered during SynFlow forward pass")

    objective = output.sum()
    objective.backward()

    parameter_scores: dict[str, float] = {}
    total_score = 0.0
    scored_parameter_count = 0
    for name, parameter in model_copy.named_parameters():
        if parameter.grad is None:
            continue
        score_tensor = (parameter * parameter.grad).abs()
        parameter_score = float(score_tensor.sum().item())
        if not torch.isfinite(score_tensor).all():
            raise FloatingPointError(f"Non-finite SynFlow saliency encountered in {name}")
        parameter_scores[name] = parameter_score
        total_score += parameter_score
        scored_parameter_count += parameter.numel()

    if scored_parameter_count == 0:
        raise RuntimeError("No parameter received a SynFlow gradient")

    result: dict[str, Any] = {
        "score": float(total_score),
        "scored_parameter_count": int(scored_parameter_count),
        "use_double": bool(use_double),
    }
    if return_parameter_scores:
        result["parameter_scores"] = parameter_scores
    return result


def synflow_iterative_prune(
    model: nn.Module,
    input_shape: tuple[int, ...],
    *,
    compression_ratio: float,
    steps: int = 100,
    device: torch.device | str | None = None,
    use_double: bool = False,
) -> dict[str, Any]:
    """Apply an educational implementation of iterative SynFlow pruning.

    ``compression_ratio`` is N / N_remaining; for example, 10 targets about
    90% sparsity after the final iteration. The exponential schedule follows
    the percentile schedule described in the SynFlow paper.

    Ties at the percentile threshold can make the achieved sparsity differ
    from the target. This function uses ``torch.quantile`` and strict ``>``;
    it is a compact reference implementation, not a claim of bitwise identity
    with the authors' original pruning code.
    """
    if compression_ratio <= 1:
        raise ValueError("compression_ratio must be > 1")
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if not input_shape or any(int(dim) <= 0 for dim in input_shape):
        raise ValueError("input_shape must contain positive dimensions")

    if device is None:
        try:
            target_device = next(model.parameters()).device
        except StopIteration:
            target_device = torch.device("cpu")
    else:
        target_device = torch.device(device)

    working_model = copy.deepcopy(model).to(target_device)
    if use_double:
        working_model = working_model.double()
    working_model.eval()

    initial_values = {
        name: parameter.detach().clone()
        for name, parameter in working_model.named_parameters()
        if parameter.requires_grad
    }
    if not initial_values:
        raise ValueError("Model has no trainable parameters to prune")

    masks = {
        name: torch.ones_like(value, dtype=torch.bool, device=target_device)
        for name, value in initial_values.items()
    }
    total_parameter_count = sum(mask.numel() for mask in masks.values())
    first_value = next(iter(initial_values.values()))
    synthetic_input = torch.ones(
        input_shape,
        device=target_device,
        dtype=first_value.dtype,
    )

    history: list[dict[str, float | int]] = []
    for step in range(1, steps + 1):
        working_model.zero_grad(set_to_none=True)
        with torch.no_grad():
            for name, parameter in working_model.named_parameters():
                if name in masks:
                    parameter.copy_(initial_values[name].abs() * masks[name])

        output = working_model(synthetic_input)
        if not isinstance(output, Tensor) or not torch.isfinite(output).all():
            raise FloatingPointError("Non-finite or invalid output during SynFlow pruning")
        output.sum().backward()

        score_tensors: dict[str, Tensor] = {}
        flat_scores = []
        for name, parameter in working_model.named_parameters():
            if name not in masks or parameter.grad is None:
                continue
            scores = (parameter * parameter.grad).abs()
            if not torch.isfinite(scores).all():
                raise FloatingPointError(f"Non-finite SynFlow saliency in {name}")
            score_tensors[name] = scores
            flat_scores.append(scores.reshape(-1))

        if not flat_scores:
            raise RuntimeError("No SynFlow saliency tensors were produced")

        percentile = 1.0 - compression_ratio ** (-step / steps)
        all_scores = torch.cat(flat_scores)
        threshold = torch.quantile(all_scores, percentile)
        masks = {
            name: masks[name] & (score_tensors[name] > threshold)
            for name in masks
        }

        remaining = sum(int(mask.sum().item()) for mask in masks.values())
        history.append({
            "step": step,
            "percentile": float(percentile * 100.0),
            "threshold": float(threshold.item()),
            "remaining_parameters": remaining,
            "sparsity": float(1.0 - remaining / total_parameter_count),
        })

    pruned_model = copy.deepcopy(model).to(target_device)
    if use_double:
        pruned_model = pruned_model.double()
    with torch.no_grad():
        for name, parameter in pruned_model.named_parameters():
            if name in masks:
                parameter.copy_(initial_values[name] * masks[name])

    return {
        "model": pruned_model,
        "masks": {name: mask.detach().cpu() for name, mask in masks.items()},
        "history": history,
        "total_parameters": int(total_parameter_count),
        "remaining_parameters": int(history[-1]["remaining_parameters"]),
        "sparsity": float(history[-1]["sparsity"]),
        "compression_ratio": float(compression_ratio),
        "steps": int(steps),
        "use_double": bool(use_double),
    }
