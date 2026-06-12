"""Guided-backprop (GBP) saliency helpers.

Adapted from network_analysisnew_data.ipynb. Provides:
- GBP-modified ReLU machinery
- Per-neuron effective receptive field computation
- Weighted gradient combination
- Gradient heatmap visualization
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from scipy.stats import multivariate_normal

import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Function
from tqdm import tqdm

from models import AnalyzableModel
from models.unit_node_id import parse_unit_node_id, format_unit_node_id


# ---------------------------------------------------------------------------
# Guided-backprop ReLU
# Adapted from github.com/jacobgil/pytorch-grad-cam/blob/master/pytorch_grad_cam/guided_backprop.py
# ---------------------------------------------------------------------------

class _GBPReLU(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return grad_output * (x > 0).type_as(grad_output) * (grad_output > 0).type_as(grad_output)


class _GBPReLUModule(nn.Module):
    def forward(self, x):
        return _GBPReLU.apply(x)


def _replace_layers(module, src_type, make_replacement):
    for name, child in module._modules.items():
        if isinstance(child, src_type):
            module._modules[name] = make_replacement()
        else:
            _replace_layers(child, src_type, make_replacement)


# ---------------------------------------------------------------------------
# Effective receptive field computation
# ---------------------------------------------------------------------------

def get_neuron_effective_receptive_field(
    model: AnalyzableModel, neuron_id, eval_digits, device, use_gbp: bool = False
):
    """Compute input gradients (effective receptive field) for a single neuron.

    Returns a list of gradient tensors, one per evaluation image.
    """
    parsed_node_id = parse_unit_node_id(neuron_id)
    assert parsed_node_id is not None
    model.eval()

    captured = []

    def _hook(module, args, forward_output: Tensor):
        captured.append(forward_output)

    hook = model.hookable_layers()[parsed_node_id['layer_name']].register_forward_hook(_hook)

    def _to_batched(x: Tensor) -> Tensor:
        x = x.detach().to(device=device)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        return x

    flattened_batches = []
    for image in eval_digits:
        x = _to_batched(image if isinstance(image, Tensor) else torch.as_tensor(image))
        for i in range(x.size(0)):
            flattened_batches.append(x[i : i + 1])

    def _get_one_grad(x1: Tensor) -> Tensor:
        x1 = x1.detach().requires_grad_(True)
        captured.clear()
        model(x1)
        assert len(captured) == 1
        h = captured[0]
        y = h[0, parsed_node_id['unit_index']]
        (g,) = torch.autograd.grad(y, x1, retain_graph=False, create_graph=False)
        return g.clone().detach().cpu()

    if use_gbp:
        _replace_layers(model, nn.ReLU, _GBPReLUModule)

    try:
        gradients = []
        for image in flattened_batches:
            gradients.append(_get_one_grad(image).squeeze())
    finally:
        hook.remove()
        if use_gbp:
            _replace_layers(model, _GBPReLUModule, nn.ReLU)
        model.train()

    return gradients


def get_all_prev_layer_neuron_erfs(
    model: AnalyzableModel,
    target_neuron_id: str,
    eval_digits,
    device,
    use_gbp: bool = False,
):
    """Compute input gradients for ALL neurons in the layer immediately before
    ``target_neuron_id``.

    Returns:
        dict[str, list[Tensor]]:
            {prev_neuron_node_id: [grad_for_img0, grad_for_img1, ...], ...}
    """
    parsed = parse_unit_node_id(target_neuron_id)
    assert parsed is not None, f"Invalid node id: {target_neuron_id}"

    hookable = model.hookable_layers()
    layer_names = list(hookable.keys())

    cur_layer = parsed["layer_name"]
    assert cur_layer in hookable, f"Layer {cur_layer} not in hookable layers"

    cur_idx = layer_names.index(cur_layer)
    if cur_idx == 0:
        raise ValueError(
            f"{target_neuron_id} is in the first hookable layer ({cur_layer}); "
            "there is no previous neuron layer."
        )

    prev_layer_name = layer_names[cur_idx - 1]
    prev_layer = hookable[prev_layer_name]

    if not isinstance(prev_layer, nn.Linear):
        raise TypeError(
            f"Expected nn.Linear for DNN previous layer, got {type(prev_layer).__name__}"
        )

    n_prev_neurons = prev_layer.out_features
    scope = str(target_neuron_id).split("|")[0] if "|" in str(target_neuron_id) else "dnn"

    out = {}
    for neuron_idx in tqdm(
        range(n_prev_neurons),
        total=n_prev_neurons,
        desc="Computing ERFs for all neurons in the previous layer",
    ):
        prev_neuron_id = format_unit_node_id(
            scope=scope,
            layer_name=prev_layer_name,
            unit_type="neuron",
            unit_index=neuron_idx,
        )
        out[prev_neuron_id] = get_neuron_effective_receptive_field(
            model=model,
            neuron_id=prev_neuron_id,
            eval_digits=eval_digits,
            device=device,
            use_gbp=use_gbp,
        )

    return out


def weighted_sum_gradients(all_prev_grads, w_cp):
    """Weighted combination of per-neuron gradient maps.

    Args:
        all_prev_grads: dict[str, list[Tensor]] from get_all_prev_layer_neuron_erfs
        w_cp: 1-D weight vector, one entry per previous-layer neuron

    Returns:
        list[Tensor]: one combined gradient map per evaluation sample
    """
    ids = list(all_prev_grads.keys())
    n_samples = len(all_prev_grads[ids[0]])
    w = torch.as_tensor(w_cp, dtype=torch.float32)
    out = []
    for s in range(n_samples):
        g = torch.zeros_like(all_prev_grads[ids[0]][s], dtype=torch.float32)
        for i, nid in enumerate(ids):
            g = g + w[i] * all_prev_grads[nid][s].float()
        out.append(g)
    return out


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def gaussian_from_grad(matrix: np.ndarray | Tensor) -> np.ndarray:
    """Fit a 2D Gaussian to ``matrix`` (weighted by absolute values) and return
    the evaluated density on the same grid, scaled to preserve total weight."""
    matrix = matrix if isinstance(matrix, np.ndarray) else matrix.cpu().numpy()
    matrix = np.abs(matrix)
    h, w = matrix.shape
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    coords = np.stack([ys.ravel(), xs.ravel()], axis=1).astype(float)
    weights = matrix.ravel()
    weights_sum = weights.sum()
    if weights_sum == 0:
        raise ValueError("Matrix sums to zero — cannot compute a weighted mean/cov.")
    weights = weights / weights_sum
    mean = (weights[:, None] * coords).sum(axis=0)
    diff = coords - mean
    cov = (weights[:, None] * diff).T @ diff
    rv = multivariate_normal(mean=mean, cov=cov)
    gaussian = rv.pdf(coords).reshape(h, w)
    return gaussian * weights_sum


def _global_min_max(arrs):
    lo = min(float(a.min()) for a in arrs)
    hi = max(float(a.max()) for a in arrs)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return 0.0, 1.0
    return lo, hi


def plot_gradient_heatmaps(
    all_gradients,
    *,
    n_digits=10,
    n_samples=5,
    cmap="RdBu",
    use_gaussian=False,
    neuron_id=None,
    only_digits=None,
):
    """Plot per-sample gradient maps and per-digit mean maps.

    Returns:
        (fig1, axes1), (fig2, axes2)  — figure 1 is the full grid, figure 2 the per-digit means.
    """
    if only_digits is None:
        only_digits = range(n_digits)
        use_digits = n_digits
    else:
        use_digits = len(only_digits)
    total = n_digits * n_samples
    if len(all_gradients) < total:
        raise ValueError(f"Expected {total} gradients, got {len(all_gradients)}")

    mats = [all_gradients[i] for i in range(total)]
    vmin, vmax = _global_min_max(mats)
    mm = max(abs(vmin), abs(vmax))
    vmin, vmax = -mm, mm

    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 12,
        "axes.labelsize": 8,
    })

    # Figure 1: full sample grid
    fig1, axes1 = plt.subplots(
        use_digits, n_samples,
        figsize=(2.2 * n_samples, 1.9 * use_digits),
        constrained_layout=True,
    )

    for i,d in enumerate(only_digits):
        for s in range(n_samples):
            ax = axes1[i, s]
            img = mats[d * n_samples + s]
            if use_gaussian:
                img = gaussian_from_grad(img)
            ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
            if d == 0:
                ax.set_title(f"s={s}")
            if s == 0:
                ax.set_ylabel(f"{d}", rotation=0, labelpad=10, va="center")
            ax.set_xticks([])
            ax.set_yticks([])

    # Add y-axis label "Digit" on the left (first column, center)
    fig1.text(
        -0.03, 0.5, "Digit", va="center", ha="center", rotation="vertical", fontsize=12
    )

    def parse_neuron_id(neuron_id):
        parsed = parse_unit_node_id(neuron_id)
        if neuron_id is None or parsed is None:
            return ""
        layer_idx = int(parsed["layer_name"].split(".")[-1]) // 2
        return r"$n^{("+str(layer_idx)+r")}_{"+str(parsed["unit_index"])+r"}$"

    print(parse_neuron_id(neuron_id))
    fig1.suptitle("Sensitivity maps per digit for neuron " + parse_neuron_id(neuron_id) + " at "+ r"$t_\text{start}$", fontsize=16)
    fig1.colorbar(
        cm.ScalarMappable(norm=colors.Normalize(vmin, vmax), cmap=cmap),
        ax=axes1, shrink=0.6, location="right",
    ).set_label("intensity")

    # Figure 2: per-digit mean
    means = []
    for d in range(n_digits):
        stack = np.stack([mats[d * n_samples + s] for s in range(n_samples)], axis=0)
        means.append(stack.mean(0))

    mn, mx = _global_min_max(means)
    mm = max(abs(mn), abs(mx))
    mn, mx = -mm, mm

    fig2, axes2 = plt.subplots(5, 2, figsize=(6, 10), constrained_layout=True)

    for d in range(n_digits):
        r, c = divmod(d, 2)
        ax = axes2[r, c]
        img = means[d]
        if use_gaussian:
            img = gaussian_from_grad(img)
        ax.imshow(img, cmap=cmap, vmin=mn, vmax=mx, interpolation="nearest")
        ax.set_title(f"digit {d}")
        ax.set_xticks([])
        ax.set_yticks([])

    fig2.suptitle(f"Mean gradient map per digit (n={n_samples})", fontsize=12)
    fig2.colorbar(
        cm.ScalarMappable(norm=colors.Normalize(mn, mx), cmap=cmap),
        ax=axes2, shrink=0.7, location="right",
    ).set_label("mean intensity")

    return (fig1, axes1), (fig2, axes2)
