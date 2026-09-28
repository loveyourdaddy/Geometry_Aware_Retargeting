"""
Utilities that extract self-attention (among joints of the same character) / cross-attention
(how much the partner's joints are attended) weights and save them as heatmaps.

The training code is untouched: a forward hook sets need_weights=True only when
nn.MultiheadAttention is called, captures the weights, and is removed afterwards.
(CrossTransformerBlock.sa_block / .ca_block and TransformerBlock.sa_block all use
 nn.MultiheadAttention internally, so they can be told apart by name (module path) alone.)
"""
import os
import re

import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from option_parser import bones

# spat_num_token = num_joint(22) + 1(root, concatenated as the last token) = 23
JOINT_LABELS = bones + ["Root"]


class AttentionRecorder:
    def __init__(self, root_module, average_heads=True):
        self.root_module = root_module
        self.average_heads = average_heads
        self.records = []
        self._originals = []  # [(module, original_forward), ...]

    def _make_wrapped_forward(self, name, module, original_forward):
        def wrapped_forward(*args, **kwargs):
            kwargs = dict(kwargs)
            kwargs['need_weights'] = True
            kwargs['average_attn_weights'] = self.average_heads
            output = original_forward(*args, **kwargs)
            _, attn_weights = output
            if attn_weights is not None:
                self.records.append({'name': name, 'weights': attn_weights.detach().cpu()})
            return output
        return wrapped_forward

    def __enter__(self):
        for name, module in self.root_module.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                original_forward = module.forward
                self._originals.append((module, original_forward))
                module.forward = self._make_wrapped_forward(name, module, original_forward)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for module, original_forward in self._originals:
            module.forward = original_forward
        self._originals = []


def _kind_and_side(name):
    """Extract (attention kind, which character, layer index) from a module path string.
    e.g. 'spat_encoder.trans0.1.sa_block.attention'       -> ('self',  'A', 1)   (SharingTransformer)
        'spat_encoder.trans1.0.ca_block.attention'       -> ('cross', 'B', 0)   (SharingTransformer)
        'spat_encoder.trans1.trans0.0.sa_block.attention' -> ('self', 'B', 0)   (TwinTransformer)
    trans0 = character A (the query side), trans1 = character B. (convention of SharingTransformer/TwinTransformer in define_network.py)

    Note: in TwinTransformer (no_cross_attn ablation) the inner Transformer class always names its layer
    ModuleList self.trans0, so the substring 'trans0' also appears in character B's path
    (e.g. 'trans1.trans0.0...'). So instead of a plain substring test (`'trans0' in name`), the path is split
    by dots and scanned from the left; the first trans0/trans1 found (= the character level) decides.
    """
    if 'sa_block' in name:
        kind = 'self'
    elif 'ca_block' in name:
        kind = 'cross'
    else:
        kind = 'other'
    side = '?'
    for part in name.split('.'):
        if part == 'trans0':
            side = 'A'
            break
        if part == 'trans1':
            side = 'B'
            break
    m = re.search(r'\.(\d+)\.(sa_block|ca_block)', name)
    layer = int(m.group(1)) if m else -1
    return kind, side, layer


def _dedupe_first_per_name(records):
    """Keep only the first occurrence when the same name (module path) is recorded several times in one forward.
    (TransformerBlock.forward calls self.sa_block(x, ...) twice on x, so the existing code always
    records the same result twice - in eval mode dropout is off, so the two calls are
    numerically identical. This does not matter for the mean, but when picking frames by index
    the duplicated frames along the N axis shift the indices, so they must be removed first.)
    """
    seen = set()
    out = []
    for rec in records:
        if rec['name'] in seen:
            continue
        seen.add(rec['name'])
        out.append(rec)
    return out


def _strip_zero_attn(weights, num_token):
    """With add_zero_attn, if S == num_token+1, drop the last (zero) key column."""
    if weights.shape[-1] == num_token + 1:
        weights = weights[..., :num_token]
    return weights


def collect_attention_maps(records, num_token=23):
    """Group records by (kind, side, layer) into attention maps averaged over all frames (dict[key] = (num_token, num_token) ndarray)."""
    grouped = {}
    for rec in _dedupe_first_per_name(records):
        kind, side, layer = _kind_and_side(rec['name'])
        if kind == 'other':
            continue
        grouped.setdefault((kind, side, layer), []).append(rec['weights'])

    maps = {}
    for key, wlist in grouped.items():
        w_cat = torch.cat(wlist, dim=0)
        w_cat = _strip_zero_attn(w_cat, num_token)
        maps[key] = w_cat.reshape(-1, num_token, num_token).mean(dim=0).numpy()
    return maps


def collect_attention_maps_per_frame(records, frame_indices, num_token=23):
    """For each (kind, side, layer) in records, extract the attention map of each of the given frame_indices.
    Assumes batch_size=1 (the test/visualize scripts always use a batch of one motion pair, and
    SpatioTemporalTransformer.forward reshapes (batch, len_frame, ...) -> (batch*len_frame, ...),
    so with batch=1 the index along this N axis is the frame number).

    Returns: {(kind, side, layer): {frame_idx: (num_token, num_token) ndarray}}
    """
    grouped = {}
    for rec in _dedupe_first_per_name(records):
        kind, side, layer = _kind_and_side(rec['name'])
        if kind == 'other':
            continue
        grouped[(kind, side, layer)] = _strip_zero_attn(rec['weights'], num_token)

    maps = {}
    for key, w in grouped.items():
        per_frame = {}
        for f in frame_indices:
            if f < 0 or f >= w.shape[0]:
                print(f"[collect_attention_maps_per_frame] frame {f}: outside the motion length ({w.shape[0]} frames), skipped")
                continue
            per_frame[f] = w[f].numpy()
        maps[key] = per_frame
    return maps


def _drop_root(matrix):
    """The root is always attached as the last token (index -1) of spat_num_token (see
    `torch.cat((concat_a_t, root_a_), dim=-2)` in define_network.py), so cutting the last row/column
    leaves only the attention among joints."""
    return matrix[:-1, :-1]


def _axis_labels(kind, side):
    if kind == 'cross':
        partner = 'B' if side == 'A' else 'A'
        return f'char {partner} joint (key)', f'char {side} joint (query)'
    return f'char {side} joint (key)', f'char {side} joint (query)'


def plot_attention_heatmap(matrix, title, save_path, xlabel='key joint', ylabel='query joint', labels=None):
    labels = labels or JOINT_LABELS
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(matrix, cmap='viridis', aspect='auto')
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def visualize_attention(net, dataset, save_dir, num_token=23, frame_indices=None, include_root=True):
    """Run net.forward(dataset) once, extract all self/cross attention, and save them as heatmap pngs.

    With frame_indices=None (default), saves one map averaged over all frames, as before.
    With frame_indices=[71, 94, ...], saves a separate heatmap for each of those frames instead of the mean
    (the frame number is added to the file name). Assumes batch_size=1.
    With include_root=False, drops the root token (last row/column) and draws only the attention among the 22 joints
    (useful because the root's attention can be much larger/smaller than the others and hide the pattern among joints).

    Returns: frame_indices=None -> {(kind, side, layer): (num_token, num_token) ndarray}
            frame_indices given -> {(kind, side, layer): {frame_idx: (num_token, num_token) ndarray}}
    (the returned matrices always keep the original size including the root, regardless of include_root. Dropping the root applies only to the saved pngs)
    """
    with AttentionRecorder(net.spatio_temp_net) as rec:
        net.forward(dataset)

    kind_label = {'self': 'self-attn', 'cross': 'cross-attn'}
    labels = JOINT_LABELS if include_root else bones
    suffix = '' if include_root else '_noroot'

    if frame_indices is None:
        maps = collect_attention_maps(rec.records, num_token=num_token)
        for (kind, side, layer), mat in sorted(maps.items(), key=lambda kv: kv[0]):
            title = f"{kind_label[kind]} · char {side} · layer {layer} (avg over all frames)"
            fname = f"{kind}_{side}_layer{layer}{suffix}.png"
            xlabel, ylabel = _axis_labels(kind, side)
            plot_mat = mat if include_root else _drop_root(mat)
            plot_attention_heatmap(plot_mat, title, os.path.join(save_dir, fname),
                                    xlabel=xlabel, ylabel=ylabel, labels=labels)
        return maps

    maps = collect_attention_maps_per_frame(rec.records, frame_indices, num_token=num_token)
    for (kind, side, layer), per_frame in sorted(maps.items(), key=lambda kv: kv[0]):
        xlabel, ylabel = _axis_labels(kind, side)
        for f, mat in per_frame.items():
            title = f"{kind_label[kind]} · char {side} · layer {layer} · frame {f}"
            fname = f"{kind}_{side}_layer{layer}_frame{f}{suffix}.png"
            plot_mat = mat if include_root else _drop_root(mat)
            plot_attention_heatmap(plot_mat, title, os.path.join(save_dir, fname),
                                    xlabel=xlabel, ylabel=ylabel, labels=labels)
    return maps
