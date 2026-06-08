"""
evaluation.py
-------------
Generate all evaluation results, tables, and figures for Group 15 – Problem 5:
Arabic Text Simplification Benchmarking.

Outputs (all saved to ./figures/):
    fig1_overall_sari_corrected.png   – Overall SARI (single vs multi-ref) + IAA line
    fig2_sari_by_type_corrected.png   – SARI by Arabic type, grouped bar chart
    fig3_length_ratio_corrected.png   – Average length ratio bar chart

Usage:
    python evaluation.py
    python evaluation.py --data-path /path/to/all_models_and_gold_CLEANED.json
    python evaluation.py --output-dir ./my_figures
"""

import json
import re
import os
import argparse
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.spines.top']   = False
matplotlib.rcParams['axes.spines.right'] = False


# ─────────────────────────────────────────────
#  Arabic normalization (copied from sari_metric.py for self-containment)
# ─────────────────────────────────────────────

def normalize_arabic(text: str) -> str:
    text = re.sub(r'[\u064B-\u065F\u0670\u0640]', '', text)
    text = re.sub(r'[إأآا]', 'ا', text)
    text = text.replace('ى', 'ي')
    text = text.replace('ؤ', 'ء').replace('ئ', 'ء')
    text = text.replace('ة', 'ه')
    text = re.sub(r'[^\w\s]', ' ', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def ngram_set(tokens, n):
    return set(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def compute_sari_multi(source, system, references, max_n=4):
    src_tok = normalize_arabic(source).split()
    sys_tok = normalize_arabic(system).split()
    ref_tok_list = [normalize_arabic(r).split() for r in references]

    add_scores, keep_scores, del_scores = [], [], []

    for n in range(1, max_n + 1):
        src_ng = ngram_set(src_tok, n)
        sys_ng = ngram_set(sys_tok, n)
        ref_union = set()
        for rt in ref_tok_list:
            ref_union |= ngram_set(rt, n)

        # ADD
        add_cand    = sys_ng - src_ng
        ref_new     = ref_union - src_ng
        add_correct = add_cand & ref_new
        p_add = len(add_correct) / len(add_cand) if add_cand else 0.0
        r_add = len(add_correct) / len(ref_new)  if ref_new  else 0.0
        f_add = 2 * p_add * r_add / (p_add + r_add) if (p_add + r_add) > 0 else 0.0

        # KEEP
        keep_cand    = src_ng & sys_ng
        keep_ref     = src_ng & ref_union
        keep_correct = keep_cand & keep_ref
        p_keep = len(keep_correct) / len(keep_cand) if keep_cand else 0.0
        r_keep = len(keep_correct) / len(keep_ref)  if keep_ref  else 0.0
        f_keep = 2 * p_keep * r_keep / (p_keep + r_keep) if (p_keep + r_keep) > 0 else 0.0

        # DELETE
        del_cand    = src_ng - sys_ng
        del_correct = del_cand - ref_union
        p_del = len(del_correct) / len(del_cand) if del_cand else 0.0

        add_scores.append(f_add)
        keep_scores.append(f_keep)
        del_scores.append(p_del)

    return (sum(add_scores) + sum(keep_scores) + sum(del_scores)) / (3 * max_n)


def compute_sari_single(source, system, reference, max_n=4):
    return compute_sari_multi(source, system, [reference], max_n)


# ─────────────────────────────────────────────
#  Jais invalid outputs
# ─────────────────────────────────────────────

JAIS_INVALID = {f'P0{i:02d}' for i in range(27, 41)}   # P027–P040 inclusive


# ─────────────────────────────────────────────
#  Compute all scores
# ─────────────────────────────────────────────

MODELS       = ['GPT', 'Gemini', 'ALLaM', 'Jais', 'Fanar']
ARABIC_TYPES = ['MSA', 'Classical', 'Dialect']

MODEL_COLORS = {
    'ALLaM' : '#2196F3',
    'Jais'  : '#E53935',
    'Fanar' : '#9C27B0',
    'Gemini': '#43A047',
    'GPT'   : '#FB8C00',
}

MODEL_LABELS = {
    'GPT'   : 'GPT-4o',
    'Gemini': 'Gemini 1.5 Pro',
    'ALLaM' : 'ALLaM 7B',
    'Jais'  : 'Jais 8B',
    'Fanar' : 'Fanar 9B',
}


def score_all(data):
    """Return (single_ref, multi_ref) dicts: model → type → list of scores."""
    single = {m: defaultdict(list) for m in MODELS}
    multi  = {m: defaultdict(list) for m in MODELS}

    for pid, entry in data.items():
        src  = entry['original']
        a1   = entry['annotator1_gold']
        a2   = entry['annotator2_gold']
        atype = entry['arabic_type']

        for model in MODELS:
            out = entry.get(model, '').strip()
            if model == 'Jais' and pid in JAIS_INVALID:
                continue
            if not out:
                continue

            s_single = compute_sari_single(src, out, a1)
            s_multi  = compute_sari_multi(src, out, [a1, a2])

            single[model][atype].append(s_single)
            multi[model][atype].append(s_multi)

    return single, multi


def score_iaa(data):
    """Bidirectional IAA SARI."""
    iaa = defaultdict(list)
    for _, entry in data.items():
        src = entry['original']
        a1  = entry['annotator1_gold']
        a2  = entry['annotator2_gold']
        atype = entry['arabic_type']

        s = (compute_sari_single(src, a1, a2) + compute_sari_single(src, a2, a1)) / 2
        iaa[atype].append(s)

    return {t: sum(v) / len(v) for t, v in iaa.items()}


def avg_all(d, model):
    all_s = [s for lst in d[model].values() for s in lst]
    return sum(all_s) / len(all_s) if all_s else 0.0


def avg_type(d, model, atype):
    lst = d[model].get(atype, [])
    return sum(lst) / len(lst) if lst else None


def length_ratios(data):
    ratios = defaultdict(list)
    for pid, entry in data.items():
        src_len = len(entry['original'].split())
        if src_len == 0:
            continue
        for model in MODELS:
            out = entry.get(model, '').strip()
            if model == 'Jais' and pid in JAIS_INVALID:
                continue
            if out:
                ratios[model].append(len(out.split()) / src_len)
        for ann in ['annotator1_gold', 'annotator2_gold']:
            gold = entry.get(ann, '').strip()
            if gold:
                ratios[ann].append(len(gold.split()) / src_len)
    return {k: sum(v) / len(v) for k, v in ratios.items()}


# ─────────────────────────────────────────────
#  Figure 1: Overall SARI (single + multi)
# ─────────────────────────────────────────────

def plot_fig1(single, multi, iaa_overall, output_dir):
    fig, ax = plt.subplots(figsize=(10, 6))

    model_order = ['ALLaM', 'Jais', 'Fanar', 'Gemini', 'GPT']
    x = np.arange(len(model_order))
    width = 0.35

    single_vals = [avg_all(single, m) for m in model_order]
    multi_vals  = [avg_all(multi,  m) for m in model_order]

    bars1 = ax.bar(x - width / 2, single_vals, width,
                   color=[MODEL_COLORS[m] for m in model_order],
                   alpha=1.0, label='Single-Reference', zorder=3)
    bars2 = ax.bar(x + width / 2, multi_vals, width,
                   color=[MODEL_COLORS[m] for m in model_order],
                   alpha=0.55, label='Multi-Reference', zorder=3)

    # Value labels
    for bar, val in zip(bars1, single_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    for bar, val in zip(bars2, multi_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=8)

    # IAA line
    ax.axhline(iaa_overall, color='red', linestyle='--', linewidth=1.6,
               label=f'Human IAA = {iaa_overall:.4f}', zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in model_order], fontsize=10)
    ax.set_ylabel('SARI Score', fontsize=11)
    ax.set_ylim(0, 0.70)
    ax.set_title(
        'Overall SARI Scores: Arabic Text Simplification (Task #15, Group 15)\n'
        'Corrected with Arabic Normalization',
        fontsize=12, fontweight='bold'
    )
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)

    plt.tight_layout()
    path = os.path.join(output_dir, 'fig1_overall_sari_corrected.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


# ─────────────────────────────────────────────
#  Figure 2: SARI by Arabic type
# ─────────────────────────────────────────────

def plot_fig2(multi, iaa_by_type, output_dir):
    fig, ax = plt.subplots(figsize=(12, 6))

    model_order = ['ALLaM', 'Jais', 'Fanar', 'Gemini', 'GPT']
    n_models = len(model_order)
    n_types  = len(ARABIC_TYPES)
    group_width = 0.8
    bar_width   = group_width / n_models

    x = np.arange(n_types)

    for i, model in enumerate(model_order):
        offsets = x + (i - n_models / 2 + 0.5) * bar_width
        vals = [avg_type(multi, model, t) for t in ARABIC_TYPES]

        for j, (offset, val) in enumerate(zip(offsets, vals)):
            if val is None:
                ax.bar(offset, 0.005, bar_width * 0.9,
                       color='lightgray', zorder=3, linewidth=0)
                ax.text(offset, 0.01, 'N/A', ha='center', va='bottom',
                        fontsize=7, color='gray')
            else:
                bar = ax.bar(offset, val, bar_width * 0.9,
                             color=MODEL_COLORS[model], zorder=3,
                             label=MODEL_LABELS[model] if j == 0 else '')
                ax.text(offset, val + 0.004, f'{val:.3f}',
                        ha='center', va='bottom', fontsize=7)

    # IAA lines per type group
    iaa_line_colors = {'MSA': 'red', 'Classical': 'red', 'Dialect': 'red'}
    for j, atype in enumerate(ARABIC_TYPES):
        if atype in iaa_by_type:
            iaa_val = iaa_by_type[atype]
            left  = x[j] - group_width / 2
            right = x[j] + group_width / 2
            ax.hlines(iaa_val, left, right, colors='red', linestyles='--',
                      linewidth=1.5, zorder=5)
            ax.text(right - 0.01, iaa_val + 0.005, f'IAA={iaa_val:.3f}',
                    ha='right', va='bottom', fontsize=7.5, color='red')

    ax.set_xticks(x)
    ax.set_xticklabels(ARABIC_TYPES, fontsize=12)
    ax.set_ylabel('SARI Score (Multi-Reference)', fontsize=11)
    ax.set_ylim(0, 0.75)
    ax.set_title(
        'SARI Scores by Arabic Type — Corrected (Group 15, Task #15)',
        fontsize=12, fontweight='bold'
    )
    ax.legend(fontsize=8.5, loc='upper right',
              handles=[mpatches.Patch(color=MODEL_COLORS[m], label=MODEL_LABELS[m])
                       for m in model_order])
    ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)

    plt.tight_layout()
    path = os.path.join(output_dir, 'fig2_sari_by_type_corrected.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


# ─────────────────────────────────────────────
#  Figure 3: Length ratios
# ─────────────────────────────────────────────

def plot_fig3(length_ratio_dict, output_dir):
    fig, ax = plt.subplots(figsize=(10, 6))

    label_order = ['annotator1_gold', 'annotator2_gold',
                   'ALLaM', 'Jais', 'Fanar', 'Gemini', 'GPT']
    display_names = {
        'annotator1_gold': 'Human A1',
        'annotator2_gold': 'Human A2',
        'ALLaM': 'ALLaM',
        'Jais':  'Jais',
        'Fanar': 'Fanar',
        'Gemini': 'Gemini',
        'GPT':   'GPT',
    }
    bar_colors = {
        'annotator1_gold': '#8D6E63',
        'annotator2_gold': '#EC407A',
        'ALLaM' : MODEL_COLORS['ALLaM'],
        'Jais'  : MODEL_COLORS['Jais'],
        'Fanar' : MODEL_COLORS['Fanar'],
        'Gemini': MODEL_COLORS['Gemini'],
        'GPT'   : MODEL_COLORS['GPT'],
    }

    vals  = [length_ratio_dict.get(k, 0) for k in label_order]
    names = [display_names[k] for k in label_order]
    cols  = [bar_colors[k] for k in label_order]

    bars = ax.bar(names, vals, color=cols, zorder=3, edgecolor='white', linewidth=0.5)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.axhline(1.0, color='red', linestyle='--', linewidth=1.6,
               label='Same length as original', zorder=4)

    ax.set_ylabel('Length Ratio (output words / input words)', fontsize=11)
    ax.set_ylim(0, 1.45)
    ax.set_title(
        'Average Length Ratio by Model (Group 15, Task #15)',
        fontsize=12, fontweight='bold'
    )
    ax.legend(fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)

    plt.tight_layout()
    path = os.path.join(output_dir, 'fig3_length_ratio_corrected.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


# ─────────────────────────────────────────────
#  Print results tables to console
# ─────────────────────────────────────────────

def print_tables(single, multi, iaa_by_type, lr):
    print('\n' + '='*70)
    print('TABLE 1: Overall SARI (Single-Ref vs Multi-Ref)')
    print('='*70)
    header = f"{'Model':<14} {'Single-Ref':>12} {'Multi-Ref':>12} {'Delta':>8} {'n valid':>8}"
    print(header)
    print('-' * len(header))
    model_order = ['GPT', 'Gemini', 'ALLaM', 'Fanar', 'Jais']
    for m in model_order:
        s = avg_all(single, m)
        r = avg_all(multi,  m)
        n = sum(len(lst) for lst in multi[m].values())
        print(f"{MODEL_LABELS[m]:<18} {s:>12.4f} {r:>12.4f} {r-s:>8.4f} {n:>8}")

    print('\n' + '='*70)
    print('TABLE 2: SARI by Arabic Type (Multi-Reference)')
    print('='*70)
    header = f"{'Model':<18} {'MSA':>10} {'Classical':>12} {'Dialect':>10} {'Overall':>10}"
    print(header)
    print('-' * len(header))
    for m in model_order:
        row = f"{MODEL_LABELS[m]:<18}"
        for t in ARABIC_TYPES:
            v = avg_type(multi, m, t)
            row += f" {v:>10.4f}" if v is not None else f" {'N/A':>10}"
        row += f" {avg_all(multi, m):>10.4f}"
        print(row)

    # IAA row
    iaa_row = f"{'Human IAA':<18}"
    for t in ARABIC_TYPES:
        v = iaa_by_type.get(t)
        iaa_row += f" {v:>10.4f}" if v is not None else f" {'---':>10}"
    all_iaa = [v for v in iaa_by_type.values()]
    iaa_row += f" {sum(all_iaa)/len(all_iaa):>10.4f}"
    print(iaa_row)

    print('\n' + '='*70)
    print('TABLE 3: Length Ratios')
    print('='*70)
    order = ['annotator1_gold', 'annotator2_gold', 'GPT', 'Gemini', 'ALLaM', 'Fanar', 'Jais']
    for k in order:
        v = lr.get(k, 0)
        flag = '  ← EXPANSION' if v > 1.0 else ''
        print(f"  {k:<20}: {v:.3f}{flag}")

    print('\n' + '='*70)
    print('IAA by Arabic Type')
    print('='*70)
    for t in ARABIC_TYPES + ['Overall']:
        v = iaa_by_type.get(t)
        if v is not None:
            print(f"  {t:<12}: {v:.4f}")


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Generate evaluation results and figures for Group 15 Arabic Simplification'
    )
    parser.add_argument('--data-path', default='all_models_and_gold_CLEANED.json')
    parser.add_argument('--output-dir', default='figures')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f'Loading data from: {args.data_path}')
    with open(args.data_path, encoding='utf-8') as f:
        data = json.load(f)

    print(f'Paragraphs loaded: {len(data)}')
    print('Computing SARI scores (this may take ~30 seconds)...\n')

    single, multi = score_all(data)
    iaa_by_type   = score_iaa(data)

    # Compute overall IAA
    all_iaa = []
    for _, entry in data.items():
        src = entry['original']
        a1  = entry['annotator1_gold']
        a2  = entry['annotator2_gold']
        s = (compute_sari_single(src, a1, a2) + compute_sari_single(src, a2, a1)) / 2
        all_iaa.append(s)
    iaa_overall = sum(all_iaa) / len(all_iaa)
    iaa_by_type['Overall'] = iaa_overall

    lr = length_ratios(data)

    # Print tables
    print_tables(single, multi, iaa_by_type, lr)

    # Generate figures
    print('\nGenerating figures...')
    plot_fig1(single, multi, iaa_overall,  args.output_dir)
    plot_fig2(multi,         iaa_by_type,  args.output_dir)
    plot_fig3(lr,                          args.output_dir)

    print('\nDone. All outputs saved to:', os.path.abspath(args.output_dir))


if __name__ == '__main__':
    main()