"""
sari_metric.py
--------------
SARI evaluation script for Arabic Text Simplification (Group 15, Problem 5).

Usage:
    python sari_metric.py                     # evaluate all models on all paragraphs
    python sari_metric.py --model GPT         # evaluate a single model
    python sari_metric.py --paragraph P001    # evaluate a single paragraph
    python sari_metric.py --by-type           # print breakdown by Arabic type
    python sari_metric.py --length-ratio      # print length ratio stats
    python sari_metric.py --iaa               # compute inter-annotator agreement

Requires: all_models_and_gold_CLEANED.json in the same directory (or use --data-path).
"""

import json
import re
import argparse
from collections import defaultdict

# ─────────────────────────────────────────────
#  Arabic Normalization
# ─────────────────────────────────────────────

def normalize_arabic(text: str) -> str:
    """
    Apply Arabic-specific normalization before SARI scoring.

    Steps:
        1. Remove diacritics (tashkeel: ً ٌ ٍ َ ُ ِ ّ ْ ٰ)
        2. Normalize alef variants (أ إ آ ا  →  ا)
        3. Normalize alef maqsura (ى → ي)
        4. Normalize hamza (ؤ ئ → ء)
        5. Normalize ta marbuta (ة → ه)
        6. Remove punctuation
        7. Lowercase and strip
    """
    # 1. Remove diacritics (Unicode block 0x064B–0x065F + tatweel 0x0640 + superscript alef 0x0670)
    text = re.sub(r'[\u064B-\u065F\u0670\u0640]', '', text)

    # 2. Normalize alef variants
    text = re.sub(r'[إأآا]', 'ا', text)

    # 3. Alef maqsura
    text = text.replace('ى', 'ي')

    # 4. Hamza variants
    text = text.replace('ؤ', 'ء').replace('ئ', 'ء')

    # 5. Ta marbuta
    text = text.replace('ة', 'ه')

    # 6. Remove punctuation (keep Arabic letters, digits, whitespace)
    text = re.sub(r'[^\w\s]', ' ', text)

    # 7. Lowercase, collapse whitespace, strip
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# ─────────────────────────────────────────────
#  N-gram helpers
# ─────────────────────────────────────────────

def get_ngrams(tokens: list, n: int) -> list:
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def ngram_set(tokens: list, n: int) -> set:
    return set(get_ngrams(tokens, n))


# ─────────────────────────────────────────────
#  SARI implementation
# ─────────────────────────────────────────────

def compute_sari(
    source: str,
    system: str,
    references: list,
    max_n: int = 4,
) -> dict:
    """
    Compute SARI between a system output and one or more references,
    given the source text.

    Returns a dict with:
        'sari'  : overall SARI score
        'add'   : average F1_ADD across n-grams
        'keep'  : average F1_KEEP across n-grams
        'delete': average P_DEL across n-grams
    """
    src_tokens  = normalize_arabic(source).split()
    sys_tokens  = normalize_arabic(system).split()
    ref_tokens_list = [normalize_arabic(r).split() for r in references]

    add_scores, keep_scores, del_scores = [], [], []

    for n in range(1, max_n + 1):
        src_ng  = ngram_set(src_tokens, n)
        sys_ng  = ngram_set(sys_tokens, n)

        # Multi-reference union
        ref_ng_union = set()
        for rt in ref_tokens_list:
            ref_ng_union |= ngram_set(rt, n)

        # ── ADD ──────────────────────────────────────────
        # n-grams added by system (not in source), rewarded if in any reference
        add_cand    = sys_ng - src_ng
        ref_new     = ref_ng_union - src_ng  # what references add vs source
        add_correct = add_cand & ref_new

        p_add = len(add_correct) / len(add_cand) if add_cand else 0.0
        r_add = len(add_correct) / len(ref_new)  if ref_new  else 0.0
        f_add = 2 * p_add * r_add / (p_add + r_add) if (p_add + r_add) > 0 else 0.0

        # ── KEEP ─────────────────────────────────────────
        # n-grams from source kept in system, rewarded if also in reference
        keep_cand    = src_ng & sys_ng
        keep_ref     = src_ng & ref_ng_union
        keep_correct = keep_cand & keep_ref

        p_keep = len(keep_correct) / len(keep_cand) if keep_cand else 0.0
        r_keep = len(keep_correct) / len(keep_ref)  if keep_ref  else 0.0
        f_keep = 2 * p_keep * r_keep / (p_keep + r_keep) if (p_keep + r_keep) > 0 else 0.0

        # ── DELETE ───────────────────────────────────────
        # n-grams from source removed by system, rewarded if not in any reference
        del_cand    = src_ng - sys_ng
        del_correct = del_cand - ref_ng_union

        p_del = len(del_correct) / len(del_cand) if del_cand else 0.0
        # DEL uses precision only (standard SARI formulation)

        add_scores.append(f_add)
        keep_scores.append(f_keep)
        del_scores.append(p_del)

    add_avg  = sum(add_scores)  / max_n
    keep_avg = sum(keep_scores) / max_n
    del_avg  = sum(del_scores)  / max_n
    sari     = (add_avg + keep_avg + del_avg) / 3

    return {
        'sari'  : sari,
        'add'   : add_avg,
        'keep'  : keep_avg,
        'delete': del_avg,
    }


# ─────────────────────────────────────────────
#  Jais validity check
# ─────────────────────────────────────────────

# Based on character-level comparison:
#   P027–P039: verbatim echoes of the source
#   P040:      partial echo with country name substituted → treated as invalid
JAIS_INVALID = {f'P0{i:02d}' for i in range(27, 41)}  # P027–P040


def jais_is_valid(pid: str) -> bool:
    return pid not in JAIS_INVALID


# ─────────────────────────────────────────────
#  Main evaluation
# ─────────────────────────────────────────────

def evaluate(data: dict, models: list) -> dict:
    """
    Evaluate all models on all paragraphs.
    Returns nested dict: results[model][pid] = sari_dict
    """
    results = {m: {} for m in models}

    for pid, entry in data.items():
        src  = entry['original']
        refs = [entry['annotator1_gold'], entry['annotator2_gold']]

        for model in models:
            output = entry.get(model, '').strip()

            # Skip Jais invalid outputs
            if model == 'Jais' and not jais_is_valid(pid):
                continue

            if not output:
                continue

            results[model][pid] = {
                **compute_sari(src, output, refs),
                'arabic_type': entry['arabic_type'],
                'domain':      entry.get('domain', ''),
            }

    return results


def compute_iaa(data: dict) -> dict:
    """Compute bidirectional SARI IAA between Annotator 1 and Annotator 2."""
    iaa = defaultdict(list)
    for pid, entry in data.items():
        src = entry['original']
        a1  = entry['annotator1_gold']
        a2  = entry['annotator2_gold']
        atype = entry['arabic_type']

        s12 = compute_sari(src, a1, [a2])['sari']
        s21 = compute_sari(src, a2, [a1])['sari']
        iaa[atype].append((s12 + s21) / 2)
        iaa['Overall'].append((s12 + s21) / 2)

    summary = {}
    for atype, scores in iaa.items():
        summary[atype] = sum(scores) / len(scores)
    return summary


def compute_length_ratios(data: dict, models: list) -> dict:
    ratios = {m: [] for m in models + ['annotator1_gold', 'annotator2_gold']}

    for pid, entry in data.items():
        src_len = len(entry['original'].split())
        if src_len == 0:
            continue

        for model in models:
            output = entry.get(model, '').strip()
            if model == 'Jais' and not jais_is_valid(pid):
                continue
            if output:
                ratios[model].append(len(output.split()) / src_len)

        for ann in ['annotator1_gold', 'annotator2_gold']:
            gold = entry.get(ann, '').strip()
            if gold:
                ratios[ann].append(len(gold.split()) / src_len)

    return {k: sum(v) / len(v) for k, v in ratios.items() if v}


def print_summary_table(results: dict, models: list):
    types = ['MSA', 'Classical', 'Dialect']

    header = f"{'Model':<14} {'MSA':>10} {'Classical':>12} {'Dialect':>10} {'Overall':>10} {'n valid':>8}"
    print(header)
    print('-' * len(header))

    for model in models:
        by_type = defaultdict(list)
        for pid, r in results[model].items():
            by_type[r['arabic_type']].append(r['sari'])

        row = f"{model:<14}"
        for t in types:
            scores = by_type.get(t, [])
            if scores:
                row += f" {sum(scores)/len(scores):>10.4f}"
            else:
                row += f" {'---':>10}"

        all_scores = [r['sari'] for r in results[model].values()]
        n_valid    = len(all_scores)
        overall    = sum(all_scores) / n_valid if all_scores else 0.0
        row += f" {overall:>10.4f} {n_valid:>8}"
        print(row)


# ─────────────────────────────────────────────
#  CLI entry point
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='SARI evaluation for Arabic Text Simplification (Group 15)')
    parser.add_argument('--data-path', default='all_models_and_gold_CLEANED.json',
                        help='Path to the JSON dataset file')
    parser.add_argument('--model',  default=None, help='Evaluate only this model')
    parser.add_argument('--paragraph', default=None, help='Evaluate only this paragraph ID (e.g. P001)')
    parser.add_argument('--by-type', action='store_true', help='Print breakdown by Arabic type')
    parser.add_argument('--length-ratio', action='store_true', help='Print length ratio statistics')
    parser.add_argument('--iaa', action='store_true', help='Compute inter-annotator agreement')
    parser.add_argument('--verbose', action='store_true', help='Print per-paragraph scores')
    args = parser.parse_args()

    with open(args.data_path, encoding='utf-8') as f:
        data = json.load(f)

    all_models = ['GPT', 'Gemini', 'ALLaM', 'Jais', 'Fanar']
    models = [args.model] if args.model else all_models

    # Filter paragraphs if requested
    if args.paragraph:
        data = {args.paragraph: data[args.paragraph]}

    # ── IAA ──────────────────────────────────────────────────
    if args.iaa:
        print('\n=== Inter-Annotator Agreement (SARI-based, bidirectional) ===')
        iaa = compute_iaa(data)
        for atype in ['MSA', 'Classical', 'Dialect', 'Overall']:
            if atype in iaa:
                print(f'  {atype:<12}: {iaa[atype]:.4f}')
        print()

    # ── Main evaluation ───────────────────────────────────────
    results = evaluate(data, models)

    print('\n=== SARI Scores (Multi-Reference, Arabic Normalized) ===\n')
    print_summary_table(results, models)

    # ── Verbose per-paragraph ────────────────────────────────
    if args.verbose:
        print('\n=== Per-Paragraph Scores ===')
        for model in models:
            print(f'\n{model}:')
            for pid in sorted(results[model].keys()):
                r = results[model][pid]
                print(f'  {pid} ({r["arabic_type"]:<10}) SARI={r["sari"]:.4f}  '
                      f'ADD={r["add"]:.4f}  KEEP={r["keep"]:.4f}  DEL={r["delete"]:.4f}')

    # ── Length ratios ────────────────────────────────────────
    if args.length_ratio:
        print('\n=== Length Ratios (output words / source words) ===')
        ratios = compute_length_ratios(data, all_models)
        for label in ['annotator1_gold', 'annotator2_gold'] + all_models:
            if label in ratios:
                flag = ' ← EXPANSION' if ratios[label] > 1.0 else ''
                print(f'  {label:<20}: {ratios[label]:.3f}{flag}')

    print()


if __name__ == '__main__':
    main()