#!/usr/bin/env python3
"""
Pipeline 3: LLM-Based Automatic Labelling with Agreement Validation
====================================================================
Leverages pre-trained LLMs as zero-shot labellers.  Inter-model agreement
is used as a proxy for label confidence; human effort is reserved for the
practical ground-truth set already required by the assignment, and for
disagreement resolution.

Steps
-----
1. Benchmark 5 candidate LLMs on the practical ground-truth subset
2. Select the two top-performing LLMs (preferring different families)
3. Full dataset labelling by both LLMs (10,000 images each)
4. Agreement-based validation  &  disagreement resolution
5. Corrective actions if target accuracy is not achieved
"""

import os
import re
import sys
import base64
import io
import time
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from collections import Counter
from datetime import datetime

# ---------------------------------------------------------------------------
# Path setup -- add parent so we can import the accuracy oracle
# ---------------------------------------------------------------------------
PIPELINE_DIR = Path(__file__).resolve().parent
ASSIGNMENT_DIR = PIPELINE_DIR.parent
sys.path.insert(0, str(ASSIGNMENT_DIR))
from check_accuracy import check_accuracy

from dotenv import load_dotenv
load_dotenv(PIPELINE_DIR / ".env")

# ===================================================================
#  CONFIGURATION
# ===================================================================
IMAGE_DIR       = Path("F:/Uni/EECE 4/Semester 2/Neural Networks/Assignments/Assign1/Indian_Digits_Train")
GROUND_TRUTH    = Path("F:/Uni/EECE 4/Semester 2/Neural Networks/Assignments/Assign1/true_labels.csv")
OUTPUT_DIR      = PIPELINE_DIR / "output"
CACHE_DIR       = PIPELINE_DIR / "cache"

BENCHMARK_SIZE  = 500       # practical ground-truth subset size
BENCHMARK_SEED  = 42
TOTAL_IMAGES    = 10_000
ACCURACY_TARGET = 0.99      # 99 %

# Few-shot teaching examples -- one representative image per digit (0-based index)
# These are selected from the ground truth so their labels are known.
FEW_SHOT_EXAMPLES = {
    0: 26,   # index 26 -> label 0
    1: 3,    # index 3  -> label 1
    2: 19,   # index 19 -> label 2
    3: 6,    # index 6  -> label 3
    4: 14,   # index 14 -> label 4
    5: 2,    # index 2  -> label 5
    6: 1,    # index 1  -> label 6
    7: 7,    # index 7  -> label 7
    8: 20,   # index 20 -> label 8
    9: 0,    # index 0  -> label 9
}

SYSTEM_PROMPT = (
    "You are a specialist in reading handwritten Eastern Arabic-Indic numerals "
    "(Hindu-Arabic numerals) as written in Egypt and Arab countries. You have studied thousands of "
    "examples of these numerals in various handwriting styles. You never confuse them with Western "
    "digits because you understand their distinct visual shapes."
)

USER_PROMPT = (
    "This image contains exactly ONE handwritten Eastern Arabic-Indic numeral "
    "written on a white background. All images are upright — no rotation.\n\n"
    "Your task: identify the numeral and return its Western digit equivalent (0–9).\n\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    "VISUAL SHAPE GUIDE:\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    '0 → ٠  A TINY DOT or extremely small speck. Much smaller than any other shape.\n'
    '1 → ١  A single VERTICAL STROKE, like a thin upright line or Western "1".\n'
    '2 → ٢  An open curve sweeping RIGHT, like a reversed "C" or fishhook opening right.\n'
    '3 → ٣  TWO BUMPS stacked vertically, open on the RIGHT side. Like a rounder Western "3".\n'
    '        The bumps open to the RIGHT — use the example image above as reference.\n'
    '4 → ٤  MIRRORED "3" — bumps open to the LEFT. Like epsilon (ε) or a zigzag.\n'
    '        The bumps open to the LEFT — use the example image above as reference.\n'
    '5 → ٥  A LARGE CLOSED CIRCLE, like "O" or Western "0". Full loop, no gap.\n'
    '        Use the example image above as reference.\n'
    '6 → ٦  A DIAGONAL SLASH from upper-left to lower-right. Looks EXACTLY like Western "7".\n'
    '7 → ٧  A V-SHAPE or downward checkmark (✓). Point at the BOTTOM.\n'
    '8 → ٨  An INVERTED V or caret (^). Point at the TOP. Opposite of 7.\n'
    '9 → ٩  A HOOK opening to the LEFT, like a reversed "9" or letter "J".\n'
    '        Use the example image above as reference.\n\n'
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    "CRITICAL RULES:\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    '- Diagonal line like Western "7"  →  output 6  (NOT 7)\n'
    '- V-shape pointing DOWN           →  output 7  (NOT 6)\n'
    '- Closed full circle "O"          →  output 5  (NOT 0)\n'
    '- Tiny dot or speck               →  output 0  (NOT 5)\n'
    '- Caret ^ pointing UP             →  output 8\n'
    '- Hook opening LEFT               →  output 9\n'
    '- Bumps open RIGHT                →  output 3\n'
    '- Bumps open LEFT                 →  output 4\n\n'
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    "DECISION STEPS (follow in order):\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    '1. Tiny dot or speck?                        → 0\n'
    '2. Large closed circle like western "O" must be closed or nearly closed?             → 5\n'
    '3. Single straight vertical stroke?         → 1\n'
    '4. Diagonal slash (like Western "7")?        → 6  ⚠ NOT 7\n'
    '5. V-shape with point at BOTTOM?             → 7  ⚠ NOT 6\n'
    '6. Caret with point at TOP?                  → 8\n'
    '7. similar to the western 9 or a hook to the LEFT?                        → 9\n'
    '8. Open rightward curve (reversed C)?        → 2\n'
    '9. Two bumps open to the RIGHT?              → 3  (see example)\n'
    '10. mirrored western 3 or zigzag ?              → 4  (see example)\n\n'
    "Reply with ONLY one digit (0–9). No explanation, no punctuation, nothing else."
)

# Candidate models -- must have vision / image-input support
# (Gemini commented out -- uncomment to re-enable)
CANDIDATE_MODELS = [
    #{"name": "gpt-5.4",          "provider": "openai"},
    #{"name": "gpt-4.1",          "provider": "openai"},
    {"name": "gpt-4.1-mini",     "provider": "openai"},
    # {"name": "gemini-2.5-flash", "provider": "google"},
    # {"name": "gemini-2.0-flash", "provider": "google"},
]

# ===================================================================
#  API CLIENT SETUP
# ===================================================================
from openai import OpenAI
from google import genai

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# google_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))  # re-enable with Gemini

# ===================================================================
#  UTILITIES
# ===================================================================

def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def load_ground_truth():
    """Load full ground truth. Returns dict {image_index: label}."""
    df = pd.read_csv(GROUND_TRUTH)
    return dict(zip(df["index"].astype(int), df["label"].astype(int)))


def get_benchmark_indices(gt_dict, size=BENCHMARK_SIZE, seed=BENCHMARK_SEED):
    """Select a reproducible random subset for benchmarking."""
    rng = np.random.RandomState(seed)
    pool = sorted(gt_dict.keys())
    chosen = rng.choice(pool, size=min(size, len(pool)), replace=False)
    return sorted(chosen.tolist())


def idx_to_path(idx):
    """0-based index -> image file path  (index 0 -> 1.bmp)."""
    p = IMAGE_DIR / f"{idx + 1}.bmp"
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")
    return p


def _encode_b64(image_path, scale=224):
    """Return a base-64 PNG string, optionally upscaled."""
    with Image.open(image_path) as img:
        if scale:
            img = img.resize((scale, scale), Image.NEAREST)
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()


def _open_pil(image_path, scale=224):
    """Return an RGB PIL Image, optionally upscaled."""
    img = Image.open(image_path)
    if scale:
        img = img.resize((scale, scale), Image.NEAREST)
    return img.convert("RGB")


def _parse_digit(text):
    """Extract a single digit 0-9 from LLM output, or -1 on failure."""
    text = text.strip()
    if len(text) == 1 and text.isdigit():
        return int(text)
    for ch in text:
        if ch.isdigit():
            return int(ch)
    return -1

# ===================================================================
#  FEW-SHOT TEACHING BATCH  (built once, reused per request)
# ===================================================================

_few_shot_cache = {}   # provider -> list of content parts

def _build_few_shot_openai():
    """Build the few-shot example messages for OpenAI."""
    if "openai" in _few_shot_cache:
        return _few_shot_cache["openai"]
    messages = []
    for digit in range(10):
        idx = FEW_SHOT_EXAMPLES[digit]
        b64 = _encode_b64(idx_to_path(idx))
        messages.append({
            "role": "user",
            "content": [
                {"type": "input_text", "text": f"What digit is this? (example {digit})"},
                {"type": "input_image",
                 "image_url": f"data:image/png;base64,{b64}"},
            ],
        })
        messages.append({
            "role": "assistant",
            "content": [{"type": "output_text", "text": str(digit)}],
        })
    _few_shot_cache["openai"] = messages
    return messages


def _build_few_shot_google():
    """Build the few-shot contents list for Google."""
    if "google" in _few_shot_cache:
        return _few_shot_cache["google"]
    parts = []
    for digit in range(10):
        idx = FEW_SHOT_EXAMPLES[digit]
        pil_img = _open_pil(idx_to_path(idx))
        parts.append(f"Example -- this is digit {digit}:")
        parts.append(pil_img)
    _few_shot_cache["google"] = parts
    return parts

# ===================================================================
#  PREDICTION  (provider dispatch + retry + rate-limit awareness)
# ===================================================================

_google_last_call = 0.0
GOOGLE_MIN_INTERVAL = 13.0   # free tier: 5 req/min -> 12 s + buffer

# OpenAI TPM throttle: each request ~3500 tokens; TPM limit 30000
# -> max ~8 req/min -> minimum 9 s between calls (with buffer)
_openai_last_call = 0.0
OPENAI_MIN_INTERVAL = 9.0

def _predict_openai(model_name, image_path, retries=6):
    global _openai_last_call
    b64 = _encode_b64(image_path)
    few_shot = _build_few_shot_openai()
    messages = [
        {"role": "developer", "content": SYSTEM_PROMPT},
    ] + few_shot + [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": USER_PROMPT},
                {"type": "input_image",
                 "image_url": f"data:image/png;base64,{b64}"},
            ],
        },
    ]
    for attempt in range(retries):
        # proactive throttle: enforce minimum interval before every call
        elapsed = time.time() - _openai_last_call
        if elapsed < OPENAI_MIN_INTERVAL:
            time.sleep(OPENAI_MIN_INTERVAL - elapsed)
        try:
            _openai_last_call = time.time()
            resp = openai_client.responses.create(
                model=model_name,
                input=messages,
            )
            return _parse_digit(resp.output[0].content[0].text)
        except Exception as exc:
            exc_str = str(exc)
            if "429" in exc_str or "rate_limit" in exc_str.lower():
                m = re.search(r'try again in ([\d.]+)s', exc_str)
                wait = float(m.group(1)) + 2.0 if m else 15.0 * (attempt + 1)
                print(f"      ERR {model_name}: {exc}")
                print(f"      rate-limited, waiting {wait:.1f}s ...")
                time.sleep(wait)
            elif attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"      ERR {model_name}: {exc}")
                return -1
    return -1


def _predict_google(model_name, image_path, retries=5):
    global _google_last_call
    pil_img = _open_pil(image_path)
    few_shot = _build_few_shot_google()
    contents = [SYSTEM_PROMPT] + few_shot + [USER_PROMPT, pil_img]
    for attempt in range(retries):
        # rate-limit: wait if needed
        elapsed = time.time() - _google_last_call
        if elapsed < GOOGLE_MIN_INTERVAL:
            time.sleep(GOOGLE_MIN_INTERVAL - elapsed)
        try:
            _google_last_call = time.time()
            resp = google_client.models.generate_content(
                model=model_name,
                contents=contents,
            )
            return _parse_digit(resp.text)
        except Exception as exc:
            exc_str = str(exc)
            if "429" in exc_str or "RESOURCE_EXHAUSTED" in exc_str:
                retry_s = 45 + attempt * 15
                print(f"      rate-limited, waiting {retry_s}s …")
                time.sleep(retry_s)
            elif attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"      ERR {model_name}: {exc}")
                return -1


def predict(model_info, image_path):
    prov = model_info["provider"]
    if prov == "openai":
        return _predict_openai(model_info["name"], image_path)
    if prov == "google":
        return _predict_google(model_info["name"], image_path)
    raise ValueError(f"Unknown provider: {prov}")

# ===================================================================
#  CACHING
# ===================================================================

def _cache_path(model_name, phase):
    safe = model_name.replace("/", "_").replace(":", "_")
    return CACHE_DIR / f"{phase}_{safe}.csv"


def _load_cache(model_name, phase):
    p = _cache_path(model_name, phase)
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    return dict(zip(df["index"].astype(int), df["prediction"].astype(int)))


def _save_cache(model_name, phase, preds):
    p = _cache_path(model_name, phase)
    df = pd.DataFrame(
        [{"index": i, "prediction": v} for i, v in sorted(preds.items()) if v != -1]
    )
    df.to_csv(p, index=False)

# ===================================================================
#  CHECKPOINT HELPER
# ===================================================================

def _ask_continue(n_done, model_name, phase):
    """Pause at a checkpoint and ask whether to continue.
    If the user says no, progress is saved and the script exits cleanly.
    The cache is KEPT so the next run resumes from this point."""
    print(f"\n  -- Checkpoint: {n_done} prompts completed --")
    try:
        ans = input("  Continue? [y/n] (Enter = yes): ").strip().lower()
    except EOFError:
        ans = "y"
    if ans == "" or ans.startswith("y"):
        return
    # user said no -- keep cache so next run resumes from here
    print(f"  Progress saved ({n_done} done). Run again to resume.")
    sys.exit(0)

# ===================================================================
#  STEP 1 -- BENCHMARK
# ===================================================================

def step1_benchmark(gt, bench_idx):
    hdr = "STEP 1: BENCHMARK 5 LLMs ON PRACTICAL GROUND-TRUTH SET"
    print(f"\n{'='*70}\n{hdr}\n  ({len(bench_idx)} images, {len(CANDIDATE_MODELS)} models)\n{'='*70}")

    results = {}
    for minfo in CANDIDATE_MODELS:
        name = minfo["name"]
        print(f"\n  Model: {name} ({minfo['provider']})")

        cached = _load_cache(name, "bench")
        preds = dict(cached)
        # include previously errored (-1) items so they are retried
        todo = [i for i in bench_idx if preds.get(i, -1) == -1]

        if todo:
            n_valid_cached = sum(1 for v in preds.values() if v != -1)
            print(f"    cached={n_valid_cached}, remaining={len(todo)}")
            t0 = time.time()
            n_cached = n_valid_cached
            for n, idx in enumerate(todo, 1):
                try:
                    pred = predict(minfo, idx_to_path(idx))
                except Exception as exc:
                    print(f"      !! Unhandled error on img {idx+1}: {exc} -- skipping")
                    pred = -1

                elapsed = time.time() - t0
                rate    = n / elapsed if elapsed else 0

                if pred == -1:
                    # do NOT store -1; image will be retried on next run
                    done    = [i for i in bench_idx if preds.get(i, -1) != -1]
                    correct = sum(1 for i in done if preds[i] == gt[i])
                    run_acc = correct / len(done) if done else 0.0
                    print(f"    [{n}/{len(todo)}] img={idx+1:>5}  gt={gt[idx]} pred=ERR [skip]"
                          f"  acc={run_acc:.4f} ({correct}/{len(done)})  {rate:.2f} img/s")
                else:
                    preds[idx] = pred
                    # save after every successful prediction
                    _save_cache(name, "bench", preds)

                    # per-image running accuracy over all bench items seen so far
                    done    = [i for i in bench_idx if preds.get(i, -1) != -1]
                    correct = sum(1 for i in done if preds[i] == gt[i])
                    run_acc = correct / len(done) if done else 0.0
                    ok_sym  = "OK" if pred == gt[idx] else "XX"
                    print(f"    [{n}/{len(todo)}] img={idx+1:>5}  gt={gt[idx]} pred={pred} {ok_sym}"
                          f"  acc={run_acc:.4f} ({correct}/{len(done)})  {rate:.2f} img/s")

                # checkpoint every 100 total prompts for this model
                total_done = n_cached + n
                if total_done % 100 == 0:
                    _ask_continue(total_done, name, "bench")
        else:
            print(f"    all {len(bench_idx)} cached [done]")

        correct = sum(1 for i in bench_idx if preds.get(i, -1) == gt[i])
        valid   = sum(1 for i in bench_idx if preds.get(i, -1) != -1)
        errors  = len(bench_idx) - valid
        acc     = correct / valid if valid else 0.0

        results[name] = dict(accuracy=acc, correct=correct, total=valid,
                             errors=errors, provider=minfo["provider"])
        print(f"    accuracy={acc:.4f}  ({correct}/{valid})  errors={errors}")

    return results

# ===================================================================
#  STEP 2 -- SELECT TOP TWO
# ===================================================================

def step2_select(bench_res):
    print(f"\n{'='*70}\nSTEP 2: SELECT TOP TWO LLMs\n{'='*70}")

    ranked = sorted(bench_res.items(), key=lambda kv: kv[1]["accuracy"], reverse=True)
    print("\n  Rank | Model               | Provider | Accuracy")
    print("  " + "-"*56)
    for i, (n, r) in enumerate(ranked, 1):
        print(f"  {i:>4} | {n:<19} | {r['provider']:<8} | {r['accuracy']:.4f}")

    top1 = ranked[0]
    # prefer a different provider for the second model
    top2 = None
    for n, r in ranked[1:]:
        if r["provider"] != top1[1]["provider"]:
            top2 = (n, r)
            break
    if top2 is None:
        top2 = ranked[1]

    a_info = next(m for m in CANDIDATE_MODELS if m["name"] == top1[0])
    b_info = next(m for m in CANDIDATE_MODELS if m["name"] == top2[0])
    print(f"  [A] Model A: {a_info['name']} ({a_info['provider']})  acc={top1[1]['accuracy']:.4f}")
    print(f"  [B] Model B: {b_info['name']} ({b_info['provider']})  acc={top2[1]['accuracy']:.4f}")

    lo = min(top1[1]["accuracy"], top2[1]["accuracy"])
    if lo < 0.90:
        print(f"\n  [!] Both models below 90 % -- prompt refinement recommended (Step 5).")

    return a_info, b_info

# ===================================================================
#  STEP 3 -- FULL-DATASET LABELLING
# ===================================================================

def _label_full(minfo):
    """Label all 10 000 images with one model (resumable)."""
    name = minfo["name"]
    print(f"\n  Labelling 10,000 images with {name} …")

    cached = _load_cache(name, "full")
    preds = dict(cached)
    # include previously errored (-1) items so they are retried
    todo = [i for i in range(TOTAL_IMAGES) if preds.get(i, -1) == -1]

    if not todo:
        print(f"    all 10,000 cached [done]")
        return preds

    n_valid_cached = sum(1 for v in preds.values() if v != -1)
    print(f"    cached={n_valid_cached}, remaining={len(todo)}")
    t0 = time.time()
    n_cached = n_valid_cached
    n_saved = 0
    for n, idx in enumerate(todo, 1):
        try:
            pred = predict(minfo, idx_to_path(idx))
        except Exception as exc:
            print(f"      !! Unhandled error on img {idx+1}: {exc} -- skipping")
            pred = -1

        if pred != -1:
            preds[idx] = pred
            n_saved += 1
            # save every 50 successful predictions
            if n_saved % 50 == 0:
                _save_cache(name, "full", preds)

        if n % 100 == 0 or n == len(todo):
            el   = time.time() - t0
            rate = n / el if el else 0
            eta  = (len(todo) - n) / rate / 60 if rate else 0
            n_valid = sum(1 for v in preds.values() if v != -1)
            print(f"    [{n}/{len(todo)}]  {rate:.1f} img/s  ETA {eta:.1f} min  valid={n_valid}")

        # checkpoint every 100 total prompts for this model
        if (n_cached + n) % 100 == 0:
            _ask_continue(n_cached + n, name, "full")

    _save_cache(name, "full", preds)
    return preds


def step3_label(a_info, b_info):
    print(f"\n{'='*70}\nSTEP 3: FULL-DATASET LABELLING (10,000 x 2 models)\n{'='*70}")
    pa = _label_full(a_info)
    pb = _label_full(b_info)
    return pa, pb

# ===================================================================
#  STEP 4 -- AGREEMENT VALIDATION
# ===================================================================

def step4_validate(pa, pb, a_name, b_name, gt, bench_idx):
    print(f"\n{'='*70}\nSTEP 4: AGREEMENT-BASED VALIDATION\n{'='*70}")

    agreed, disagreed, errors = [], [], []
    for i in range(TOTAL_IMAGES):
        va, vb = pa.get(i, -1), pb.get(i, -1)
        if va == -1 or vb == -1:
            errors.append(i)
        elif va == vb:
            agreed.append(i)
        else:
            disagreed.append(i)

    n_agr  = len(agreed)
    n_dis  = len(disagreed)
    n_err  = len(errors)
    print(f"\n  Agreed:    {n_agr:>6}  ({100*n_agr/TOTAL_IMAGES:.1f} %)")
    print(f"  Disagreed: {n_dis:>6}  ({100*n_dis/TOTAL_IMAGES:.1f} %)")
    print(f"  API errors:{n_err:>6}")

    # 4a -- accuracy of agreed labels on benchmark overlap
    bench_set   = set(bench_idx)
    agr_bench   = [i for i in agreed if i in bench_set]
    agr_correct = sum(1 for i in agr_bench if pa[i] == gt[i])
    agr_total   = len(agr_bench)
    agr_acc     = agr_correct / agr_total if agr_total else 0.0

    print(f"\n  4a  Agreed-label accuracy on GT overlap: "
          f"{agr_acc:.4f}  ({agr_correct}/{agr_total})")
    if agr_acc >= ACCURACY_TARGET:
        print(f"      [OK] >= {ACCURACY_TARGET:.0%}  ->  accept all agreed labels")
    else:
        print(f"      [!!] < {ACCURACY_TARGET:.0%}  ->  corrective actions may follow (Step 5)")

    # Build final labels
    final = {}
    for i in agreed:
        final[i] = pa[i]

    # 4b -- disagreement resolution (use GT = simulated human annotation)
    human_count = 0
    auto_count  = 0
    for i in disagreed:
        if i in gt:                       # "human" labels it from GT
            final[i] = gt[i]
            human_count += 1
        else:                             # fallback to Model A
            final[i] = pa[i]
            auto_count += 1

    for i in errors:
        if i in gt:
            final[i] = gt[i]
        elif pa.get(i, -1) != -1:
            final[i] = pa[i]
        elif pb.get(i, -1) != -1:
            final[i] = pb[i]
        else:
            final[i] = 0

    manual_s = n_dis * 10
    print(f"\n  4b  Disagreement resolution:")
    print(f"      GT-resolved (human):     {human_count}")
    print(f"      Auto-resolved (Model A): {auto_count}")
    print(f"      Estimated manual time:   {manual_s} s  ({manual_s/60:.1f} min)")

    return dict(
        final=final,
        agreed=agreed, disagreed=disagreed, errors=errors,
        agr_acc=agr_acc, agr_correct=agr_correct, agr_total=agr_total,
        human_count=human_count, auto_count=auto_count,
        manual_s=manual_s,
    )

# ===================================================================
#  STEP 5 -- CORRECTIVE ACTIONS
# ===================================================================

def step5_correct(s4, pa, pb, a_info, b_info, gt, bench_res):
    print(f"\n{'='*70}\nSTEP 5: CORRECTIVE ACTIONS\n{'='*70}")

    if s4["agr_acc"] >= ACCURACY_TARGET:
        print("  No correction needed -- agreed accuracy meets target.")
        return s4["final"]

    print(f"  Agreed accuracy {s4['agr_acc']:.4f} < {ACCURACY_TARGET}")
    print("  Applying majority-vote with 3rd-ranked tie-breaker …")

    ranked = sorted(bench_res.items(), key=lambda kv: kv[1]["accuracy"], reverse=True)
    used   = {a_info["name"], b_info["name"]}
    tb     = None
    for n, _ in ranked:
        if n not in used:
            tb = next(m for m in CANDIDATE_MODELS if m["name"] == n)
            break
    if tb is None:
        print("  No tie-breaker available -- keeping current labels.")
        return s4["final"]

    print(f"  Tie-breaker: {tb['name']} ({tb['provider']})")

    need = [i for i in s4["disagreed"] if i not in gt]
    if not need:
        print("  All disagreements already resolved via GT.")
        return s4["final"]

    cached  = _load_cache(tb["name"], "tiebreak")
    tb_pred = dict(cached)
    todo    = [i for i in need if i not in tb_pred]

    if todo:
        print(f"  Running tie-breaker on {len(todo)} images ...")
        for n, idx in enumerate(todo, 1):
            pred = predict(tb, idx_to_path(idx))
            if pred != -1:
                tb_pred[idx] = pred
            if n % 50 == 0 or n == len(todo):
                print(f"    [{n}/{len(todo)}]")
                _save_cache(tb["name"], "tiebreak", tb_pred)
        _save_cache(tb["name"], "tiebreak", tb_pred)

    final = dict(s4["final"])
    for i in need:
        votes = [v for v in (pa.get(i, -1), pb.get(i, -1), tb_pred.get(i, -1)) if v != -1]
        if votes:
            final[i] = Counter(votes).most_common(1)[0][0]

    return final

# ===================================================================
#  REPORTING
# ===================================================================

def _draw_block(s4, a_info, b_info, acc, n_cor, n_tot):
    w = 58
    agr  = len(s4["agreed"])
    dis  = len(s4["disagreed"])
    pag  = 100 * agr / TOTAL_IMAGES
    pdis = 100 * dis / TOTAL_IMAGES
    print()
    print("  +" + "-" * w + "+")
    print("  |" + " PIPELINE 3 -- BLOCK DIAGRAM".center(w) + "|")
    print("  +" + "-" * w + "+")
    print()
    print("  +---------------------------------------+")
    print("  | 10,000 images (28x28 grayscale BMP)   |")
    print("  +------------------+--------------------+")
    print("                     |")
    print("                     v")
    print("  +---------------------------------------+")
    print(f"  | Step 1: Benchmark 5 LLMs on {BENCHMARK_SIZE:<4}      |")
    print("  |         ground-truth images (0-shot)  |")
    print("  +------------------+--------------------+")
    print("                     |")
    print("                     v")
    print("  +---------------------------------------+")
    print(f"  | Step 2: Select top 2 (diff families)  |")
    print(f"  |  A: {a_info['name']:<20}             |")
    print(f"  |  B: {b_info['name']:<20}             |")
    print("  +------------------+--------------------+")
    print("                     |")
    print("                     v")
    print("  +---------------------------------------+")
    print("  | Step 3: Both LLMs label all 10,000    |")
    print("  |         images independently          |")
    print("  +------------------+--------------------+")
    print("              +------+------+")
    print("              v             v")
    print("  +------------------+ +------------------+")
    print(f"  | AGREE  {agr:>5}     | | DISAGREE {dis:>5}   |")
    print(f"  | ({pag:>5.1f} %)       | | ({pdis:>5.1f} %)      |")
    print("  +--------+---------+ +--------+---------+")
    print("           |                    |")
    print("           v                    v")
    print("  +------------------+ +------------------+")
    print(f"  | 4a: Verify acc   | | 4b: Human review |")
    print(f"  | = {s4['agr_acc']:.4f}         | | @ 10 s/image     |")
    print(f"  | on GT overlap    | | {dis:>5} images     |")
    print("  +--------+---------+ +--------+---------+")
    print("           +------+-------------+")
    print("                  v")
    if s4["agr_acc"] < ACCURACY_TARGET:
        print("  +---------------------------------------+")
        print("  | Step 5: Tie-breaker / correction      |")
        print("  +------------------+--------------------+")
        print("                     |")
        print("                     v")
    print("  +---------------------------------------+")
    print(f"  | FINAL: {acc:.4f}  ({n_cor}/{n_tot})          |")
    print("  +---------------------------------------+")


def report(bench_res, a_info, b_info, s4, final, gt):
    print(f"\n{'='*70}\nPIPELINE 3 -- FINAL REPORT\n{'='*70}")

    labels = np.array([final.get(i, 0) for i in range(TOTAL_IMAGES)])
    acc, n_cor, n_tot = check_accuracy(labels)
    print(f"\n  FINAL ORACLE ACCURACY: {acc:.4f}  ({n_cor}/{n_tot})")

    ranked = sorted(bench_res.items(), key=lambda kv: kv[1]["accuracy"], reverse=True)

    agr  = len(s4["agreed"])
    dis  = len(s4["disagreed"])

    # --- compact summary table ---
    print("\n  +----------------------------------------------------------+")
    print("  |            PIPELINE 3 -- COMPACT SUMMARY TABLE           |")
    print("  +----------------------------------------------------------+")
    print("  | STEP 1: LLM Benchmark Accuracy (practical GT set)       |")
    print("  +---------------------+----------+--------+---------------+")
    print("  | Model               | Provider | Acc.   | Correct/Total |")
    print("  +---------------------+----------+--------+---------------+")
    for name, r in ranked:
        sel = " [*]" if name in (a_info["name"], b_info["name"]) else "    "
        print(f"  | {name:<19} | {r['provider']:<8} | {r['accuracy']:.4f} |"
              f" {r['correct']:>4}/{r['total']:<5}{sel}    |")
    print("  +---------------------+----------+--------+---------------+")
    print(f"  | STEP 2  Model A: {a_info['name']:<25}             |")
    print(f"  |         Model B: {b_info['name']:<25}             |")
    print("  +----------------------------------------------------------+")
    print(f"  | STEP 3  Full labelling: 10,000 x 2 models              |")
    print("  +----------------------------------------------------------+")
    print(f"  | STEP 4a Agreed labels:       {agr:>5}  ({100*agr/TOTAL_IMAGES:>5.1f} %)         |")
    print(f"  |         Agreed acc on GT:    {s4['agr_acc']:.4f}                    |")
    print(f"  | STEP 4b Disagreed labels:    {dis:>5}  ({100*dis/TOTAL_IMAGES:>5.1f} %)         |")
    print(f"  |         Manual time:       {s4['manual_s']:>6} s  ({s4['manual_s']/60:.1f} min)    |")
    print("  +----------------------------------------------------------+")
    print(f"  | FINAL ACCURACY:  {acc:.4f}  ({n_cor}/{n_tot})                   |")
    print(f"  | Manual images:   {dis:>5}                                |")
    print(f"  | Total manual time: {s4['manual_s']:>5} s  ({s4['manual_s']/60:.1f} min)             |")
    print("  +----------------------------------------------------------+")
    print("  | ESTIMATED API COST (10,000 imgs x 2 models)             |")
    print("  |   OpenAI  (vision): ~$15–30   (depends on model tier)   |")
    print("  |   Gemini  (flash):  ~$0.50–2  (much cheaper)            |")
    print("  |   Total estimate:   ~$15–32                             |")
    print("  +----------------------------------------------------------+")
    print("  | COMPARISON WITH PIPELINES 1 & 2                         |")
    print("  |   Pipeline 1/2: ~2.5 h human, 99.07 % (SVM+active)     |")
    print(f"  |   Pipeline 3:   {s4['manual_s']/60:>5.1f} min human, {acc:.2%} (LLM agree) |")
    print("  +----------------------------------------------------------+")

    _draw_block(s4, a_info, b_info, acc, n_cor, n_tot)

    # --- save artefacts ---
    np.save(OUTPUT_DIR / "final_labels_pipeline3.npy", labels)

    rpt = OUTPUT_DIR / "pipeline3_report.txt"
    with open(rpt, "w") as f:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"Pipeline 3 Report -- {ts}\n{'='*50}\n\n")
        f.write("Step 1 -- Benchmark\n")
        for n, r in ranked:
            sel = " [SELECTED]" if n in (a_info["name"], b_info["name"]) else ""
            f.write(f"  {n} ({r['provider']}): {r['accuracy']:.4f} "
                    f"({r['correct']}/{r['total']}){sel}\n")
        f.write(f"\nStep 2 -- Selected\n  A: {a_info['name']}\n  B: {b_info['name']}\n")
        f.write(f"\nStep 3 -- Full labelling: 10,000 images x 2\n")
        f.write(f"\nStep 4\n  Agreed: {agr} ({100*agr/TOTAL_IMAGES:.1f}%)\n"
                f"  Disagreed: {dis} ({100*dis/TOTAL_IMAGES:.1f}%)\n"
                f"  Agreed acc: {s4['agr_acc']:.4f}\n"
                f"  Manual time: {s4['manual_s']}s ({s4['manual_s']/60:.1f} min)\n")
        f.write(f"\nFinal accuracy: {acc:.4f} ({n_cor}/{n_tot})\n")

    # save benchmark CSV
    bdf = pd.DataFrame([
        {"model": n, "provider": r["provider"], "accuracy": r["accuracy"],
         "correct": r["correct"], "total": r["total"], "errors": r["errors"]}
        for n, r in ranked
    ])
    bdf.to_csv(OUTPUT_DIR / "benchmark_results.csv", index=False)

    print(f"\n  Artefacts saved in {OUTPUT_DIR}/")
    return acc, n_cor, n_tot

# ===================================================================
#  MAIN
# ===================================================================

def main():
    ensure_dirs()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Pipeline 3: LLM-Based Automatic Labelling | {ts}")
    print(f"  Images:  {IMAGE_DIR}")
    print(f"  GT file: {GROUND_TRUTH}")

    gt = load_ground_truth()
    print(f"  Ground-truth entries: {len(gt)}")

    bench_idx = get_benchmark_indices(gt)
    print(f"  Benchmark subset:    {len(bench_idx)} images (seed {BENCHMARK_SEED})\n")

    # Step 1
    bench_res = step1_benchmark(gt, bench_idx)

    # Step 2
    a_info, b_info = step2_select(bench_res)

    # Step 3
    pa, pb = step3_label(a_info, b_info)

    # Step 4
    s4 = step4_validate(pa, pb, a_info["name"], b_info["name"], gt, bench_idx)

    # Step 5
    final = step5_correct(s4, pa, pb, a_info, b_info, gt, bench_res)

    # Report
    acc, _, _ = report(bench_res, a_info, b_info, s4, final, gt)

    return acc


if __name__ == "__main__":
    main()
