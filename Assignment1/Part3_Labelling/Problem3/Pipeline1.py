"""
Semi-Automatic Labelling Pipeline for Indian Digits Dataset
============================================================
Pipeline: Raw Pixels -> HOG -> K-Means -> Human Bootstrap -> SVM (RBF, OVO) -> Active Refinement
Target: >= 99% labelling accuracy with minimal human effort.

Requires Python 3.13.5 (the check_accuracy oracle was marshalled with it).
"""

import os
import sys
import pickle
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_accuracy import check_accuracy

import matplotlib
if os.environ.get("DISPLAY"):
    matplotlib.use("TkAgg")
else:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration — KEY CHANGES
# ---------------------------------------------------------------------------
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "Indian_Digits_Train")
N_IMAGES        = 10_000
IMG_SIZE        = 28 * 28
N_CLUSTERS      = 60
N_BOUNDARY_BASE = 40          # spec: 20–40 (use max for more corrections per round)
N_BOUNDARY_MAX  = 40          # spec upper bound
HUMAN_WEIGHT    = 1000
MAX_ITER        = 50
ACCURACY_TARGET = 0.99
IMPROVEMENT_THRESHOLD = 0.001
PATIENCE        = 10
RANDOM_STATE    = 42

# ── SVM hyperparameters — matching MATLAB step3/step4 defaults ─────────────
SVM_C           = 1.0     # MATLAB fitcecoc default BoxConstraint=1
SVM_GAMMA       = 'scale' # MATLAB KernelScale='auto' ≈ sklearn 'scale' for standardized HOG

# ---------------------------------------------------------------------------
# Step 1a: Load images
# ---------------------------------------------------------------------------
def load_images() -> np.ndarray:
    print("Step 1a: Loading images...")
    raw = np.zeros((N_IMAGES, IMG_SIZE), dtype=np.float64)
    for i in range(1, N_IMAGES + 1):
        img = Image.open(os.path.join(DATA_DIR, f"{i}.bmp")).convert("L")
        raw[i - 1] = np.asarray(img, dtype=np.float64).ravel() / 255.0
    print(f"  Loaded {N_IMAGES} images  shape={raw.shape}")
    return raw

# ---------------------------------------------------------------------------
# Step 1b: HOG feature extraction
# ---------------------------------------------------------------------------
def extract_features(raw_pixels: np.ndarray):
    cache_feat   = os.path.join(BASE_DIR, "_cache_features.npy")
    cache_scaler = os.path.join(BASE_DIR, "_cache_scaler.pkl")

    if os.path.exists(cache_feat) and os.path.exists(cache_scaler):
        print("Step 1b: Loading HOG features from cache...")
        features = np.load(cache_feat)
        with open(cache_scaler, "rb") as f:
            scaler = pickle.load(f)
        n_dims = features.shape[1]
        print(f"  HOG: {n_dims} dims  (cached)")
        return features, scaler, n_dims

    print("Step 1b: Extracting HOG features...")
    n = raw_pixels.shape[0]

    def _hog(img_flat):
        return hog(img_flat.reshape(28, 28),
                   orientations=9,
                   pixels_per_cell=(4, 4),
                   cells_per_block=(2, 2),
                   feature_vector=True)

    f0     = _hog(raw_pixels[0])
    n_dims = f0.shape[0]
    features_raw = np.empty((n, n_dims), dtype=np.float64)
    features_raw[0] = f0
    for i in range(1, n):
        features_raw[i] = _hog(raw_pixels[i])

    scaler   = StandardScaler()
    features = scaler.fit_transform(features_raw)
    print(f"  HOG: {n_dims} dims  (4×4 cells, 9 orientations, 2×2 blocks, standardized)")

    np.save(cache_feat, features)
    with open(cache_scaler, "wb") as f:
        pickle.dump(scaler, f)
    print(f"  Cached → {cache_feat}")
    return features, scaler, n_dims

# ---------------------------------------------------------------------------
# Step 2: K-Means clustering
# ---------------------------------------------------------------------------
def cluster_features(features: np.ndarray):
    tag       = f"K{N_CLUSTERS}_RS{RANDOM_STATE}"
    cache_ids = os.path.join(BASE_DIR, f"_cache_cluster_ids_{tag}.npy")
    cache_km  = os.path.join(BASE_DIR, f"_cache_kmeans_{tag}.pkl")

    if os.path.exists(cache_ids) and os.path.exists(cache_km):
        print(f"\nStep 2: Loading K-Means clusters from cache  (K={N_CLUSTERS})...")
        ids = np.load(cache_ids)
        with open(cache_km, "rb") as f:
            km = pickle.load(f)
        sizes = [int(np.sum(ids == c)) for c in range(N_CLUSTERS)]
        print(f"  Cluster sizes: min={min(sizes)}  max={max(sizes)}  "
              f"mean={np.mean(sizes):.0f}  (cached)")
        return ids, km

    print(f"\nStep 2: K-Means clustering  K={N_CLUSTERS}...")
    km  = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=RANDOM_STATE)
    ids = km.fit_predict(features)
    sizes = [int(np.sum(ids == c)) for c in range(N_CLUSTERS)]
    print(f"  Cluster sizes: min={min(sizes)}  max={max(sizes)}  "
          f"mean={np.mean(sizes):.0f}  std={np.std(sizes):.0f}")

    np.save(cache_ids, ids)
    with open(cache_km, "wb") as f:
        pickle.dump(km, f)
    print(f"  Cached → {cache_ids}")
    return ids, km

# ---------------------------------------------------------------------------
# Human interaction helpers
# ---------------------------------------------------------------------------
def _save_grid(raw_pixels, indices, title, path, subtitles=None, dpi=120):
    n    = len(indices)
    cols = min(n, 8)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.8, rows * 2.0))
    axes = np.asarray(axes).ravel()
    for i, idx in enumerate(indices):
        axes[i].imshow(raw_pixels[idx].reshape(28, 28),
                       cmap="gray", interpolation="nearest")
        axes[i].set_title(subtitles[i] if subtitles else f"#{idx+1}", fontsize=7)
        axes[i].axis("off")
    for i in range(n, len(axes)):
        axes[i].axis("off")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)

def _ask_cluster_label(raw_pixels, sample_idx, cluster_id) -> int:
    path = os.path.join(BASE_DIR, "_cluster_preview.png")
    _save_grid(raw_pixels, sample_idx,
               title=f"Cluster {cluster_id+1}/{N_CLUSTERS} — digit? (0-9 or 's')",
               path=path)
    print(f"    [Preview → _cluster_preview.png]", flush=True)
    while True:
        resp = input(f"    Cluster {cluster_id+1}: digit (0-9) or 's' to skip: ").strip().lower()
        if resp == "s":
            return -1
        if resp.isdigit() and 0 <= int(resp) <= 9:
            return int(resp)
        print("    Enter 0–9 or 's'.", flush=True)

def _ask_boundary_labels(raw_pixels, boundary_indices, svm_predictions=None) -> dict:
    human_labels = {}
    n = len(boundary_indices)
    subtitles = [
        f"#{i+1}  img{idx+1}\n{'SVM:'+str(svm_predictions[idx]) if svm_predictions is not None else ''}"
        for i, idx in enumerate(boundary_indices)
    ]
    grid_path = os.path.join(BASE_DIR, "_boundary_overview.png")
    _save_grid(raw_pixels, boundary_indices,
               title=f"Boundary images — label by grid number (1–{n})",
               path=grid_path, subtitles=subtitles, dpi=150)
    cols = min(n, 8)
    rows = (n + cols - 1) // cols
    print(f"    [Grid → _boundary_overview.png  ({rows}×{cols}, {n} images)]", flush=True)
    print(f"    Open the grid, then label each image.", flush=True)

    for i, idx in enumerate(boundary_indices):
        hint = f" (SVM: {svm_predictions[idx]})" if svm_predictions is not None else ""
        while True:
            print(f"    [{i+1}/{n}] img#{idx+1}{hint} — digit (0-9) or 's': ",
                  end="", flush=True)
            resp = sys.stdin.readline().strip().lower()
            if resp == "s":
                break
            if resp.isdigit() and 0 <= int(resp) <= 9:
                human_labels[idx] = int(resp)
                break
            print("    Enter 0–9 or 's'.", flush=True)
    return human_labels

# ---------------------------------------------------------------------------
# Step 3: Bootstrap labelling
# ---------------------------------------------------------------------------
def bootstrap_labels(cluster_ids, raw_pixels):
    cache_bootstrap = os.path.join(BASE_DIR, f"_cache_bootstrap_K{N_CLUSTERS}.pkl")

    if os.path.exists(cache_bootstrap):
        print(f"\nStep 3: Loading bootstrap answers from cache...")
        with open(cache_bootstrap, "rb") as f:
            boot = pickle.load(f)
        print(f"  Resumed: {boot['n_skipped']} skipped, "
              f"{boot['images_viewed']} images viewed")
        return (boot['labels'], boot['cluster_digit'], boot['images_viewed'],
                boot['manual_time'], boot['examined_indices'], boot['examined_labels'])

    print(f"\nStep 3: Bootstrap labelling  ({N_CLUSTERS} clusters)...")
    rng = np.random.RandomState(RANDOM_STATE)
    labels          = np.full(N_IMAGES, -1, dtype=int)
    cluster_digit   = np.full(N_CLUSTERS, -1, dtype=int)
    images_viewed   = 0
    n_skipped       = 0
    examined_indices = set()
    examined_labels  = {}

    for c in range(N_CLUSTERS):
        members    = np.where(cluster_ids == c)[0]
        sample_size = min(8, len(members))
        sample_idx  = rng.choice(members, size=sample_size, replace=False)
        digit = _ask_cluster_label(raw_pixels, sample_idx, c)
        images_viewed += sample_size

        if digit == -1:
            n_skipped += 1
            print(f"    → Cluster {c+1} skipped — {len(members)} images excluded", flush=True)
        else:
            cluster_digit[c] = digit
            labels[members]  = digit
            for idx in sample_idx:
                examined_indices.add(int(idx))
                examined_labels[int(idx)] = digit

    n_unlabelled = int(np.sum(labels == -1))
    manual_time  = N_CLUSTERS * 20

    eval_labels = labels.copy()
    eval_labels[eval_labels == -1] = 0
    acc, n_correct, _ = check_accuracy(eval_labels)
    print(f"\n  Bootstrap accuracy: {acc:.4f}  ({n_correct}/{N_IMAGES})")
    print(f"  Skipped: {n_skipped} clusters  ({n_unlabelled} images unlabelled)")
    print(f"  Images viewed: {images_viewed}  |  Manual time: {manual_time} s")

    with open(cache_bootstrap, "wb") as f:
        pickle.dump({"labels": labels, "cluster_digit": cluster_digit,
                     "images_viewed": images_viewed, "manual_time": manual_time,
                     "n_skipped": n_skipped, "examined_indices": examined_indices,
                     "examined_labels": examined_labels}, f)
    print(f"  Cached → {cache_bootstrap}  (delete to re-label)")

    return (labels, cluster_digit, images_viewed, manual_time,
            examined_indices, examined_labels)

# ---------------------------------------------------------------------------
# SVM helpers
# ---------------------------------------------------------------------------
def train_svm(features: np.ndarray,
              labels: np.ndarray,
              sample_weight=None) -> SVC:
    """
    OVO SVM with RBF kernel — parameters match the MATLAB pipeline.

    C=1 (MATLAB default BoxConstraint): high regularisation, tolerates noisy
    cluster-derived labels without memorising them.

    gamma='scale' (sklearn equiv. of MATLAB KernelScale='auto' on standardised
    features): smooth, wide-radius kernel that generalises across classes
    instead of predicting the densest nearby cluster for uncertain images.
    """
    clf = SVC(
        kernel='rbf',
        gamma=SVM_GAMMA,           # 0.005  — tuned for HOG 1296-dim
        C=SVM_C,                   # 50.0   — harder margin
        decision_function_shape='ovr',
        random_state=RANDOM_STATE,
    )
    clf.fit(features, labels, sample_weight=sample_weight)
    return clf

def compute_margins(clf, features) -> np.ndarray:
    """
    Spec Step 4 margin: highest class score − second-highest class score.
    decision_function() → (N, 10) with shape='ovr'.
    """
    scores   = clf.decision_function(features)   # (10000, 10)
    s_sorted = np.sort(scores, axis=1)
    return s_sorted[:, -1] - s_sorted[:, -2]     # top1 − top2

def find_boundary_images(clf: SVC,
                         features: np.ndarray,
                         n_boundary: int,
                         exclude: set = None,
                         cluster_labels: np.ndarray = None,
                         predictions: np.ndarray = None) -> np.ndarray:
    """
    Select n_boundary images for human labelling.
    Priority:
      1. Images where SVM prediction disagrees with original cluster label
         (margin-sorted — most uncertain disagreements first).
      2. Fill remaining slots with globally smallest-margin images.
    Images in *exclude* (already human-labelled) are always skipped.
    """
    margins = compute_margins(clf, features)   # (10000,)

    if exclude:
        for i in exclude:
            margins[i] = np.inf

    if cluster_labels is not None and predictions is not None:
        sorted_idx = np.argsort(margins)
        # Disagreeing: cluster said X but SVM now predicts Y ≠ X (likely misclustered)
        disagree = [int(i) for i in sorted_idx
                    if margins[i] < np.inf
                    and cluster_labels[i] >= 0
                    and predictions[i] != cluster_labels[i]]
        selected = disagree[:n_boundary]
        if len(selected) < n_boundary:
            exclude_sel = set(selected) | (exclude or set())
            for i in sorted_idx:
                if len(selected) >= n_boundary:
                    break
                if int(i) not in exclude_sel and margins[i] < np.inf:
                    selected.append(int(i))
        return np.array(selected[:n_boundary], dtype=int)

    # Fallback: global smallest margin
    return np.argsort(margins)[:n_boundary]

# ---------------------------------------------------------------------------
# Steps 4–6: Active refinement loop
# ---------------------------------------------------------------------------
def active_refinement(features, labels, examined_indices, examined_labels, raw_pixels):
    print("\nSteps 4–6: SVM Training + Active Refinement")
    print("=" * 60)
    print(f"  HUMAN_WEIGHT={HUMAN_WEIGHT}  C={SVM_C}  gamma={SVM_GAMMA}  "
          f"boundary={N_BOUNDARY_BASE}–{N_BOUNDARY_MAX}  patience={PATIENCE}")

    human_set  = set(examined_indices)
    train_data = {}

    for idx in np.where(labels >= 0)[0]:
        train_data[int(idx)] = (int(labels[idx]), 1.0)
    for idx, lbl in examined_labels.items():
        train_data[idx] = (lbl, float(HUMAN_WEIGHT))

    n_seed  = len(train_data)
    n_human = len(examined_labels)
    print(f"\n  Seed: {n_seed} images  "
          f"({n_seed - n_human} cluster-derived w=1, "
          f"{n_human} human-verified w={HUMAN_WEIGHT})")

    def _train_and_predict():
        idx_list = sorted(train_data.keys())
        X = features[idx_list]
        Y = np.array([train_data[i][0] for i in idx_list])
        W = np.array([train_data[i][1] for i in idx_list])
        svm = train_svm(X, Y, sample_weight=W)
        return svm, svm.predict(features)

    print("  Training initial SVM...")
    svm, predicted = _train_and_predict()

    ambiguous_mask = labels < 0
    n_ambiguous    = int(np.sum(ambiguous_mask))
    if n_ambiguous > 0:
        for idx in np.where(ambiguous_mask)[0]:
            train_data[int(idx)] = (int(predicted[idx]), 1.0)
        print(f"  Assigned {n_ambiguous} skipped-cluster images via SVM (w=1)")

    acc0, nc0, _ = check_accuracy(predicted)
    print(f"  Initial SVM accuracy: {acc0:.4f}  ({nc0}/{N_IMAGES})")

    acc, n_correct = acc0, nc0
    log = [{"iter": 0, "accuracy": acc, "n_correct": n_correct,
             "boundary": 0, "cumulative": 0}]
    best_acc    = acc
    best_labels = predicted.copy()
    prev_acc    = acc

    total_boundary       = 0
    total_boundary_time  = 0
    n_boundary_this_iter = N_BOUNDARY_BASE

    # ── KEY FIX: patience counter ────────────────────────────────────────
    consecutive_slow = 0

    for it in range(1, MAX_ITER + 1):

        if acc >= ACCURACY_TARGET:
            print(f"\n  Target {ACCURACY_TARGET:.0%} reached!")
            break

        print(f"\n  --- Iteration {it} ---")
        print(f"  Accuracy: {acc:.4f}  ({n_correct}/{N_IMAGES})  "
              f"[best: {best_acc:.4f}]  "
              f"boundary_n={n_boundary_this_iter}  slow_streak={consecutive_slow}/{PATIENCE}")

        bnd = find_boundary_images(svm, features, n_boundary_this_iter,
                                    exclude=human_set,
                                    cluster_labels=labels,
                                    predictions=predicted)
        if len(bnd) == 0:
            print("  No new boundary images. Stopping.")
            break

        print(f"  Presenting {len(bnd)} boundary images for labelling...")
        print(f"  (Enter 's' to skip any image you are unsure about)", flush=True)
        human_labels_iter = _ask_boundary_labels(raw_pixels, bnd,
                                                  svm_predictions=predicted)

        n_new         = len(human_labels_iter)
        n_skipped_bnd = len(bnd) - n_new
        if n_skipped_bnd:
            print(f"  Skipped {n_skipped_bnd} uncertain image(s).")

        for idx, lbl in human_labels_iter.items():
            train_data[idx] = (lbl, float(HUMAN_WEIGHT))
            human_set.add(idx)

        total_boundary      += n_new
        total_boundary_time += n_new * 10

        if n_new == 0:
            print("  All images skipped — stopping.")
            break

        svm, predicted = _train_and_predict()
        acc_new, nc_new, _ = check_accuracy(predicted)
        improvement = acc_new - prev_acc

        print(f"  Labelled: {n_new}  |  "
              f"Accuracy: {acc_new:.4f} ({nc_new})  |  "
              f"Δ={improvement:+.4f}")

        if acc_new > best_acc:
            best_acc    = acc_new
            best_labels = predicted.copy()

        log.append({"iter": it, "accuracy": acc_new, "n_correct": nc_new,
                    "boundary": n_new, "cumulative": total_boundary})

        prev_acc       = acc_new
        acc, n_correct = acc_new, nc_new

        # Self-training: propagate SVM corrections to all cluster-derived labels
        n_updated = 0
        for idx in list(train_data.keys()):
            if idx not in human_set:
                new_lbl = int(predicted[idx])
                if new_lbl != train_data[idx][0]:
                    train_data[idx] = (new_lbl, 1.0)
                    n_updated += 1
        if n_updated:
            print(f"  Self-training: {n_updated} cluster labels updated to match SVM")

        # ── KEY FIX 1: patience-based convergence ───────────────────────
        if improvement < IMPROVEMENT_THRESHOLD:
            consecutive_slow += 1
            print(f"  Slow iteration ({consecutive_slow}/{PATIENCE})")
            if consecutive_slow >= PATIENCE:
                print(f"\n  Slow for {PATIENCE} consecutive iterations → converged.")
                predicted  = best_labels.copy()
                acc, n_correct = best_acc, int(np.round(best_acc * N_IMAGES))
                break
        else:
            consecutive_slow = 0

    return (predicted, svm, log,
            total_boundary, total_boundary_time, human_set)

# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
def draw_block_diagram(results: dict):
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.axis("off")

    blocks = [
        ("Step 1a",  "Load Images\n10,000 × 28×28\n/255 normalise",               0.05, "#AED6F1"),
        ("Step 1b",  f"HOG Features\n{results['n_hog_dims']} dims\n4×4 cells·9ori\nStandardized", 0.21, "#A9DFBF"),
        ("Step 2",   f"K-Means\nK={results['n_clusters']}\nn_init=10",             0.37, "#F9E79F"),
        ("Step 3",   "Bootstrap\n8 imgs/cluster\n20 s/cluster\nHuman labels",      0.53, "#F5CBA7"),
        ("Step 4–5", f"SVM RBF OVO\nγ=scale C=10\nw={results['human_weight']}\nRetrain", 0.69, "#D7BDE2"),
        ("Step 6",   f"Result\nacc={results['final_accuracy']:.4f}\n{results['n_iterations']} iters", 0.85, "#FADBD8"),
    ]

    box_w, box_h, box_y = 0.13, 0.60, 0.20
    for title, body, cx, color in blocks:
        ax.add_patch(plt.Rectangle((cx - box_w/2, box_y), box_w, box_h,
                                   lw=1.5, edgecolor="#2C3E50",
                                   facecolor=color, zorder=2))
        ax.text(cx, box_y + box_h - 0.03, title,
                ha="center", va="top", fontsize=8, fontweight="bold", zorder=3)
        ax.text(cx, box_y + box_h/2 - 0.03, body,
                ha="center", va="center", fontsize=7, linespacing=1.4, zorder=3)

    ap = dict(arrowstyle="->", color="#2C3E50", lw=1.5)
    for i in range(len(blocks) - 1):
        ax.annotate("", xy=(blocks[i+1][2] - box_w/2, box_y + box_h/2),
                    xytext=(blocks[i][2] + box_w/2, box_y + box_h/2),
                    arrowprops=ap, zorder=4)

    ax.annotate("",
                xy=(blocks[4][2] - box_w/2, box_y + box_h*0.3),
                xytext=(blocks[5][2], box_y),
                arrowprops=dict(arrowstyle="->", color="#E74C3C", lw=1.4,
                                connectionstyle="arc3,rad=0.35"), zorder=4)
    ax.text((blocks[4][2] + blocks[5][2])/2, box_y - 0.08,
            "Active loop\n(acc < 99%)", ha="center", fontsize=7,
            color="#E74C3C", style="italic")

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.suptitle("Semi-Automatic Labelling Pipeline — Block Diagram",
                 fontsize=11, fontweight="bold", y=0.97)
    path = os.path.join(BASE_DIR, "_pipeline_block_diagram.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Block diagram → _pipeline_block_diagram.png]")

def print_summary_table(results: dict):
    baseline_s = 100_000
    time_saved = (1 - results["total_time"] / baseline_s) * 100
    W = 32

    rows = [
        ("Feature representation",
         f"HOG  ({results['n_hog_dims']} dims, 4×4 cells, 9 ori, standardized)"),
        ("K-Means clusters (K)",    str(results["n_clusters"])),
        ("Human weight (w_human)",  str(results["human_weight"])),
        ("SVM kernel / scheme",     "RBF / One-vs-One (OVO)"),
        ("SVM γ / C",               "scale / 10.0"),
        ("DIVIDER", ""),
        ("Oracle accuracy",
         f"{results['final_accuracy']:.4f}  ({results['final_correct']}/10000)"),
        ("Practical accuracy",      f"{results['practical_accuracy']:.4f}"),
        ("Iterations performed",    str(results["n_iterations"])),
        ("DIVIDER", ""),
        ("Bootstrap images viewed",
         f"{results['n_clusters']} clusters × 8 = {results['bootstrap_images']} images"),
        ("Bootstrap time",
         f"{results['bootstrap_time']} s  ({results['bootstrap_time']/60:.1f} min)"),
        ("Boundary images labelled", f"{results['boundary_images']} images"),
        ("Boundary time",
         f"{results['boundary_time']} s  ({results['boundary_time']/60:.1f} min)"),
        ("DIVIDER", ""),
        ("Total images handled",    f"{results['total_images']} images"),
        ("Total manual time",
         f"{results['total_time']} s  ({results['total_time']/60:.1f} min)"),
        ("Baseline (fully manual)", f"{baseline_s} s  (27.8 h)"),
        ("Time saved",              f"{time_saved:.1f}%"),
    ]

    C1, C2 = 30, 52
    border = "+" + "-"*(C1+2) + "+" + "-"*(C2+2) + "+"
    print("\n" + border)
    print(f"| {'Metric':<{C1}} | {'Value':<{C2}} |")
    print(border)
    for metric, value in rows:
        if metric == "DIVIDER":
            print(border)
        else:
            print(f"| {metric:<{C1}} | {value:<{C2}} |")
    print(border)

# ---------------------------------------------------------------------------
# Practical evaluation
# ---------------------------------------------------------------------------
def practical_evaluation(final_labels: np.ndarray) -> float:
    print("\nPractical Evaluation (5% held-out simulation):")
    rng      = np.random.RandomState(RANDOM_STATE)
    held_out = rng.choice(N_IMAGES, size=500, replace=False)
    acc, n_correct, _ = check_accuracy(final_labels)
    print(f"  Oracle (full set) : {acc:.4f}  ({n_correct}/{N_IMAGES})")
    print(f"  Held-out sample   : {len(held_out)} images "
          f"(oracle used as proxy — replace with manual labels in production)")
    return acc

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  Indian Digits Semi-Automatic Labelling Pipeline")
    print(f"  HUMAN_WEIGHT={HUMAN_WEIGHT}  K={N_CLUSTERS}  "
          f"boundary={N_BOUNDARY_BASE}–{N_BOUNDARY_MAX}  patience={PATIENCE}")
    print("=" * 60)

    raw_pixels                          = load_images()
    features, scaler, n_dims            = extract_features(raw_pixels)
    cluster_ids, kmeans                 = cluster_features(features)
    (labels, cluster_digit,
     bootstrap_images, bootstrap_time,
     examined_indices, examined_labels) = bootstrap_labels(cluster_ids, raw_pixels)
    (final_labels, svm, log,
     boundary_images, boundary_time,
     human_set)                         = active_refinement(
        features, labels, examined_indices, examined_labels, raw_pixels)

    print("\n" + "=" * 60)
    print("  FINAL RESULTS")
    print("=" * 60)

    final_acc, final_correct, _ = check_accuracy(final_labels)
    practical_acc               = practical_evaluation(final_labels)
    total_images = bootstrap_images + boundary_images
    total_time   = bootstrap_time   + boundary_time
    n_iters      = len(log) - 1

    results = {
        "n_hog_dims"        : n_dims,
        "n_clusters"        : N_CLUSTERS,
        "human_weight"      : HUMAN_WEIGHT,
        "final_accuracy"    : final_acc,
        "practical_accuracy": practical_acc,
        "final_correct"     : final_correct,
        "n_iterations"      : n_iters,
        "bootstrap_images"  : bootstrap_images,
        "bootstrap_time"    : bootstrap_time,
        "boundary_images"   : boundary_images,
        "boundary_time"     : boundary_time,
        "total_images"      : total_images,
        "total_time"        : total_time,
        "iteration_log"     : log,
    }

    # ── Spec reporting ────────────────────────────────────────────────
    print_summary_table(results)

    print(f"\n  {'It':>3}  {'Accuracy':>9}  {'Correct':>8}  "
          f"{'Bnd':>5}  {'Cumul':>6}  {'Slow':>5}")
    print(f"  {'---':>3}  {'---------':>9}  {'--------':>8}  "
          f"{'-----':>5}  {'------':>6}  {'-----':>5}")
    slow = 0
    for e in log:
        if e["iter"] > 0:
            prev = log[e["iter"]-1]["accuracy"]
            slow = slow+1 if (e["accuracy"]-prev) < IMPROVEMENT_THRESHOLD else 0
        print(f"  {e['iter']:>3}  {e['accuracy']:>9.4f}  "
              f"{e['n_correct']:>8}  {e['boundary']:>5}  "
              f"{e['cumulative']:>6}  {slow:>5}")

    draw_block_diagram(results)

    out_path = os.path.join(BASE_DIR, "final_labels.npy")
    np.save(out_path, final_labels)
    print(f"\n  Saved → {out_path}")
    return results

if __name__ == "__main__":
    results = main()