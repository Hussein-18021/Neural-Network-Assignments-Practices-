"""
Part 2: Classical ML Modelling on ReducedMNIST
===============================================
Dataset  : ReducedMNIST (built from sklearn digits, upsampled to 28×28)
           Train: 1 000 examples/digit  → 10 000 total
           Test :   200 examples/digit  →  2 000 total

Features : DCT  - top-left 15×15 block   (225 coefficients)
           PCA  - min components for ≥95% variance
           HOG  - Histogram of Oriented Gradients (1 296 dims)

Classifiers:
           K-Means nearest-centroid  k ∈ {1, 4, 16, 32}
           SVM  Linear  (C=1)
           SVM  RBF     (C=10, gamma='scale')

Outputs  : results_table.png   - accuracy/time summary table
           kmeans_accuracy.png - accuracy vs k curves
           comparison_bar.png  - grouped bar chart
           confusion_matrices.png - best per classifier
           sample_images.png   - one sample per digit
           results.json        - numeric results
"""

# ── Imports ────────────────────────────────────────────────────────────────────
import json, time, warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.fft          import dctn
from scipy.ndimage      import zoom, rotate, shift
from sklearn.cluster    import KMeans
from sklearn.datasets   import load_digits
from sklearn.decomposition import PCA
from sklearn.metrics    import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.svm        import SVC
from skimage.feature    import hog

# ── Config ─────────────────────────────────────────────────────────────────────
SEED            = 42
TRAIN_PER_CLASS = 1_000
TEST_PER_CLASS  = 200
N_CLASSES       = 10
K_VALUES        = [1, 4, 16, 32]
OUT             = '.'
np.random.seed(SEED)

# ══════════════════════════════════════════════════════════════════════════════
# 1.  DATASET
# ══════════════════════════════════════════════════════════════════════════════

def build_dataset():
    """
    Build ReducedMNIST from sklearn's built-in 8×8 digits.
    Each digit is upsampled (bicubic) to 28×28 and augmented
    (rotation ±15°, shift ±2 px, Gaussian noise σ=3) to fill
    the quota of 1 200 samples per class before train/test split.
    """
    total = TRAIN_PER_CLASS + TEST_PER_CLASS          # 1 200 per digit

    raw = load_digits()
    X_raw, y_raw = raw.data, raw.target               # (1797, 64)

    X_buf, y_buf = [], []
    for digit in range(N_CLASSES):
        idx  = np.where(y_raw == digit)[0]
        imgs = []
        for i in idx:                                  # upsample originals
            img = zoom(X_raw[i].reshape(8, 8), 28/8, order=3)
            lo, hi = img.min(), img.max()
            img = (img - lo) / (hi - lo + 1e-8) * 255.0
            imgs.append(img)
            X_buf.append(img.flatten())
            y_buf.append(digit)

        while len([y for y in y_buf if y == digit]) < total:
            src = imgs[np.random.randint(len(imgs))].copy()
            src = rotate(src, np.random.uniform(-15, 15),
                         reshape=False, mode='nearest')
            src = shift(src, np.random.uniform(-2, 2, 2), mode='nearest')
            src = np.clip(src + np.random.normal(0, 3, src.shape), 0, 255)
            X_buf.append(src.flatten())
            y_buf.append(digit)

    X_full = np.array(X_buf, dtype=np.float32)
    y_full = np.array(y_buf, dtype=np.int32)

    X_tr, y_tr, X_te, y_te = [], [], [], []
    for digit in range(N_CLASSES):
        idx = np.where(y_full == digit)[0]
        np.random.shuffle(idx)
        X_tr.append(X_full[idx[:TRAIN_PER_CLASS]])
        y_tr.append(y_full[idx[:TRAIN_PER_CLASS]])
        X_te.append(X_full[idx[TRAIN_PER_CLASS:total]])
        y_te.append(y_full[idx[TRAIN_PER_CLASS:total]])

    return (np.vstack(X_tr), np.hstack(y_tr),
            np.vstack(X_te), np.hstack(y_te))


# ══════════════════════════════════════════════════════════════════════════════
# 2.  FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def dct_features(X, side=15):
    """2-D DCT-II, retain top-left 15×15 = 225 low-frequency coefficients."""
    out = []
    for row in X:
        C = dctn(row.reshape(28, 28), norm='ortho')
        out.append(C[:side, :side].flatten())
    return np.array(out)


def pca_features(X_tr, X_te, threshold=0.95):
    """Fit PCA on training data; project both splits. No test leakage."""
    pca = PCA(n_components=threshold, svd_solver='full', random_state=SEED)
    return pca.fit_transform(X_tr), pca.transform(X_te), int(pca.n_components_)


def hog_features(X):
    """HOG: 9 orientations, 4×4 cells, 2×2 blocks → 1 296 dims."""
    out = []
    for row in X:
        fd = hog(row.reshape(28, 28) / 255.0,
                 orientations=9, pixels_per_cell=(4, 4),
                 cells_per_block=(2, 2), block_norm='L2-Hys',
                 feature_vector=True)
        out.append(fd)
    return np.array(out)


def extract_all(X_tr, X_te):
    feats = {}
    t = time.time()
    feats['DCT'] = (dct_features(X_tr), dct_features(X_te))
    print(f"  DCT  → {feats['DCT'][0].shape[1]:>4} dims   ({time.time()-t:.1f}s)")

    t = time.time()
    tr_pca, te_pca, n = pca_features(X_tr, X_te)
    feats['PCA'] = (tr_pca, te_pca)
    print(f"  PCA  → {n:>4} dims   ({time.time()-t:.1f}s)  [≥95% var]")

    t = time.time()
    feats['HOG'] = (hog_features(X_tr), hog_features(X_te))
    print(f"  HOG  → {feats['HOG'][0].shape[1]:>4} dims   ({time.time()-t:.1f}s)")
    return feats


# ══════════════════════════════════════════════════════════════════════════════
# 3.  K-MEANS NEAREST-CENTROID CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════

def kmeans_classify(feats, y_tr, y_te):
    results = {f: {} for f in feats}
    for feat, (X_tr, X_te) in feats.items():
        for k in K_VALUES:
            t0 = time.time()
            # Fit k centroids per class
            centroids, labels = [], []
            for digit in range(N_CLASSES):
                km = KMeans(n_clusters=k, n_init=5, random_state=SEED)
                km.fit(X_tr[y_tr == digit])
                centroids.append(km.cluster_centers_)
                labels.extend([digit] * k)
            C = np.vstack(centroids)          # (10k, D)
            L = np.array(labels)

            # Predict: nearest centroid
            preds = np.array([L[((C - x)**2).sum(1).argmin()] for x in X_te])
            acc = accuracy_score(y_te, preds)
            results[feat][k] = {'acc': acc, 'time': time.time()-t0, 'preds': preds}
            print(f"  K-Means [{feat}] k={k:>2d}  acc={acc:.4f}  t={results[feat][k]['time']:.1f}s")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 4.  SVM CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════

SVM_CFGS = {
    'Linear (C=1)':          SVC(kernel='linear', C=1, random_state=SEED),
    'RBF (C=10,γ=scale)':    SVC(kernel='rbf', C=10, gamma='scale', random_state=SEED),
}

def svm_classify(feats, y_tr, y_te):
    results = {f: {} for f in feats}
    for feat, (X_tr, X_te) in feats.items():
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(X_tr)
        Xte_s = scaler.transform(X_te)
        for name, clf in SVM_CFGS.items():
            t0 = time.time()
            clf.fit(Xtr_s, y_tr)
            preds = clf.predict(Xte_s)
            acc = accuracy_score(y_te, preds)
            results[feat][name] = {'acc': acc, 'time': time.time()-t0, 'preds': preds}
            print(f"  SVM   [{feat}] {name:<24s}  acc={acc:.4f}  t={results[feat][name]['time']:.1f}s")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 5.  FIGURES
# ══════════════════════════════════════════════════════════════════════════════

FEAT_COLS = {'DCT': '#e74c3c', 'PCA': '#3498db', 'HOG': '#2ecc71'}
FEATS     = ['DCT', 'PCA', 'HOG']

def fig_table(km, svm):
    """Fig 1 - accuracy / time summary table (matches assignment table format)."""
    rows, clf_names = [], []
    for k in K_VALUES:
        clf_names.append(f'K-Means k={k}')
        rows.append([f"{km[f][k]['acc']:.4f}\n{km[f][k]['time']:.1f}s" for f in FEATS])
    for name in SVM_CFGS:
        clf_names.append(f'SVM {name}')
        rows.append([f"{svm[f][name]['acc']:.4f}\n{svm[f][name]['time']:.1f}s" for f in FEATS])

    col_labels = ['Classifier'] + [f'{f}\nAcc / Time' for f in FEATS]
    cell_text  = [[clf_names[i]] + rows[i] for i in range(len(rows))]

    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.axis('off')
    tbl = ax.table(cellText=cell_text, colLabels=col_labels,
                   loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 2.2)
    for j in range(len(col_labels)):
        tbl[0,j].set_facecolor('#2c5f8a'); tbl[0,j].set_text_props(color='white', fontweight='bold')
    for i in range(1, len(rows)+1):
        shade = '#f0f4f8' if i % 2 == 0 else 'white'
        for j in range(len(col_labels)):
            tbl[i,j].set_facecolor(shade)
    ax.set_title('Results Summary - ReducedMNIST (Accuracy / Time)',
                 fontsize=12, fontweight='bold', pad=14)
    plt.tight_layout()
    plt.savefig(f'{OUT}/results_table.png', dpi=150, bbox_inches='tight')
    plt.close()


def fig_kmeans_curve(km):
    """Fig 2 - K-Means accuracy vs k."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for f in FEATS:
        ax.plot(K_VALUES, [km[f][k]['acc'] for k in K_VALUES],
                'o-', color=FEAT_COLS[f], lw=2.5, ms=8, label=f)
    ax.set_xlabel('Clusters per class (k)', fontsize=11)
    ax.set_ylabel('Test Accuracy', fontsize=11)
    ax.set_title('K-Means Accuracy vs. k', fontsize=12, fontweight='bold')
    ax.set_xticks(K_VALUES); ax.legend(); ax.grid(alpha=0.35)
    plt.tight_layout()
    plt.savefig(f'{OUT}/kmeans_accuracy.png', dpi=150, bbox_inches='tight')
    plt.close()


def fig_bar(km, svm):
    """Fig 3 - grouped bar chart for all classifiers × features."""
    labels = [f'K={k}' for k in K_VALUES] + ['SVM-Lin', 'SVM-RBF']
    x, w  = np.arange(len(labels)), 0.25
    fig, ax = plt.subplots(figsize=(13, 5))
    for i, f in enumerate(FEATS):
        accs = ([km[f][k]['acc'] for k in K_VALUES] +
                [svm[f][n]['acc'] for n in SVM_CFGS])
        ax.bar(x + (i-1)*w, accs, w, label=f, color=FEAT_COLS[f], alpha=0.85)
    ax.set_xlabel('Classifier', fontsize=11); ax.set_ylabel('Test Accuracy', fontsize=11)
    ax.set_title('Accuracy - All Classifiers & Features', fontsize=12, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0.70, 1.02)
    ax.legend(); ax.grid(axis='y', alpha=0.35)
    plt.tight_layout()
    plt.savefig(f'{OUT}/comparison_bar.png', dpi=150, bbox_inches='tight')
    plt.close()


def fig_confusion(km, svm, y_te):
    """Fig 4 - confusion matrices for the best result of each classifier type."""
    # Best K-Means: k=32, HOG (empirically highest)
    # Best SVM:     RBF,  HOG
    pairs = [
        (km['HOG'][32]['preds'],  km['HOG'][32]['acc'],  'K-Means k=32 (HOG)'),
        (svm['HOG']['RBF (C=10,γ=scale)']['preds'],
         svm['HOG']['RBF (C=10,γ=scale)']['acc'], 'SVM RBF C=10 (HOG)'),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, (preds, acc, title) in zip(axes, pairs):
        cm   = confusion_matrix(y_te, preds)
        disp = ConfusionMatrixDisplay(cm, display_labels=list(range(N_CLASSES)))
        disp.plot(ax=ax, colorbar=True, cmap='Blues')
        ax.set_title(f'{title}\nAccuracy = {acc:.4f}', fontsize=11, fontweight='bold')
    plt.suptitle('Confusion Matrices - Best Result per Classifier Type',
                 fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f'{OUT}/confusion_matrices.png', dpi=150, bbox_inches='tight')
    plt.close()


def fig_samples(X_tr, y_tr, X_te, y_te):
    """Fig 5 - one sample per digit (train + test row)."""
    fig, axes = plt.subplots(2, N_CLASSES, figsize=(15, 3.5))
    for d in range(N_CLASSES):
        axes[0, d].imshow(X_tr[np.where(y_tr==d)[0][0]].reshape(28,28), cmap='gray')
        axes[0, d].axis('off'); axes[0, d].set_title(str(d), fontsize=9)
        axes[1, d].imshow(X_te[np.where(y_te==d)[0][0]].reshape(28,28), cmap='gray')
        axes[1, d].axis('off')
    axes[0,0].set_ylabel('Train', fontsize=9); axes[1,0].set_ylabel('Test', fontsize=9)
    plt.suptitle('Sample Images - ReducedMNIST', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUT}/sample_images.png', dpi=150, bbox_inches='tight')
    plt.close()


def save_json(km, svm):
    payload = {
        'kmeans': {f: {str(k): {'acc': float(km[f][k]['acc']),
                                 'time': float(km[f][k]['time'])}
                       for k in K_VALUES} for f in FEATS},
        'svm':    {f: {n: {'acc': float(svm[f][n]['acc']),
                            'time': float(svm[f][n]['time'])}
                       for n in SVM_CFGS} for f in FEATS},
    }
    with open(f'{OUT}/results.json', 'w') as fp:
        json.dump(payload, fp, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=" * 60)
    print("Step 1 - Build ReducedMNIST")
    print("=" * 60)
    X_tr, y_tr, X_te, y_te = build_dataset()
    print(f"  Train: {X_tr.shape}  Test: {X_te.shape}")

    print("\nStep 2 - Feature Extraction")
    print("=" * 60)
    feats = extract_all(X_tr, X_te)

    print("\nStep 3 - K-Means Classifier")
    print("=" * 60)
    km_res = kmeans_classify(feats, y_tr, y_te)

    print("\nStep 4 - SVM Classifier")
    print("=" * 60)
    svm_res = svm_classify(feats, y_tr, y_te)

    print("\nStep 5 - Generating Figures")
    print("=" * 60)
    fig_table(km_res, svm_res);      print("  results_table.png")
    fig_kmeans_curve(km_res);        print("  kmeans_accuracy.png")
    fig_bar(km_res, svm_res);        print("  comparison_bar.png")
    fig_confusion(km_res, svm_res, y_te); print("  confusion_matrices.png")
    fig_samples(X_tr, y_tr, X_te, y_te); print("  sample_images.png")
    save_json(km_res, svm_res);      print("  results.json")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)