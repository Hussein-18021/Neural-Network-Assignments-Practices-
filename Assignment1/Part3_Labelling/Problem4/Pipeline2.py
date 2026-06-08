import os
import csv
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from skimage.feature import hog
import check_accuracy
import logging
from datetime import datetime
import time

# ==========================================
# Configuration
# ==========================================
DATASET_PATH             = "./dataset"
NUM_IMAGES               = 10000
SEED_SIZE                = 300
BOUNDARY_IMAGES_PER_ITER = 20
PSEUDO_LABELS_PER_CLASS  = 50
TARGET_ACCURACY          = 0.99
MIN_IMPROVEMENT_PERCENT  = 0.1   # stop if gain < 0.1%

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

GROUND_TRUTH_CSV_PATH = "./ground_truth.csv"
PRACTICAL_GT_SIZE = 500

# ==========================================
# 1. Data Ingestion
# ==========================================
def load_dataset(path):
    print("Loading and sorting dataset...")
    files = [f for f in os.listdir(path) if f.endswith('.bmp')]
    # CRITICAL: Sort numerically to match the oracle (1.bmp, 2.bmp ... 10000.bmp)
    files = sorted(files, key=lambda x: int(os.path.splitext(x)[0]))

    images = []
    for f in files:
        img_path = os.path.join(path, f)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        images.append(img.flatten().astype(np.float32) / 255.0)

    return np.array(images), files


# ==========================================
# 2. Interactive Manual Labeling
# ==========================================
def manual_label_images(indices, images, files, title="Manual Labeling"):
    """Displays images one by one and waits for key 0-9 from the annotator."""
    labels = []
    total_images = len(indices)

    for img_num, idx in enumerate(indices, 1):
        img_2d = images[idx].reshape(28, 28)

        key_pressed = {'key': None}

        def on_key(event, kp=key_pressed):
            if event.key and event.key.isdigit():
                kp['key'] = event.key
                plt.close('all')

        while key_pressed['key'] is None:
            fig = plt.figure(figsize=(4, 4))
            plt.imshow(img_2d, cmap='gray')
            plt.title(
                f"{title}\n"
                f"Image {img_num}/{total_images}  |  File: {files[idx]}\n"
                f"Press a digit key 0-9 on your keyboard"
            )
            plt.axis('off')
            fig.canvas.mpl_connect('key_press_event', on_key)
            plt.show(block=True)

        labels.append(int(key_pressed['key']))
        print(f"  [{img_num}/{total_images}] {files[idx]} -> labeled as: {key_pressed['key']}")

    return labels


# ==========================================
# 3. Data Augmentation
# ==========================================
def augment_image(image_vector, label):
    """
    Generates 7 augmented copies of a single 784-D normalised image.
    Transforms applied:
      1-2 : Rotation  +5° and -5°
      3   : Additive Gaussian noise  (mu=0, sigma=0.05)
      4-7 : Spatial shifts  (right, left, down, up)  by 2 pixels
    All copies inherit the seed label with sample_weight = 1.
    NOTE: np.random.seed is set globally so noise is reproducible.
    """
    img = image_vector.reshape(28, 28)
    aug_vectors, aug_labels, aug_weights = [], [], []
    rows, cols = img.shape

    # 1-2: Rotations
    for angle in [5, -5]:
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        rotated = cv2.warpAffine(img, M, (cols, rows))
        aug_vectors.append(rotated.flatten())

    # 3: Gaussian noise
    noise = np.random.normal(0, 0.05, img.shape)
    noisy_img = np.clip(img + noise, 0, 1)
    aug_vectors.append(noisy_img.flatten())

    # 4-7: Spatial shifts  (tx, ty)
    for tx, ty in [(2, 0), (-2, 0), (0, 2), (0, -2)]:
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        shifted = cv2.warpAffine(img, M, (cols, rows))
        aug_vectors.append(shifted.flatten())

    for v in aug_vectors:
        aug_labels.append(label)
        aug_weights.append(1)   # augmented images always get weight 1

    return aug_vectors, aug_labels, aug_weights


# ==========================================
# 3.5 HOG Feature Extraction
# ==========================================
def extract_hog_features(images):
    """
    Extract HOG (Histogram of Oriented Gradients) features from images.
    
    HOG captures edge directions instead of raw pixels:
    - More robust to handwriting variations
    - Smaller feature space (64D instead of 784D)
    - Better generalization for digit recognition
    
    Parameters:
    -----------
    images : ndarray (N, 784) - flattened 28×28 grayscale images
    
    Returns:
    --------
    hog_features : ndarray (N, 64) - HOG features for all images
    """
    hog_features = []
    for img_vector in images:
        img_2d = img_vector.reshape(28, 28)
        # orientations=8: 8 gradient directions
        # pixels_per_cell=(4,4): Cell size (28/4 = 7×7 cells)
        features = hog(img_2d, orientations=8, pixels_per_cell=(4, 4), 
                       cells_per_block=(2, 2), visualize=False)
        hog_features.append(features)
    
    return np.array(hog_features)


# ==========================================
# 4. Practical Ground Truth (CSV-based)
# ==========================================
def build_practical_gt(file_names, logger):
    """
    Load fixed held-out set from ground_truth.csv.
    CSV format: image,label (e.g., "12.bmp,5")

    Returns
    -------
    gt_indices : list[int]   indices into X of the 500 held-out images
    gt_labels  : list[int]   corresponding human-assigned labels
    """
    logger.info("Loading practical ground truth from CSV...")
    print("\n" + "="*70)
    print(f"PRACTICAL GROUND TRUTH SETUP\nLoading from: {GROUND_TRUTH_CSV_PATH}")
    print("="*70)

    name_to_index = {name: idx for idx, name in enumerate(file_names)}
    gt_indices = []
    gt_labels = []
    seen_images = set()

    with open(GROUND_TRUTH_CSV_PATH, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required_cols = {"image", "label"}
        fieldnames = set(reader.fieldnames or [])
        if not required_cols.issubset(fieldnames):
            raise ValueError(
                f"Ground-truth CSV must contain columns {sorted(required_cols)}; "
                f"got {sorted(fieldnames)}"
            )

        for row_num, row in enumerate(reader, start=2):
            image_name = row["image"].strip()
            label_text = row["label"].strip()

            if image_name in seen_images:
                raise ValueError(f"Duplicate image '{image_name}' in ground truth CSV (row {row_num})")
            if image_name not in name_to_index:
                raise ValueError(f"Image '{image_name}' in ground truth CSV not found in dataset")

            try:
                label = int(label_text)
            except ValueError as exc:
                raise ValueError(f"Invalid label '{label_text}' for {image_name} at row {row_num}") from exc

            if label < 0 or label > 9:
                raise ValueError(f"Label out of range for {image_name} at row {row_num}: {label}")

            seen_images.add(image_name)
            gt_indices.append(name_to_index[image_name])
            gt_labels.append(label)

    if len(gt_labels) != PRACTICAL_GT_SIZE:
        raise ValueError(
            f"Ground-truth CSV must contain exactly {PRACTICAL_GT_SIZE} rows; "
            f"found {len(gt_labels)}"
        )

    logger.info(f"Practical GT loaded from CSV: {len(gt_labels)} images")
    print(f"\n[OK] Practical ground truth loaded: {len(gt_labels)} images\n")
    return gt_indices, gt_labels


def evaluate_practical_gt(predictions, gt_indices, gt_labels):
    """
    Computes accuracy on the 500-image held-out set.
    This is the metric a practitioner would use when the oracle is unavailable.
    """
    correct = sum(
        1 for idx, true_lbl in zip(gt_indices, gt_labels)
        if predictions[idx] == true_lbl
    )
    return correct / len(gt_labels)


# ==========================================
# 5. Main Pipeline Execution
# ==========================================
def main():
    # ── Logging setup ─────────────────────────────────────────────────────────
    log_file = f"pipeline2_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()

    start_time = time.time()
    logger.info("=" * 70)
    logger.info("PIPELINE 2 START")
    logger.info("=" * 70)
    logger.info(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("Configuration:")
    logger.info(f"  Random Seed:                     {RANDOM_SEED}  (FIXED — reproducible)")
    logger.info(f"  Dataset Path:                    {DATASET_PATH}")
    logger.info(f"  Total Images:                    {NUM_IMAGES}")
    logger.info(f"  Ground Truth CSV:                {GROUND_TRUTH_CSV_PATH}")
    logger.info(f"  Practical GT Size (expected):    {PRACTICAL_GT_SIZE}")
    logger.info(f"  Seed Size:                       {SEED_SIZE}")
    logger.info(f"  Boundary Images per Iteration:   {BOUNDARY_IMAGES_PER_ITER}")
    logger.info(f"  Pseudo-Labels per Class:         {PSEUDO_LABELS_PER_CLASS}")
    logger.info(f"  Target Accuracy:                 {TARGET_ACCURACY * 100:.2f}%")
    logger.info(f"  Min Improvement Threshold:       {MIN_IMPROVEMENT_PERCENT:.3f}%")

    # ── Load dataset ──────────────────────────────────────────────────────────
    X, file_names = load_dataset(DATASET_PATH)
    logger.info(f"Dataset loaded: {len(X)} images")

    # ── Extract HOG features ──────────────────────────────────────────────────
    X_hog = extract_hog_features(X)
    
    # ── Build practical ground truth FIRST (before any training) ───────
    # These 500 images are held out entirely — they are never added to the
    # training set and are used only to estimate accuracy independently of oracle.
    gt_indices, gt_labels = build_practical_gt(file_names, logger)
    gt_index_set = set(gt_indices)   # fast lookup to exclude from training

    # ── State tracking ────────────────────────────────────────────────────────
    # GT images are pre-excluded from the pool that can be trained on,
    # so they remain a truly held-out set throughout the pipeline.
    trained_indices  = set(gt_indices)   # start with GT excluded
    X_train, y_train, sample_weights = [], [], []
    total_manual_images   = 0   # only seed + boundary labels are manual in this run
    total_pseudo_labels   = 0
    total_rejected_labels = 0
    iteration_stats       = []

    # ── STEP 1: Random Seed Sampling ──────────────────────────────────────────
    print(f"\n--- STEP 1: Manually Labeling {SEED_SIZE} Seed Images ---")
    logger.info(f"\n--- STEP 1: Manually Labeling {SEED_SIZE} Seed Images ---")

    non_gt_pool = [i for i in range(NUM_IMAGES) if i not in gt_index_set]
    seed_indices = random.sample(non_gt_pool, SEED_SIZE)
    seed_labels  = manual_label_images(
        seed_indices, X, file_names,
        title="Seed Set Labeling"
    )
    logger.info(f"Seed images labeled: {len(seed_labels)} images")

    for idx, label in zip(seed_indices, seed_labels):
        X_train.append(X_hog[idx])  # Use HOG features instead of raw pixels
        y_train.append(label)
        sample_weights.append(100)   # human-verified weight
        trained_indices.add(idx)

    total_manual_images += SEED_SIZE
    logger.info("Seed images added to training set")

    # ── STEP 2: Augmentation ──────────────────────────────────────────────────
    print("\n--- STEP 2: Augmenting Seed Data ---")
    logger.info("\n--- STEP 2: Augmenting Seed Data ---")

    n_before_aug = len(X_train)
    for idx, label in zip(seed_indices, seed_labels):
        a_vecs, a_labels, a_weights = augment_image(X[idx], label)
        # Convert augmented images to HOG features
        a_hog = extract_hog_features(np.array(a_vecs))
        X_train.extend(a_hog)
        y_train.extend(a_labels)
        sample_weights.extend(a_weights)

    num_augmented = len(X_train) - n_before_aug
    logger.info(f"Generated {num_augmented} augmented images from {SEED_SIZE} seed images")
    logger.info(f"Training set size after augmentation: {len(X_train)}")
    print(f"Generated {num_augmented} augmented images from {SEED_SIZE} seed images")
    print(f"Training set size after augmentation: {len(X_train)}")

    X_train        = np.array(X_train)
    y_train        = np.array(y_train)
    sample_weights = np.array(sample_weights)

    # ── Active learning loop (Steps 3-6) ──────────────────────────────────────
    iteration    = 1
    prev_accuracy = 0.0
    stop_reason  = ""

    while True:
        # ── STEP 3: Train SVM ─────────────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration}")
        print(f"{'='*60}")
        logger.info(f"\n{'='*60}")
        logger.info(f"ITERATION {iteration}")
        logger.info(f"{'='*60}")
        logger.info(f"Training SVM on {len(X_train)} samples...")
        print(f"Training SVM on {len(X_train)} samples...")

        svm = SVC(kernel='rbf', decision_function_shape='ovr', probability=False)
        svm.fit(X_train, y_train, sample_weight=sample_weights)

        print("Predicting full dataset...")
        logger.info("Predicting full dataset...")
        scores      = svm.decision_function(X_hog)   # (10000, 10) using HOG features
        predictions = svm.predict(X_hog)

        # Margin = top-1 score minus top-2 score
        sorted_scores = np.sort(scores, axis=1)
        margins       = sorted_scores[:, -1] - sorted_scores[:, -2]

        # ── Oracle evaluation ─────────────────────────────────────────────────
        try:
            result = check_accuracy.check_accuracy(predictions)
            acc    = float(result[0]) if isinstance(result, tuple) else float(result)
            print(f"Oracle Accuracy:    {acc * 100:.2f}%")
            logger.info(f"Oracle Accuracy:    {acc * 100:.2f}%")
        except Exception as e:
            logger.error(f"Oracle error: {e}")
            acc = 0.0

        # Practical GT evaluation — independent accuracy estimate
        practical_acc = evaluate_practical_gt(predictions, gt_indices, gt_labels)
        print(f"Practical GT Accuracy ({len(gt_labels)} images): {practical_acc * 100:.2f}%")
        logger.info(f"Practical GT Accuracy ({len(gt_labels)} images): {practical_acc * 100:.2f}%")

        # Improvement calculation: on iteration 1, prev_accuracy=0 so improvement
        # shows the full jump from zero — stagnation check applied from iteration 2 onward.
        improvement = (acc - prev_accuracy) * 100
        if iteration == 1:
            improvement_display = "N/A (first iteration)"
        else:
            improvement_display = f"{improvement:+.3f}%"
        print(f"Improvement from previous iteration: {improvement_display}")
        logger.info(f"Improvement from previous iteration: {improvement_display}")

        # Record stats for this iteration
        iteration_stats.append({
            'iteration'            : iteration,
            'oracle_accuracy'      : acc * 100,
            'practical_accuracy'   : practical_acc * 100,
            'improvement'          : improvement if iteration > 1 else None,
            'boundary_images'      : 0,
            'pseudo_labels_added'  : 0,
            'pseudo_labels_rejected': 0,
            'training_set_size'    : len(X_train),
        })

        # ── Stopping conditions ───────────────────────────────────────────────
        if acc >= TARGET_ACCURACY:
            stop_reason = (
                f"Accuracy target reached "
                f"({acc * 100:.2f}% >= {TARGET_ACCURACY * 100:.2f}%)"
            )
            print(f"\nConvergence reached. {stop_reason}")
            logger.info(f"Convergence reached. {stop_reason}")
            break

        if iteration > 1 and improvement < MIN_IMPROVEMENT_PERCENT:
            stop_reason = (
                f"Improvement below threshold "
                f"({improvement:.3f}% < {MIN_IMPROVEMENT_PERCENT:.3f}%)"
            )
            print(f"\nConvergence reached. {stop_reason}")
            logger.info(f"Convergence reached. {stop_reason}")
            break

        prev_accuracy = acc

        # ── STEP 4: Active Refinement — Boundary Image Labelling ──────────────
        print(f"\n--- STEP 4: Active Refinement (Iteration {iteration}) ---")
        logger.info(f"\n--- STEP 4: Active Refinement (Iteration {iteration}) ---")

        # Select the BOUNDARY_IMAGES_PER_ITER images with the smallest margin
        # that have NOT yet been trained on (GT images are already excluded via
        # trained_indices, so they can never be selected here).
        unlabeled_margins = [
            (i, margins[i])
            for i in range(NUM_IMAGES)
            if i not in trained_indices
        ]
        unlabeled_margins.sort(key=lambda x: x[1])
        boundary_indices = [i for i, _ in unlabeled_margins[:BOUNDARY_IMAGES_PER_ITER]]

        print(f"Manually labeling {len(boundary_indices)} boundary images...")
        logger.info(f"Manually labeling {len(boundary_indices)} boundary images...")
        boundary_labels = manual_label_images(
            boundary_indices, X, file_names,
            title=f"Iter {iteration} — Boundary Labeling"
        )

        # Copy current training arrays before extending
        X_train_new      = list(X_train)
        y_train_new      = list(y_train)
        weights_new      = list(sample_weights)
        boundary_count   = 0

        for idx, label in zip(boundary_indices, boundary_labels):
            X_train_new.append(X_hog[idx])  # Use HOG features
            y_train_new.append(label)
            weights_new.append(100)   # human-verified weight
            trained_indices.add(idx)
            boundary_count += 1

        total_manual_images += boundary_count
        print(f"Added {boundary_count} boundary images to training set")
        logger.info(f"Added {boundary_count} boundary images to training set")

        # ── STEP 5: Self-Training — High-Confidence Pseudo-Labelling ──────────
        print(f"\n--- STEP 5: Self-Training (Iteration {iteration}) ---")
        logger.info(f"\n--- STEP 5: Self-Training (Iteration {iteration}) ---")

        # Dynamic threshold: recalculated from ALL 10k margins each iteration
        threshold_75 = np.percentile(margins, 75)
        print(f"Margin threshold (75th percentile): {threshold_75:.4f}")
        logger.info(f"Margin threshold (75th percentile): {threshold_75:.4f}")

        pseudo_count_iter   = 0
        rejected_count_iter = 0

        for c in range(10):
            # All images predicted as class c that are not yet in training set
            class_candidates = [
                (i, margins[i])
                for i in range(NUM_IMAGES)
                if predictions[i] == c and i not in trained_indices
            ]

            above = [(i, m) for i, m in class_candidates if m > threshold_75]
            below = [(i, m) for i, m in class_candidates if m <= threshold_75]
            rejected_count_iter += len(below)

            # Sort by descending margin, take top PSEUDO_LABELS_PER_CLASS
            above.sort(key=lambda x: x[1], reverse=True)
            for idx, m in above[:PSEUDO_LABELS_PER_CLASS]:
                X_train_new.append(X_hog[idx])  # Use HOG features
                y_train_new.append(c)   # pseudo-label from SVM
                weights_new.append(1)   # machine-confidence weight
                trained_indices.add(idx)
                pseudo_count_iter   += 1
                total_pseudo_labels += 1

        total_rejected_labels += rejected_count_iter
        print(f"Added {pseudo_count_iter} high-confidence pseudo-labels")
        print(f"Rejected {rejected_count_iter} low-confidence candidates")
        logger.info(f"Added {pseudo_count_iter} high-confidence pseudo-labels")
        logger.info(f"Rejected {rejected_count_iter} low-confidence candidates (below threshold)")

        # Update stats entry for this iteration
        iteration_stats[-1].update({
            'boundary_images'       : boundary_count,
            'pseudo_labels_added'   : pseudo_count_iter,
            'pseudo_labels_rejected': rejected_count_iter,
            'training_set_size'     : len(X_train_new),
        })

        # Update training arrays for next iteration
        X_train        = np.array(X_train_new)
        y_train        = np.array(y_train_new)
        sample_weights = np.array(weights_new)
        iteration     += 1

    # ── FINAL REPORTING ───────────────────────────────────────────────────────
    end_time     = time.time()
    elapsed_time = end_time - start_time

    # Manual time breakdown
    gt_time       = 0                                   # loaded from CSV (no manual input)
    seed_time     = SEED_SIZE * 10                      # 300 × 10 s
    boundary_imgs = total_manual_images - SEED_SIZE
    boundary_time = boundary_imgs * 10
    total_manual_time = seed_time + boundary_time

    print("\n" + "=" * 70)
    print("PIPELINE 2 FINAL SUMMARY")
    print("=" * 70)
    print("\nITERATION-BY-ITERATION STATISTICS:")
    print("-" * 110)
    print(f"{'Iter':<5} {'Oracle Acc':<12} {'Practical Acc':<15} {'Improvement':<14} "
          f"{'Boundary':<10} {'Pseudo+':<10} {'Pseudo-':<10} {'Train Size':<10}")
    print("-" * 110)
    for s in iteration_stats:
        imp_str = f"{s['improvement']:+.3f}%" if s['improvement'] is not None else "N/A"
        print(
            f"{s['iteration']:<5} "
            f"{s['oracle_accuracy']:>10.2f}% "
            f"{s['practical_accuracy']:>13.2f}% "
            f"{imp_str:>13} "
            f"{s['boundary_images']:>9} "
            f"{s['pseudo_labels_added']:>9} "
            f"{s['pseudo_labels_rejected']:>9} "
            f"{s['training_set_size']:>9}"
        )
    print("-" * 110)

    print("\nMANUAL TIME BREAKDOWN:")
    print(f"  Practical GT source:        {GROUND_TRUTH_CSV_PATH} ({len(gt_labels)} images, 0 s manual)")
    print(f"  Seed labelling:             {SEED_SIZE} images × 10 s = {seed_time:>7} s  ({seed_time/60:.1f} min)")
    print(f"  Boundary labelling:         {boundary_imgs} images × 10 s = {boundary_time:>7} s  ({boundary_time/60:.1f} min)")
    print(f"  {'─'*55}")
    print(f"  Total manual time:          {total_manual_images} images × 10 s = {total_manual_time:>7} s  ({total_manual_time/60:.1f} min)")

    print("\nOVERALL SUMMARY:")
    print(f"  Random Seed Used:                    {RANDOM_SEED}  (results reproducible)")
    print(f"  Total Iterations Performed:          {iteration}")
    print(f"  Final Oracle Accuracy:               {acc * 100:.2f}%")
    print(f"  Final Practical GT Accuracy:         {practical_acc * 100:.2f}%")
    print(f"  Oracle vs Practical difference:      {abs(acc - practical_acc) * 100:.2f}%")
    print(f"  Stopping Reason:                     {stop_reason}")
    print(f"  Practical GT Source:                 {GROUND_TRUTH_CSV_PATH} ({len(gt_labels)} fixed images)")
    print(f"  Total Images Manually Labeled:       {total_manual_images}")
    print(f"    - Seed:                            {SEED_SIZE}")
    print(f"    - Boundary (all iters):            {boundary_imgs}")
    print(f"  Total Pseudo-Labels Accepted:        {total_pseudo_labels}")
    print(f"  Total Pseudo-Labels Rejected:        {total_rejected_labels}")
    print(f"  Total Manual Labelling Time:         {total_manual_time} s  ({total_manual_time/60:.1f} min)")
    print(f"  Baseline (full manual):              100,000 s (27.78 h)")
    print(f"  Time Saved:                          {100 - (total_manual_time/100000)*100:.1f}%")

    hours, rem     = divmod(elapsed_time, 3600)
    minutes, secs  = divmod(rem, 60)
    print(f"\n  Pipeline Execution Time:             {elapsed_time:.2f} s  "
          f"({int(hours):02d}:{int(minutes):02d}:{secs:05.2f})")
    print("=" * 70)

    # Mirror full summary to log file
    logger.info("=" * 70)
    logger.info("PIPELINE 2 FINAL SUMMARY")
    logger.info("=" * 70)
    logger.info("\nITERATION-BY-ITERATION STATISTICS:")
    logger.info("-" * 110)
    logger.info(
        f"{'Iter':<5} {'Oracle Acc':<12} {'Practical Acc':<15} {'Improvement':<14} "
        f"{'Boundary':<10} {'Pseudo+':<10} {'Pseudo-':<10} {'Train Size':<10}"
    )
    logger.info("-" * 110)
    for s in iteration_stats:
        imp_str = f"{s['improvement']:+.3f}%" if s['improvement'] is not None else "N/A"
        logger.info(
            f"{s['iteration']:<5} "
            f"{s['oracle_accuracy']:>10.2f}% "
            f"{s['practical_accuracy']:>13.2f}% "
            f"{imp_str:>13} "
            f"{s['boundary_images']:>9} "
            f"{s['pseudo_labels_added']:>9} "
            f"{s['pseudo_labels_rejected']:>9} "
            f"{s['training_set_size']:>9}"
        )
    logger.info("-" * 110)
    logger.info(f"\n  Random Seed:                         {RANDOM_SEED}")
    logger.info(f"  Total Iterations:                    {iteration}")
    logger.info(f"  Final Oracle Accuracy:               {acc * 100:.2f}%")
    logger.info(f"  Final Practical GT Accuracy:         {practical_acc * 100:.2f}%")
    logger.info(f"  Stopping Reason:                     {stop_reason}")
    logger.info(f"  Practical GT Source:                 {GROUND_TRUTH_CSV_PATH} ({len(gt_labels)} fixed images)")
    logger.info(f"  Total Images Manually Labeled:       {total_manual_images}")
    logger.info(f"  Total Pseudo-Labels Accepted:        {total_pseudo_labels}")
    logger.info(f"  Total Pseudo-Labels Rejected:        {total_rejected_labels}")
    logger.info(f"  Total Manual Time:                   {total_manual_time} s ({total_manual_time/60:.1f} min)")
    logger.info(f"  Time Saved:                          {100 - (total_manual_time/100000)*100:.1f}%")
    logger.info(f"  Pipeline Execution Time:             {elapsed_time:.2f} s")
    logger.info("=" * 70)
    logger.info(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("PIPELINE 2 END")
    logger.info("=" * 70)

    print(f"\n[OK] Log saved to: {log_file}")


if __name__ == "__main__":
    plt.ion()
    main()