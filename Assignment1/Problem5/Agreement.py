import csv
import os

# ── Set your file paths here ──────────────────────────────────────────────────
bench_file  = r"F:/Uni/EECE 4/Semester 2/Neural Networks/Assignments/Assign1/Fab/bench_gpt-5.4.csv"        # 500 labels (used as index only)
gt_file     = r"F:/Uni/EECE 4/Semester 2/Neural Networks/Assignments/Assign1/Fab/true_labels.csv"     # 10k ground truth
llm1_file   = r"F:/Uni/EECE 4/Semester 2/Neural Networks/Assignments/Assign1/Fab/bench_gpt-4.1_10k.csv"
llm2_file   = r"F:/Uni/EECE 4/Semester 2/Neural Networks/Assignments/Assign1/Fab/bench_gpt-5.4_10k.csv"

# Output files (saved in same folder as inputs)
out_dir      = r"F:\Uni\EECE 4\Semester 2\Neural Networks\Assignments\Assign1\Fab"
agreed_file  = os.path.join(out_dir, "agreed.csv")
disagreed_file = os.path.join(out_dir, "disagreed.csv")
# ─────────────────────────────────────────────────────────────────────────────


def load_csv(filepath):
    """Load CSV as dict: image -> second column value."""
    data = {}
    with open(filepath, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        fieldnames = [k.strip() for k in reader.fieldnames]
        image_col = fieldnames[0]
        value_col = fieldnames[1]
        for row in reader:
            row = {k.strip(): v.strip() for k, v in row.items()}
            data[row[image_col]] = row[value_col]
    return data


def load_index(filepath):
    """Load only the image indices (first column) from the 500-label file."""
    indices = set()
    with open(filepath, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        fieldnames = [k.strip() for k in reader.fieldnames]
        image_col = fieldnames[0]
        for row in reader:
            row = {k.strip(): v.strip() for k, v in row.items()}
            indices.add(row[image_col])
    return indices


# ── Load all files ────────────────────────────────────────────────────────────
print("Loading files...")
bench_indices = load_index(bench_file)
ground_truth  = load_csv(gt_file)
llm1_preds    = load_csv(llm1_file)
llm2_preds    = load_csv(llm2_file)

print(f"  Benchmark index size : {len(bench_indices)} images")
print(f"  Ground truth size    : {len(ground_truth)} images")
print(f"  LLM1 predictions     : {len(llm1_preds)} images")
print(f"  LLM2 predictions     : {len(llm2_preds)} images")

# ── Step 1: Agreement analysis across all 10k images ─────────────────────────
print("\n── Step 1: Agreement Analysis (all 10k) ────────────────────────────────")

all_images = set(llm1_preds.keys()) | set(llm2_preds.keys())

agreed_rows    = []
disagreed_rows = []

for image in all_images:
    label1 = llm1_preds.get(image, "")
    label2 = llm2_preds.get(image, "")

    if label1 == "" or label2 == "":
        # One is missing — treat as disagreement
        disagreed_rows.append({
            "image"     : image,
            "llm1_pred" : label1,
            "llm2_pred" : label2
        })
    elif label1 == label2:
        agreed_rows.append({
            "image"        : image,
            "agreed_label" : label1
        })
    else:
        disagreed_rows.append({
            "image"     : image,
            "llm1_pred" : label1,
            "llm2_pred" : label2
        })

total         = len(all_images)
n_agreed      = len(agreed_rows)
n_disagreed   = len(disagreed_rows)
agreement_rate = n_agreed / total * 100

print(f"  Total images         : {total}")
print(f"  Agreed               : {n_agreed}  ({agreement_rate:.2f}%)")
print(f"  Disagreed            : {n_disagreed}  ({100 - agreement_rate:.2f}%)")

# Save agreed CSV
with open(agreed_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=["image", "agreed_label"])
    writer.writeheader()
    # Sort numerically by image index
    agreed_rows.sort(key=lambda r: int(r["image"]) if r["image"].isdigit() else r["image"])
    writer.writerows(agreed_rows)
print(f"\n  Agreed labels saved to   : {agreed_file}")

# Save disagreed CSV
with open(disagreed_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=["image", "llm1_pred", "llm2_pred"])
    writer.writeheader()
    disagreed_rows.sort(key=lambda r: int(r["image"]) if r["image"].isdigit() else r["image"])
    writer.writerows(disagreed_rows)
print(f"  Disagreed labels saved to: {disagreed_file}")

# ── Step 2: Accuracy on the 500-image benchmark subset ───────────────────────
print("\n── Step 2: Accuracy on 500-image Benchmark Subset ─────────────────────")

# Build a quick lookup for agreed labels
agreed_lookup = {row["image"]: row["agreed_label"] for row in agreed_rows}

in_bench_and_agreed = 0
correct             = 0
not_agreed_in_bench = 0

for image in bench_indices:
    if image in agreed_lookup:
        in_bench_and_agreed += 1
        if agreed_lookup[image] == ground_truth.get(image, ""):
            correct += 1
    else:
        not_agreed_in_bench += 1

accuracy = correct / in_bench_and_agreed * 100 if in_bench_and_agreed > 0 else 0.0

print(f"  Benchmark images (total)         : {len(bench_indices)}")
print(f"  Benchmark images with agreement  : {in_bench_and_agreed}")
print(f"  Benchmark images with disagreement (excluded): {not_agreed_in_bench}")
print(f"  Correct agreed labels vs GT      : {correct}")
print(f"  Accuracy (agreed subset only)    : {accuracy:.2f}%")

print("\nDone.")