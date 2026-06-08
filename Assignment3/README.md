# Assignment 3 — Generative Models, Attention & Arabic NLP

## Overview
The most advanced assignment in the course, covering three distinct areas: generative model comparison (VAE vs. GAN) under low-data conditions, attention mechanisms in CNNs, and Arabic NLP evaluation using LLMs.

---

## Problem 1 — VAE Synthetic Data with Low-Data Stabilization

**Setting:** Only 350 real examples per digit (ReducedMNIST). Test set: 200 images/digit (2,000 total).

**Pipeline:**
1. Augment 350 real examples **15×** (rotation ±15°, affine translate ±7%/scale [0.9,1.1], Gaussian noise σ=0.02) → **56,000 training images** for the VAE.
2. Train a **Conditional VAE** (latent dim=20, KL annealing β: 0→1 over 20 epochs) for 80 epochs × 5 independent runs.
3. Generate 5 × 1,000 synthetic samples/digit = **50,000 total**.
4. Filter by a LeNet-5 confidence scorer (trained on 350 real only):
   - **Set A:** all 50,000 samples
   - **Set B:** conf ≥ 0.9 → 37,770 samples (3,777/digit)
   - **Set C:** 0.6 ≤ conf < 0.9 → 1,570 samples (157/digit)
   - **Set D (causal control):** high-conf, subsampled to 157/digit (same size as Set C)
5. Train LeNet-5 on 350 real + each set; evaluate on held-out test set.

### VAE Training Convergence

| Epoch | Avg Loss | β |
|---|---|---|
| 1 | 146.52 | 0.05 |
| 20 | 136.54 | 1.00 |
| 40 | 132.44 | 1.00 |
| 60 | 131.22 | 1.00 |
| 80 | 129.98 | 1.00 |

All 5 runs converge to within ±0.09 of each other by epoch 80 (range: 129.88–130.05).

### Confidence Distribution (50,000 generated samples)

| Statistic | Value |
|---|---|
| Min confidence | 0.2150 |
| Mean confidence | 0.9526 |
| Max confidence | 1.0000 |

### Classification Accuracy

| Training Configuration | Samples | Test Accuracy |
|---|---|---|
| 350-real baseline | 3,500 | 96.55% |
| 1,000-real baseline | 10,000 | 98.00% |
| 350-real + augmentation only | 56,000 | **98.25%** |
| 350-real + Set A (all VAE) | 53,500 | 97.80% |
| 350-real + Set B (conf ≥ 0.9) | 41,270 | 98.05% |
| 350-real + Set C (0.6 ≤ conf < 0.9) | 5,070 | 97.20% |
| 350-real + Set D (size-matched high-conf) | 5,070 | 96.95% |

**Key findings:**
- **High-confidence selection (Set B, 98.05%)** is the best VAE strategy, marginally surpassing the 1,000-real baseline (+0.05 pp).
- Simple augmentation (98.25%) remains the strongest single method overall.
- The Set D causal control shows Set C beats Set D by only **+0.25 pp** at equal volume — below the noise floor — confirming that dataset size ratio, not mid-confidence diversity, drives Set C's performance.

---

## Problem 2 — GAN Synthetic Data with Low-Data Stabilization

**Setting:** Identical to Problem 1 (350 real examples/digit). Architecture: cDCGAN with spectral normalisation, projection conditioning, R1 gradient penalty.

**Pipeline:** Same augmentation + same confidence filtering (Sets A/B/C) + same LeNet-5 evaluation.

### GAN Confidence Set Sizes

| Set | Confidence Range | Total Samples |
|---|---|---|
| Set A (all) | [0, 1] | 50,000 |
| Set B (high-conf) | ≥ 0.9 | 28,546 |
| Set C (mid-conf) | [0.6, 0.9) | 13,431 |

### Classification Performance

| Dataset Configuration | Training Samples | Test Accuracy | Macro F1 |
|---|---|---|---|
| Baseline-350 | 3,500 | 95.45% | 0.9544 |
| Baseline-1000 | 10,000 | 97.45% | 0.9745 |
| Real-350 + Set A (all GAN) | 53,500 | 97.70% | 0.9770 |
| Real-350 + Set B (conf ≥ 0.9) | 32,046 | 97.55% | 0.9755 |
| **Real-350 + Set C (0.6 ≤ conf < 0.9)** | **16,931** | **97.85%** | **0.9785** |

**Key finding:** For the GAN, **mid-confidence selection (Set C, 97.85%)** wins — the opposite of the VAE result. This is because the GAN's high-confidence pool is smaller (2,855/digit vs 3,777/digit for the VAE), making mid-confidence diversity more valuable. All GAN configurations exceed both the 350-real and 1,000-real baselines.

### Combined VAE vs. GAN Summary

| Method | Configuration | Synthetic/digit | Test Acc |
|---|---|---|---|
| Baseline | 350 real only | 0 | 96.55% |
| Baseline | 1,000 real only | 0 | 98.00% |
| Augmentation (this work) | Affine + noise (15×) | 5,250 | **98.25%** |
| GAN (cDCGAN) | Set C (mid-conf) | 1,343 | 97.85% |
| VAE | Set B (high-conf) | 3,777 | 98.05% |
| VAE | Set A (all) | 5,000 | 97.80% |
| GAN | Set A (all) | 5,000 | 97.70% |

---

## Problem 3 — Attention Mechanisms

### Experiment 1: LeNet-5 + Spatial Attention on ReducedMNIST

**Training:** Adam (lr=1e-3), 10 epochs, batch size 64, input resized to 32×32. Training set: 10,000; test set: 2,000.

The spatial attention gate is inserted after Conv2. It computes parallel avg/max projections → concatenate → 7×7 conv → sigmoid → element-wise multiply with feature map.

| Model | Test Acc | Train Time | Acc Gain | Time Overhead |
|---|---|---|---|---|
| LeNet-5 (baseline) | 96.45% | 71.3 s | — | — |
| LeNet-5 + Spatial Attention | 96.70% | 78.9 s | **+0.25 pp** | +7.6 s (+10.6%) |

**Analysis:** The modest +0.25 pp gain reflects limited headroom on this saturated benchmark. The attention gate adds only +10.6% training time. Model B peaked at epoch 10 vs epoch 8 for the baseline — attention weights require additional gradient updates to stabilise.

### Experiment 2: CNN + CBAM Attention on Spoken Digit Spectrograms

**Architecture:** 3-block CNN (32→64→128 ch) with CBAM after each block. Input: 64×64 log-mel spectrograms. Adam (lr=1e-3), StepLR (γ=0.5, step=10), 25 epochs, batch size 32.

CBAM applies sequential channel gating (global avg/max pool → shared MLP → sigmoid) then spatial gating (7×7 conv → sigmoid).

| Model | Test Acc | Train Time | Acc Gain | Time Overhead |
|---|---|---|---|---|
| CNN (baseline) | 97.0% | 93.6 s | — | — |
| CNN + CBAM | 97.7% | 155.6 s | **+0.70 pp** | +62.0 s (+66.2%) |

**Analysis:** The larger gain vs. Experiment 1 is consistent with spectrograms having richer spatial locality (phonetic content is localised in frequency and time). The CBAM model peaked at epoch 10 vs epoch 2 for the baseline. Training overhead (+66.2%) is disproportionate for the +0.70 pp gain, motivating investigation of lighter alternatives (SE-only, ECA-Net).

**Suggestions for future improvements:**
- Evaluate SE-only blocks as a lower-cost alternative to full CBAM.
- Add a single multi-head self-attention layer at the bottleneck for long-range temporal dependencies.
- Apply SpecAugment (random frequency/time masking).
- Use learning-rate warm-up + cosine annealing to stabilise early attention gradient updates.

---

## Problem 4 — Arabic Information Retrieval & RAG

**Book:** *The Muqaddimah* (Ibn Khaldun, 1377 CE) — chosen because its 14th-century sociological theory is underrepresented in modern LLM training data, making retrieval benefit most visible.

**Pipeline:**
- Text normalised (diacritics removal, alef/teh-marbuta normalisation, stop-word removal) → split into 42 chunks (2–4 sentences each).
- Embeddings: `intfloat/multilingual-e5-large` (1024-dim, "passage:"/"query:" prefixes) → FAISS IndexFlatIP.
- Classical retrieval: BM25Okapi.
- Hybrid retrieval: Reciprocal Rank Fusion (RRF, k=60) of BM25 + semantic rankings.
- Two LLMs compared: **Qwen2.5-3B-Instruct** (general multilingual) and **SILMA-Kashif-2B-Instruct** (Arabic-specialised).
- 10 evaluation queries (5 direct, 5 indirect).

### Retrieval Comparison: BM25 vs. Semantic Search (relevant chunks in top-5)

| Query | Type | BM25 | Semantic |
|---|---|---|---|
| 1 | Direct | 5 | 5 |
| 2 | Direct | 2 | 3 |
| 3 | Direct | 5 | 4 |
| 4 | Direct | 5 | 5 |
| 5 (taxation) | Direct | **0** | 3 |
| 6 | Indirect | 4 | 5 |
| 7 | Indirect | 3 | 4 |
| 8 (climate) | Indirect | **0** | 4 |
| 9 | Indirect | 3 | 5 |
| 10 | Indirect | 2 | 4 |
| **Average** | | **2.9** | **4.2** |

### RAG vs. LLM-Only Answer Quality

Ratings: G = Good (accurate, specific), P = Partial, W = Weak/inaccurate. CJK = Chinese character contamination.

| Q | Type | Qwen RAG | Qwen LLM-only | SILMA RAG | SILMA LLM-only |
|---|---|---|---|---|---|
| 1 | Direct | G | W | P | W |
| 2 | Direct | G | W | P | W |
| 3 | Direct | P (CJK) | W (CJK) | P | W |
| 4 | Direct | P | W | G | W |
| 5 | Direct | W | W (CJK) | P | W |
| 6 | Indirect | G (CJK) | W (CJK) | P | W |
| 7 | Indirect | G | P (CJK) | P | P |
| 8 | Indirect | P (CJK) | P | P | P |
| 9 | Indirect | G | P | G | P |
| 10 | Indirect | G | W (CJK) | G | P |
| **Summary** | | **6G/2P/2W** | **0G/2P/8W** | **3G/7P/0W** | **0G/4P/6W** |

**Key findings:**
- RAG dominates LLM-only for both models across all query types.
- **Qwen 3B** produces more detailed answers but suffers persistent Chinese-language contamination (6/20 answers contain Chinese characters), worsening at 3B vs 1.5B.
- **SILMA 2B** never produces Chinese text but generates shorter, less detailed answers. Achieves 0 Weak RAG ratings.
- LLM-only mode is unreliable: Qwen hallucinates and code-switches; SILMA hallucinates anachronisms (e.g., Ottoman Empire for Ibn Khaldun).
- Semantic search is essential for indirect queries — BM25 returns 0 relevant results for taxation (Q5) and climate (Q8) while semantic search retrieves 3–4.
- Hybrid RRF retrieval combines BM25 keyword precision with semantic synonym handling.

---

## Problem 5 — Arabic LLM Benchmarking: Text Simplification

**Task:** Arabic Text Simplification — rewrite complex paragraphs into simpler form while preserving all key propositions.

**Dataset:** 40 paragraphs (P001–P040): 14 Modern Standard Arabic, 12 Classical Arabic, 14 Dialectal Arabic. 12 thematic domains. 2 independent annotators (native Arabic speakers). **Inter-annotator SARI: 0.4620** (overall), with 90% strong content-unit agreement.

**Models evaluated:** GPT-4o, Gemini 1.5 Pro, ALLaM 7B (4-bit, Colab), Jais 8B (web GUI), Fanar 9B (4-bit, Colab).

**Metric:** SARI (ADD + KEEP + DELETE components, multi-reference, Arabic-normalised).

### Overall SARI Scores

| Model | Single-Ref SARI | Multi-Ref SARI | Rank |
|---|---|---|---|
| **GPT-4o** | **0.4351** | **0.4725** | **1** |
| Jais 8B | 0.4087 | 0.4233 | 2 |
| Gemini 1.5 Pro | 0.3977 | 0.4223 | 3 |
| ALLaM 7B | 0.3994 | 0.4124 | 4 |
| Fanar 9B | 0.3913 | 0.3944 | 5 |
| Human IAA | — | 0.4620 | (ceiling) |

### SARI by Arabic Variety (Multi-Reference)

| Model | MSA (n=14) | Classical (n=12) | Dialect (n=14) | Overall |
|---|---|---|---|---|
| **GPT-4o** | **0.5309** | 0.4401 | 0.4419 | **0.4725** |
| Jais 8B | 0.4276 | 0.3948 | **0.4435** | 0.4233 |
| Gemini 1.5 Pro | 0.4367 | 0.4268 | 0.4041 | 0.4223 |
| ALLaM 7B | 0.4057 | **0.4296** | 0.4043 | 0.4124 |
| Fanar 9B | 0.3716 | **0.4498** | 0.3697 | 0.3944 |
| Human IAA | 0.4745 | 0.4975 | 0.4191 | 0.4620 |

### Output Length Ratio (Output Words / Input Words)

| Model | Length Ratio | Interpretation |
|---|---|---|
| Human A1 | 0.594 | Moderate compression |
| Human A2 | 0.397 | Aggressive compression |
| GPT-4o | 0.548 | Good compression |
| Gemini 1.5 Pro | 0.571 | Good compression |
| Jais 8B | 0.591 | Good compression |
| ALLaM 7B | 1.086 | **Expansion** — elaboration, not simplification |
| Fanar 9B | 1.105 | **Expansion** — elaboration, not simplification |

### SARI Component Breakdown

| Model | ADD | KEEP | DEL | SARI |
|---|---|---|---|---|
| GPT-4o | **0.2287** | 0.4115 | 0.7772 | 0.4725 |
| Jais 8B | 0.1056 | 0.3681 | **0.7963** | 0.4233 |
| Gemini 1.5 Pro | 0.1748 | 0.3917 | 0.7004 | 0.4223 |
| ALLaM 7B | 0.1332 | **0.5024** | 0.4016 | 0.4124 |
| Fanar 9B | 0.1189 | 0.4876 | 0.3767 | 0.3944 |

**Key findings:**
1. **GPT-4o** is the best overall model (0.4725), the only one approaching the human IAA ceiling on MSA (0.5309). Best ADD score (0.2287).
2. **Jais 8B** ranks 2nd (0.4233), leading Dialect (0.4435) by converting dialectal input to simplified MSA — a task-compliance issue (register not preserved) but not a quality failure.
3. **Fanar and ALLaM excel on Classical Arabic** (0.4498 and 0.4296) but fail on the length criterion (ratios >1.0) — they elaborate rather than compress.
4. **Dialectal Arabic remains the primary unsolved challenge.** All models either shift register to MSA (Jais, GPT-4o, Gemini) or produce lower-quality dialect output.
5. **SARI is necessary but insufficient** — it cannot detect hallucinations, penalise register shifts adequately, or reward genuine simplification over reference proximity.
6. Multi-reference evaluation benefits all models, especially GPT-4o (+0.0374).

---

## Files
```
Assignment3/
├── Problem1_VAE/
├── Problem2_GAN/
├── Problem3_Attention/
├── Problem4_ArabicRAG/
└── Problem5_LLMBenchmark/
```

---

## Key Observations
- **Augmentation-first, then generative models.** Simple geometric augmentation (15×) achieves 98.25% — the highest accuracy across all methods — at far less complexity than VAE or GAN training. The best VAE (Set B, 98.05%) and best GAN (Set C, 97.85%) both fall below it.
- **VAE vs GAN confidence filtering.** High-confidence filtering wins for the VAE (larger high-quality pool: 3,777/digit); mid-confidence filtering wins for the GAN (smaller high-confidence pool: 2,855/digit). The optimal filtering strategy depends on the generative model's output distribution.
- **Attention improves spectrograms more than images.** CBAM adds +0.70 pp on speech spectrograms vs +0.25 pp on MNIST, consistent with spectrograms having structured time-frequency content that attention can exploit.
- **RAG substantially helps small Arabic LLMs.** Qwen 3B improves from 0G/8W to 6G/2W with RAG; SILMA 2B from 0G/6W to 3G/0W. Hybrid RRF retrieval is essential for indirect queries where BM25 fails completely.
- **Arabic-native models avoid language contamination.** SILMA 2B produces zero Chinese tokens; Qwen 3B contaminates 30% of outputs with Chinese characters despite three layers of language enforcement.
- **GPT-4o leads Arabic text simplification** (0.4725 multi-ref SARI, best on MSA at 0.5309). ALLaM and Fanar excel on Classical Arabic but expand rather than compress.