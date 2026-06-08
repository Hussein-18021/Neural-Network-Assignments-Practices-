# Assignment 3 — Generative Models, Attention & Arabic NLP

## Overview
The most advanced assignment in the course, covering three distinct areas: generative model comparison (VAE vs. GAN) under low-data conditions, attention mechanisms in CNNs, and Arabic NLP evaluation using LLMs.

---

## Problem 1 — VAE Synthetic Data with Low-Data Stabilization

**Setting:** Only 350 real examples per digit available.

**Pipeline:**
1. Augment the 350 real examples by 10–20× (rotations, shifts, scaling, light noise)
2. Train a **conditional VAE** on real + augmented data
3. Generate 5 independent runs of 1000 synthetic samples per digit (z ~ N(0,I))
4. Filter generated samples using a LeNet-5 classifier (trained on 350 real examples) by softmax confidence:
   - **Set A:** all generated samples
   - **Set B:** high-confidence only (confidence ≥ 0.9)
   - **Set C:** mid-confidence (0.6 ≤ confidence < 0.9)
5. Train LeNet-5 on 350 real + each synthetic set
6. Compare against: 350-real baseline, 1000-real baseline, GAN results (Problem 2)

**Research question:** Does VAE-generated data benefit more from high-confidence selection or mid-confidence diversity?

---

## Problem 2 — GAN Synthetic Data with Low-Data Stabilization

**Setting:** Identical to Problem 1 (350 real examples per digit).

**Pipeline:**
1. Augment the 350 real examples by 10–20× (same transforms as Problem 1)
2. Train a **conditional DCGAN (cDCGAN)** on real + augmented data
3. Generate 5 independent runs of 1000 synthetic samples per digit
4. Apply the same confidence-based filtering (Sets A, B, C) using the same LeNet-5 classifier
5. Train LeNet-5 on 350 real + each synthetic set
6. Compare against: 350-real baseline, 1000-real baseline, VAE results (Problem 1)

**Research question:** Which GAN selection strategy gives the best improvement? Does synthetic data reduce the need for real data?

### Combined Results Table (Problems 1 & 2)

| Model | Set | Confidence Level | No. of Examples | Accuracy |
|---|---|---|---|---|
| VAE | A | All generated | | |
| VAE | B | ≥ 0.9 | | |
| VAE | C | 0.6–0.9 | | |
| GAN | A | All generated | | |
| GAN | B | ≥ 0.9 | | |
| GAN | C | 0.6–0.9 | | |
| Baseline | — | 350 real only | 350 | |
| Baseline | — | 1000 real only | 1000 | |

*Values filled from experimental results.*

---

## Problem 3 — Attention Mechanisms

### Part a: Spatial Attention on MNIST
- Baseline: LeNet-5 on ReducedMNIST (from Assignment 2)
- Variant: LeNet-5 + spatial attention module
- Comparison: accuracy and training time; analysis of what the attention mechanism learns to focus on

### Part b: Attention on Speech Spectrograms
- Baseline: CNN from Assignment 2 Problem 3
- Variant: same CNN + attention mechanism
- Comparison: accuracy, training time, and qualitative analysis of attention maps on spectrograms

**Deliverables per part:**
- Network architecture description
- Hyperparameter choices and training setup
- Accuracy/time comparison table (with vs. without attention)
- Analysis of how attention affected results
- Suggestions for future improvements

---

## Problem 4 — Arabic Information Retrieval & RAG

**Approach:** Classical keyword search vs. semantic search vs. Retrieval-Augmented Generation (RAG) over a single Arabic book corpus.

**Pipeline:**
1. Select a public-domain Arabic book; split into 2–4 sentence paragraphs
2. Generate embeddings using a multilingual/Arabic sentence embedding model; index with FAISS
3. Build a retrieval interface supporting:
   - Classical search (TF-IDF or BM25)
   - Semantic search (embedding similarity)
4. Extend to a RAG system: for each query, retrieve relevant passages → feed to a small open-source LLM (1B–3B parameters from HuggingFace) → compare RAG answer vs. LLM-only answer

**Evaluation:** 10 Arabic queries covering direct and indirect questions, easy and hard, tested across all three retrieval/generation modes.

**Model constraint:** Must be downloadable from HuggingFace and runnable locally on CPU or limited resources.

---

## Problem 5 — Arabic LLM Benchmarking

**Models evaluated:** ChatGPT, Gemini, ALLaM, Jais, Fanar  
**Task:** One assigned Arabic NLP task from a predefined list of 27 tasks (e.g., sentiment analysis, NER, summarization, diacritization, etc.)

**Dataset requirements:**
- Balanced mix: ~1/3 Modern Standard Arabic, ~1/3 Classical Arabic, ~1/3 Arabic dialects
- Real data majority; all samples include a reference (gold) output
- Inter-annotator agreement reported (≥2 annotators per sample)

**Evaluation protocol:**
- All five models evaluated with identical inputs and prompt format
- Quantitative metric defined per task (Accuracy, F1, BLEU, ROUGE, etc.)
- Error analysis: ≥3 incorrect outputs, ≥3 Arabic-specific challenges, ≥2 borderline cases

**Deliverables:** Comparison table across 5 models, at least one visualization, selected output examples, 8-page report.

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
- Pre-training the generative model on augmented data (before generating synthetic samples) is essential for stability at 350 real examples — raw GAN training on 350 samples is prone to mode collapse.
- Confidence filtering (Set B, ≥0.9) often outperforms using all generated data (Set A) since it discards low-quality generations near class boundaries.
- Attention mechanisms improve speech spectrogram classification more noticeably than MNIST classification, since spectrograms have stronger spatial locality in the frequency-time plane.
- LLM performance on 28×28 grayscale images varies significantly; upscaling to 224×224 before inference substantially improves accuracy in Pipeline 3 of Assignment 1.