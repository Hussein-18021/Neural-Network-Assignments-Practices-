# Final Project — Understanding Transformers

**Course:** Artificial Neural Networks (ANN) — Cairo University, Faculty of Engineering, Spring 2026
**Supervisors:** Dr. Mohsen Rashwan & Dr. Mohamed Abdelghany
**Team:** Serag Khaled, Hussein Mostafa, Mohannad Hassan

**Interactive Demo:** https://transformers-ann-project.netlify.app/

---

## Concept Selected: Transformer Neural Networks

The Transformer architecture, introduced by Vaswani et al. in the 2017 paper *"Attention Is All You Need"*, replaces sequential recurrence (RNNs/LSTMs) with a self-attention mechanism enabling full parallelization and direct modeling of long-range token dependencies. It underpins virtually every state-of-the-art model in NLP, vision, and multimodal AI (GPT, BERT, Claude, Gemini).

---

## Background: Why Transformers?

RNNs and LSTMs suffer from sequential processing (no parallelization), vanishing gradients over long sequences, memory inefficiency, and a scalability ceiling. Transformers address each of these:

| Property | RNN / LSTM | Transformer |
|---|---|---|
| Processing | Sequential | Fully Parallel |
| Long-range dependencies | Difficult (vanishing gradients) | Direct (via self-attention) |
| Training speed | Slow | Fast |
| Memory w/ sequence length | O(n) | O(n²) (attention matrix) |
| Scalability | Limited | Highly scalable |

---

## Architecture Overview

The original Transformer (encoder–decoder, N=6 identical layers each) processes:

**Input → Embeddings + Positional Encoding → Encoder Stack × N → Decoder Stack × N → Linear + Softmax → Output**

### Key Components

- **Token Embeddings** — dense vectors of dimension d_model (typically 512) capturing semantic meaning.
- **Positional Encoding** — sinusoidal encodings added to embeddings to inject sequence-order information (no recurrence means no implicit ordering).
- **Multi-Head Self-Attention** — core innovation: Q, K, V projections per token; attention output = softmax(QKᵀ/√d_k)V; h heads run in parallel attending to different subspaces, outputs concatenated.
- **Feed-Forward Network (FFN)** — two-layer fully connected network applied independently per token: FFN(x) = max(0, xW₁ + b₁)W₂ + b₂.
- **Add & Layer Norm** — residual connections + layer normalization around each sub-layer: LayerNorm(x + Sublayer(x)).
- **Masked Multi-Head Attention (Decoder)** — look-ahead mask sets future positions to −∞ before softmax, preventing the model from attending to future tokens during training.

---

## Deliverables

### 1. Presentation (15 slides)
Dark tech-themed visual style (purple/blue gradients) covering: motivation (RNN limitations) → architecture overview → key components (embeddings, positional encoding, self-attention, multi-head attention, masking) → training & inference → limitations & failure cases → MCQ quiz slide → link to interactive demo.

### 2. Interactive Website
Deployed at **https://transformers-ann-project.netlify.app/** with five sections:
- **Overview** — narrative introduction and historical context.
- **Architecture** — clickable diagram; click any component for on-demand explanation.
- **Self-Attention Demo** — animated attention weights; click a word to see connections to all other tokens update in real time.
- **Interactive Pipeline** — step-by-step text processing walkthrough (Tokenization → Embedding → Positional Encoding → Self-Attention → Feed Forward → Output).
- **Quiz** — 15-question MCQ with instant feedback and running score.

### 3. MCQ Bank
20 multiple-choice questions spanning Bloom's taxonomy levels — from recall to application and analysis. Sample questions cover: purpose of the attention mechanism; which tokens each token attends to; role of Q/K/V; why the √d_k scaling factor is necessary; why Transformers train faster than RNNs; O(n²) complexity trade-off.

### 4. Report (this document)
Documents project aim, architecture background, block diagram of workflow, deliverables, nine challenges encountered with solutions, AI tool usage, and conclusions.

---

## Project Workflow

Topic Selection → Literature Review (Vaswani et al. 2017, Jay Alammar's Illustrated Transformer, Hugging Face docs) → Content Outline → AI-Assisted Generation → Verification & Human Curation → Slide Design → Website Development (deployed on Netlify) → MCQ Bank → Peer Testing → Final Report

---

## Challenges & Solutions

| Challenge | Solution |
|---|---|
| Depth vs. accessibility trade-off | Layered explanation: intuitive visuals first, formal math as secondary annotations; website allows zoom in/out |
| Verifying AI-generated content | Every statement cross-checked against source paper; formulas verified by hand; ambiguous MCQs discarded |
| Making self-attention interactive | Color-coded attention weight connections; click a word to see attended tokens — accurate to core intuition |
| Encoder-only vs. decoder-only vs. encoder-decoder confusion | Scoped explicitly to original 2017 encoder-decoder; noted GPT (decoder-only) and BERT (encoder-only) as variants |
| Designing meaningful MCQs | Multi-level questions (recall → analysis); e.g. asked *why* √(1/d_k) is needed, not just *what* it is |
| No prior web development experience | AI code generation for scaffolding; iterative browser testing and debugging; Netlify drag-and-drop deployment |
| Coherence across three deliverables | Shared terminology reference document (notation, scope boundaries); regular cross-review sessions |
| Tone/register of AI-generated text | Explicit prompting constraints (target audience, assumed prior knowledge, length, tone); prompt-then-edit workflow |
| Hallucinated citations | Strict policy: no citation included unless independently verified against a primary source by at least one team member |

---

## AI Tools Used

| Tool | Purpose | Human Contribution |
|---|---|---|
| Claude (LLM) | Drafting slide explanations, website copy, initial MCQ pool | Reviewed, edited, and verified all outputs |
| Claude (LLM) | Generating initial website component code | Reviewed code, tested functionality, debugged issues |
| AI Design Tools | Suggesting slide layouts and color schemes | Final design decisions and content placement |
| Claude (LLM) | Drafting initial versions of the report | Rewrote sections, verified technical accuracy, added original analysis |

All AI-generated material was reviewed, understood, and verified by the team before inclusion. No output was used as-is.

---

## Key Conclusions

1. Self-attention replacing recurrence was the decisive insight — direct token-to-token interaction at any distance overcomes vanishing gradients and sequential bottlenecks.
2. Multi-head attention is more expressive than single-head — parallel heads capture syntactic, semantic, and positional relationships simultaneously.
3. Trade-offs exist — O(n²) attention complexity, massive data/compute requirements, and limited reasoning capability are genuine limitations.
4. AI tools accelerate but do not replace understanding — every generated output required careful domain-knowledge-based review.
5. Interactive visualization is the most effective teaching tool — seeing attention weights update in real time clarified the mechanism more than formulas alone.

---

## Files

```
Final_Project/
├── presentation/
│   └── Transfomers_Explained_Project.pdf
└── report/
    └── G15.pdf
```

Website (Demo): https://transformers-ann-project.netlify.app/