# Neural Networks & ML — Course Assignments & Projects
**Course:** Neural Networks (ELC4028) — Elective  
**Institution:** Cairo University, Faculty of Engineering — EECE Department  
**Author:** Hussein Hassan

---

## Repository Structure

| Folder | Topic | Key Techniques |
|---|---|---|
| [`Assignment1/`](./Assignment1/) | Classical ML Methods | Regression, K-means, SVM, Semi-automatic Labelling |
| [`Assignment2/`](./Assignment2/) | Artificial Neural Networks | MLP, CNN (LeNet-5), Autoencoders, GANs, Data Augmentation |
| [`Assignment3/`](./Assignment3/) | Generative Models & NLP | VAE, cDCGAN, Attention Mechanisms, Arabic NLP, LLM Benchmarking |
| [`Final_Project/`](./Final_Project/) | ANN Concept Demonstration | AI-assisted interactive concept explanation |

---

## Assignment Summaries

### Assignment 1 — Classical Machine Learning Methods
Three-part assignment covering the foundations of ML before neural networks:
- **Part 1 — Regression:** Polynomial regression (linear, quadratic, cubic) on real-world datasets (insurance data, heating oil consumption), with outlier detection and R² model comparison.
- **Part 2 — Classical ML on ReducedMNIST:** Feature extraction (DCT, PCA, HOG) combined with K-means clustering and SVM classifiers; full comparative accuracy/timing table across all feature–classifier combinations.
- **Part 3 — Semi-automatic Labelling:** Three labelling pipelines on a 10,000-image unlabelled Indian digits dataset — K-means bootstrapping + active SVM, manual seed + self-training, and LLM-based zero-shot labelling with agreement validation — all targeting ≥99% labelling accuracy.

### Assignment 2 — Artificial Neural Networks
Builds on Assignment 1 by replacing classical classifiers with neural networks:
- **Problem 1 — MLP:** Multilayer perceptrons (1, 3, 4 hidden layers) on ReducedMNIST using DCT, PCA, and autoencoder features.
- **Problem 2 — CNN:** LeNet-5 architecture adapted for 28×28 MNIST images, with multiple hyperparameter variations (filters, activation functions, layer modifications).
- **Problem 3 — Speech Recognition:** CNN-based digit recognition from spectrogram images, with speech-domain and image-domain augmentation experiments.
- **Problem 4 — Autoencoder for Speech:** Autoencoder trained to compress variable-length speech utterances into fixed-length vectors for classification.
- **Problem 5 — Data Augmentation Study:** Systematic LeNet-5 experiments measuring the effect of augmentation volume (0–2000 synthetic examples per digit) across different real-data sizes (350 / 750 / 1000 real examples).
- **Problem 6 — GAN Synthetic Data:** Conditional GAN (cDCGAN) trained on 350 real examples per digit to generate synthetic training data; comparison against augmentation-only baselines.

### Assignment 3 — Generative Models, Attention & Arabic NLP
Advanced topics:
- **Problem 1 — VAE with Confidence Filtering:** Conditional VAE trained on 350 real examples with augmentation pre-training; generated samples filtered by LeNet-5 confidence scores into high/mid/all sets and compared against GAN and real-data baselines.
- **Problem 2 — Stabilized cDCGAN:** Same low-data regime as Problem 1 with augmentation pre-training; confidence-filtered synthetic data evaluated against VAE results.
- **Problem 3 — Attention Mechanisms:** Spatial attention added to LeNet-5 for MNIST and speech spectrogram classification; ablation study comparing attention vs. no-attention variants.
- **Problem 4 — Arabic Information Retrieval & RAG:** Classical (TF-IDF/BM25) vs. semantic (embedding-based) retrieval over an Arabic book corpus; RAG system using a small open-source LLM from HuggingFace.
- **Problem 5 — Arabic LLM Benchmarking:** Evaluation of ChatGPT, Gemini, ALLaM, Jais, and Fanar across a defined Arabic NLP task, with a constructed dataset, inter-annotator agreement, and quantitative metrics.

### Final Project — AI-Assisted ANN Concept Demonstration
Each group selects one ANN concept and produces: a visual presentation, an interactive simulation or demo, 20 MCQs, and a report — using AI tools as assistants throughout. Goal is clear, accessible explanation of the chosen concept.

---

## Tools & Environment
- **Languages:** Python, MATLAB
- **Libraries:** PyTorch / TensorFlow, scikit-learn, HuggingFace Transformers, FAISS, MLflow
- **Datasets:** ReducedMNIST (10k train / 2k test), Indian Digits (10k unlabelled), spoken digits (spectrogram)

---

## Notes
- Each assignment folder contains its own README with problem-level details, results, and observations.
- Code is organized per problem within each assignment folder.
- All experiments were run and verified; result tables are included in the respective notebooks/reports.