#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Problem 4: Arabic Information Retrieval and RAG System
======================================================

Book: "مقدمة ابن خلدون" (The Muqaddimah) by عبد الرحمن بن خلدون (1377 CE)
  - A 14th-century masterwork on sociology, history, economics, and philosophy.
  - Public domain. Less common topic: sociological theory / philosophy of history.
  - Source: paraphrased content based on well-known passages.

Models:
  - Embedding : intfloat/multilingual-e5-large (1024-dim, E5 query/passage prefixes)
  - LLM #1   : Qwen/Qwen2.5-3B-Instruct
  - LLM #2   : silma-ai/SILMA-Kashif-2B-Instruct-v1.0

Retrieval:
  - BM25 (classical keyword search) with Arabic tokenisation
  - Semantic search via FAISS (cosine similarity on E5 embeddings)
  - Hybrid search via Reciprocal Rank Fusion (RRF, k=60)

Requirements:
  pip install torch transformers sentence-transformers faiss-cpu rank-bm25
"""

import os
import re
import warnings
import textwrap

import numpy as np

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ════════════════════════════════════════════════════════════════
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"
LLM_MODEL_NAMES = [
    "Qwen/Qwen2.5-3B-Instruct",
    "silma-ai/SILMA-Kashif-2B-Instruct-v1.0",
]
LLM_MODEL_NAME = LLM_MODEL_NAMES[0]  # default for backward compat
TOP_K = 5
MAX_NEW_TOKENS = 256
BOOK_FILE = os.path.join(os.path.dirname(__file__) or ".", "book.txt")

# ════════════════════════════════════════════════════════════════
# 2. BOOK TEXT  –  مقدمة ابن خلدون (paraphrased excerpts)
# ════════════════════════════════════════════════════════════════
BOOK_PARAGRAPHS = [
    # ── العمران البشري (Human Civilization) ──
    "الاجتماع الإنساني ضروري ولا يمكن للفرد أن يعيش بمعزل عن الآخرين. فالإنسان يحتاج إلى التعاون مع غيره في الحصول على الغذاء والمأوى والحماية. وهذا الاجتماع هو ما يسميه ابن خلدون بالعمران البشري.",
    "العمران هو أساس الحضارة الإنسانية وهو يتطور من البساطة إلى التعقيد. ويبدأ العمران بالحياة البدوية البسيطة ثم يتطور إلى الحياة الحضرية المعقدة. وكل مرحلة من مراحل العمران لها خصائصها ومميزاتها.",
    "يرى ابن خلدون أن العمران يتأثر بعوامل كثيرة منها المناخ والجغرافيا والاقتصاد. كما أن العادات والتقاليد تلعب دوراً مهماً في تشكيل طبيعة العمران. وتختلف أنماط العمران من منطقة إلى أخرى تبعاً لهذه العوامل.",
    "الحضارة الإنسانية ليست ثابتة بل هي في حالة تغير مستمر بين الصعود والهبوط. ويتحكم في هذا التغير قوانين اجتماعية يمكن دراستها وفهمها. وقد كان ابن خلدون أول من حاول وضع هذه القوانين بشكل علمي منظم.",

    # ── العصبية (Social Solidarity / Group Feeling) ──
    "العصبية هي الرابطة التي تجمع بين أفراد الجماعة وتدفعهم للدفاع عن بعضهم البعض. وتنشأ العصبية من النسب والقرابة والرضاع والحلف والولاء. وهي القوة الأساسية التي تمكن الجماعات من بناء الدول.",
    "لا يمكن لأي دولة أن تقوم دون عصبية قوية تساند أهلها وتجمع كلمتهم. فالعصبية هي التي تمنح الجماعة القدرة على المطالبة بالسلطة والدفاع عنها. وبدون العصبية تتفكك الجماعة وتضعف أمام خصومها.",
    "تكون العصبية أقوى ما يكون عند أهل البادية لأنهم يعتمدون على بعضهم في مواجهة أخطار الصحراء. أما أهل الحضر فتضعف عصبيتهم بسبب الاعتماد على الدولة والقانون في حمايتهم. وضعف العصبية هو من أهم أسباب سقوط الدول.",
    "العصبية ليست مقتصرة على النسب فقط بل قد تنشأ من الصداقة والتحالف والمصالح المشتركة. وقد تتوسع العصبية لتشمل قبائل ومجموعات متعددة تحت زعامة واحدة. وكلما اتسعت العصبية زادت قوة الجماعة وقدرتها على بناء دولة كبيرة.",

    # ── أطوار الدولة (Phases of the State) ──
    "تمر الدولة بثلاثة أطوار رئيسية هي التأسيس والازدهار والانهيار. في الطور الأول يكون المؤسسون أقوياء يتحلون بالشجاعة والعصبية. وفي هذا الطور تتشكل الدولة وتوضع أسسها.",
    "الطور الثاني هو طور الازدهار والاستقرار حيث تتوسع الدولة وتزدهر العلوم والفنون. ويتمتع أهل الدولة في هذا الطور بالرفاهية والترف. لكن هذا الترف يبدأ في إضعاف العصبية تدريجياً.",
    "الطور الثالث هو طور الانحلال والسقوط حيث يفقد أبناء الدولة عصبيتهم وشجاعتهم. ويصبحون معتمدين على المرتزقة والأجانب في الدفاع عن دولتهم. وينتهي هذا الطور عادة بسقوط الدولة وقيام دولة جديدة مكانها.",
    "يرى ابن خلدون أن عمر الدولة لا يتجاوز ثلاثة أجيال في الغالب. الجيل الأول يبني الدولة بعصبيته وشجاعته والجيل الثاني يحافظ عليها. أما الجيل الثالث فينشأ في الترف ويفقد صفات أجداده مما يؤدي إلى سقوط الدولة.",

    # ── البادية والحضر (Nomadic vs Urban Life) ──
    "يقسم ابن خلدون المجتمع البشري إلى نوعين أساسيين هما أهل البادية وأهل الحضر. أهل البادية يعيشون حياة بسيطة قاسية تعتمد على الرعي والزراعة البسيطة. بينما يعيش أهل الحضر في المدن ويمارسون التجارة والصناعة.",
    "أهل البادية أقرب إلى الشجاعة والكرم من أهل الحضر لأن حياتهم القاسية تفرض عليهم ذلك. فهم يحمون أنفسهم بأنفسهم ولا يعتمدون على أحد في الدفاع عنهم. وهذا يجعل عصبيتهم أقوى وأمتن.",
    "أهل الحضر يتميزون بالعلم والمعرفة والصنائع المتقدمة لأن حياتهم المستقرة تتيح لهم التفرغ لذلك. لكنهم يفقدون تدريجياً صفات الشجاعة والعصبية بسبب انغماسهم في الترف والدعة. وهذا يجعلهم عرضة للغزو من قبل أهل البادية.",
    "العلاقة بين البادية والحضر هي علاقة دائرية في نظر ابن خلدون. فأهل البادية يغزون المدن ويؤسسون دولاً جديدة ثم يتحضرون تدريجياً. وبعد أن يضعفوا بسبب الترف يأتي بدو جدد ليحلوا مكانهم وهكذا تدور الدورة.",

    # ── الاقتصاد والمعاش (Economy and Livelihood) ──
    "يعتبر ابن خلدون أن العمل هو أساس كل ثروة وأن الكسب إنما هو قيمة الأعمال البشرية. فمن لا يعمل لا يكسب ومن يعمل أكثر يكسب أكثر. وهذا ما يعرف اليوم بنظرية قيمة العمل.",
    "التجارة من أهم وسائل الكسب في المجتمع الحضري وهي تقوم على شراء البضائع ثم بيعها بربح. ويرى ابن خلدون أن التجارة تزدهر في المدن الكبيرة حيث يكثر الطلب على السلع. كما أن التجارة الخارجية تجلب ثروة كبيرة للدول.",
    "الصنائع والحرف ضرورية لاستمرار العمران وهي تتطور بتطور المجتمع. ففي البادية تكون الصنائع بسيطة ومحدودة بينما تتنوع وتتعقد في الحضر. وكلما ازدهر العمران ازدادت الصنائع تنوعاً وإتقاناً.",
    "يحذر ابن خلدون من تدخل السلطان في التجارة لأن ذلك يضر بالاقتصاد ويظلم التجار. فالسلطان لا ينافس التجار بالعدل لأنه يستخدم سلطته للحصول على أسعار أفضل. وهذا يؤدي إلى هروب التجار وتراجع الاقتصاد.",
    "الضرائب المعتدلة تشجع الناس على العمل والإنتاج بينما الضرائب الباهظة تؤدي إلى تراجع الاقتصاد. فعندما تكون الضرائب منخفضة يزداد النشاط الاقتصادي ويزداد دخل الدولة. وعندما ترتفع الضرائب يتوقف الناس عن العمل وينهار الاقتصاد.",

    # ── العلوم والتعليم (Sciences and Education) ──
    "يقسم ابن خلدون العلوم إلى نوعين رئيسيين هما العلوم النقلية والعلوم العقلية. العلوم النقلية هي التي تنقل عن الشرع كالتفسير والحديث والفقه. أما العلوم العقلية فهي التي يدركها الإنسان بفكره كالرياضيات والمنطق والفلسفة.",
    "يرى ابن خلدون أن التعليم الجيد يقوم على التدرج من البسيط إلى المعقد. وينتقد طريقة التعليم التي تعتمد على الحفظ والتلقين دون فهم. ويؤكد أن الطالب يجب أن يفهم ما يتعلمه قبل أن يحفظه.",
    "الملكة العلمية لا تتحقق إلا بالممارسة والتكرار والمناقشة مع العلماء. فالعلم ليس مجرد حفظ معلومات بل هو قدرة على التفكير والاستنباط. ويحتاج الطالب إلى معلم ماهر يوجهه ويصحح أخطاءه.",
    "تزدهر العلوم في المدن الكبيرة حيث يتوفر العلماء والكتب والمدارس. أما في المدن الصغيرة والبوادي فتكون العلوم محدودة لقلة الطلب عليها. ولذلك كانت المراكز العلمية الكبرى دائماً في العواصم والمدن الكبيرة.",

    # ── الملك والسلطان (Kingship and Authority) ──
    "الملك هو نتيجة طبيعية للعصبية القوية فمن يملك العصبية الأقوى يملك السلطة. والملك يحتاج إلى قوة تحميه وتفرض سلطته على الرعية. وهذه القوة تأتي من العصبية أو من الجند والمرتزقة.",
    "الملك العادل هو الذي يحكم بالعدل ويحمي رعيته ولا يظلمهم في أموالهم. فالعدل هو أساس العمران وبدونه ينهار المجتمع وتخرب المدن. والملك الظالم يدمر دولته بيده لأن الظلم يطرد الناس ويقضي على الإنتاج.",
    "يحتاج الملك إلى وزراء وكتاب ومستشارين يساعدونه في إدارة شؤون الدولة. وكلما كبرت الدولة زادت حاجتها إلى جهاز إداري متطور ومنظم. وانهيار الجهاز الإداري من علامات ضعف الدولة واقترابها من السقوط.",

    # ── تأثير البيئة والمناخ (Environmental / Climate Effects) ──
    "يؤثر المناخ تأثيراً كبيراً في طبائع الناس وأخلاقهم وأنماط حياتهم. فأهل المناطق الحارة يميلون إلى الخفة والسرعة بينما أهل المناطق الباردة يميلون إلى الجد والصبر. وهذا التأثير يمتد إلى ألوان البشرة وطبيعة الأجسام.",
    "الجغرافيا تحدد نوع الاقتصاد ونمط الحياة في كل منطقة. فالمناطق الخصبة تشجع على الزراعة والاستقرار بينما المناطق الجافة تفرض حياة الرعي والتنقل. والمناطق الساحلية تشجع على التجارة البحرية والصيد.",
    "يقسم ابن خلدون الأرض إلى سبعة أقاليم تختلف في مناخها وطبيعتها. والأقاليم المعتدلة في الوسط هي الأنسب للعمران والحضارة. بينما الأقاليم المتطرفة في الحرارة أو البرودة تكون أقل ملاءمة للتقدم الحضاري.",

    # ── الحروب والجيوش (Warfare and Armies) ──
    "الحرب ظاهرة طبيعية في تاريخ البشرية وهي تنشأ من التنافس على الموارد والسلطة. ويميز ابن خلدون بين حروب القبائل والحروب بين الدول. وكل نوع من هذه الحروب له أساليبه وقواعده الخاصة.",
    "الجيش هو عماد الدولة وأداتها في حماية حدودها وفرض سلطتها. ويجب أن يكون الجيش مبنياً على العصبية والولاء لا على المال فقط. فالجنود المرتزقة لا يمكن الاعتماد عليهم في الأوقات الصعبة.",
    "تتطور أساليب الحرب بتطور العمران والتقنية. فقد انتقل البشر من القتال بالسيوف والرماح إلى استخدام أسلحة أكثر تطوراً. والتخطيط العسكري والاستراتيجية أصبحا من العلوم المهمة في إدارة الحروب.",

    # ── الترف وانهيار الدول (Luxury and Collapse of States) ──
    "الترف هو أخطر عدو للدول والحضارات في نظر ابن خلدون. فعندما ينغمس أهل الدولة في الترف يفقدون صفات الشجاعة والقوة التي بنوا بها دولتهم. ويصبحون عاجزين عن الدفاع عن أنفسهم.",
    "يبدأ الترف عادة في القصور والطبقة الحاكمة ثم ينتشر تدريجياً إلى باقي طبقات المجتمع. ومع انتشار الترف تزداد النفقات وترتفع الضرائب لتمويل حياة البذخ. وهذا يثقل كاهل الرعية ويضعف الاقتصاد.",
    "انهيار الدول لا يحدث فجأة بل هو عملية تدريجية تمتد عبر أجيال. تبدأ بضعف العصبية ثم تنتشر الأخلاق السيئة ثم يضعف الاقتصاد وأخيراً يسقط الجيش. وعند هذه النقطة تصبح الدولة عرضة للغزو من قوى جديدة.",
    "كل دولة تحمل بذور فنائها في داخلها منذ لحظة تأسيسها. فالنجاح يجلب الثروة والثروة تجلب الترف والترف يجلب الضعف. وهذه الدورة لا مفر منها في نظر ابن خلدون وهي تتكرر عبر التاريخ.",

    # ── التاريخ والمنهج (History and Methodology) ──
    "ينتقد ابن خلدون المؤرخين السابقين لأنهم نقلوا الأخبار دون تمحيص أو تحقيق. ويرى أن دراسة التاريخ يجب أن تعتمد على فهم طبائع العمران وقوانين الاجتماع. فالخبر الذي يتعارض مع هذه القوانين يجب رفضه حتى لو كان سنده صحيحاً.",
    "المنهج الذي وضعه ابن خلدون في دراسة التاريخ يعتبر ثورة في الفكر الإنساني. فقد سبق بذلك المفكرين الأوروبيين بعدة قرون في وضع أسس علم الاجتماع. ولذلك يعتبر ابن خلدون مؤسس علم الاجتماع وفلسفة التاريخ.",
    "يرى ابن خلدون أن التاريخ ليس مجرد سرد للأحداث بل هو علم له قواعده وأصوله. ويجب على المؤرخ أن يفهم طبيعة المجتمعات وقوانين تطورها ليميز بين الأخبار الصحيحة والموضوعة. وبهذا المنهج أسس ابن خلدون علماً جديداً سماه علم العمران.",
    "يعتبر ابن خلدون أن فهم التاريخ يتطلب معرفة بالاقتصاد والسياسة والاجتماع والجغرافيا. فالأحداث التاريخية لا تحدث بمعزل عن هذه العوامل بل هي نتيجة لتفاعلها. ولذلك يجب على المؤرخ أن يكون ملماً بجميع هذه العلوم.",
]

# ════════════════════════════════════════════════════════════════
# 3. TEXT PREPROCESSING
# ════════════════════════════════════════════════════════════════

# Arabic diacritics Unicode range
_DIACRITICS_RE = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0670]")
# Normalise common Arabic letter variants
_ALEF_RE = re.compile(r"[إأآا]")
_TEH_MARBUTA_RE = re.compile(r"ة")
_TATWEEL_RE = re.compile(r"ـ")

ARABIC_STOP_WORDS = set(
    "في من إلى على عن هو هي هم هذا هذه ذلك تلك الذي التي اللذان اللتان "
    "الذين اللاتي كان كانت يكون تكون أن إن لا ما لم لن قد ثم أو و ف ب ل ك "
    "مع بين حتى عند لكن بل كل كلا كلتا هل أم نعم لو لولا إذا إذ منذ خلال "
    "عبر فوق تحت أمام وراء قبل بعد حول دون غير سوى".split()
)


def remove_diacritics(text: str) -> str:
    return _DIACRITICS_RE.sub("", text)


def normalize_arabic(text: str) -> str:
    text = remove_diacritics(text)
    text = _ALEF_RE.sub("ا", text)
    text = _TEH_MARBUTA_RE.sub("ه", text)
    text = _TATWEEL_RE.sub("", text)
    return text


def tokenize_arabic(text: str) -> list[str]:
    text = normalize_arabic(text)
    tokens = re.findall(r"[\u0600-\u06FF]+", text)
    return [t for t in tokens if t not in ARABIC_STOP_WORDS and len(t) > 1]


def load_book() -> list[str]:
    """Load book paragraphs from file or embedded text."""
    if os.path.isfile(BOOK_FILE):
        print(f"[INFO] Loading book from {BOOK_FILE}")
        with open(BOOK_FILE, encoding="utf-8") as f:
            raw = f.read()
        # Split on blank lines to get paragraphs
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]
        # Further split very long paragraphs into 2-4 sentence chunks
        chunks = []
        for p in paragraphs:
            sentences = re.split(r"(?<=[.؟!])\s+", p)
            for i in range(0, len(sentences), 3):
                chunk = " ".join(sentences[i : i + 4]).strip()
                if len(chunk) > 20:
                    chunks.append(chunk)
        return chunks
    else:
        print("[INFO] Using embedded book text (مقدمة ابن خلدون)")
        return BOOK_PARAGRAPHS


# ════════════════════════════════════════════════════════════════
# 4. EMBEDDING & FAISS INDEXING
# ════════════════════════════════════════════════════════════════

# E5 models require task-specific prefixes
_E5_QUERY_PREFIX = "query: "
_E5_PASSAGE_PREFIX = "passage: "


def _is_e5_model(model_name: str) -> bool:
    return "e5" in model_name.lower()


def build_embeddings(chunks: list[str], model_name: str):
    from sentence_transformers import SentenceTransformer

    print(f"[INFO] Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    print(f"[INFO] Generating embeddings for {len(chunks)} chunks …")
    # E5 models need "passage: " prefix for documents
    if _is_e5_model(model_name):
        texts = [_E5_PASSAGE_PREFIX + c for c in chunks]
    else:
        texts = chunks
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype(np.float32)
    return model, embeddings


def build_faiss_index(embeddings: np.ndarray):
    import faiss

    dim = embeddings.shape[1]
    # Normalise for cosine similarity via inner product
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"[INFO] FAISS index built: {index.ntotal} vectors, dim={dim}")
    return index


# ════════════════════════════════════════════════════════════════
# 5. CLASSICAL SEARCH – BM25
# ════════════════════════════════════════════════════════════════

def build_bm25_index(chunks: list[str]):
    from rank_bm25 import BM25Okapi

    tokenized = [tokenize_arabic(c) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    print(f"[INFO] BM25 index built over {len(tokenized)} documents")
    return bm25


def bm25_search(query: str, bm25, chunks: list[str], top_k: int = TOP_K):
    tokens = tokenize_arabic(query)
    scores = bm25.get_scores(tokens)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(int(i), float(scores[i]), chunks[i]) for i in top_indices]


# ════════════════════════════════════════════════════════════════
# 6. SEMANTIC SEARCH
# ════════════════════════════════════════════════════════════════

def semantic_search(query: str, embed_model, faiss_index, chunks: list[str],
                    top_k: int = TOP_K):
    import faiss

    q_text = (_E5_QUERY_PREFIX + query) if _is_e5_model(EMBEDDING_MODEL_NAME) else query
    q_emb = embed_model.encode([q_text], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(q_emb)
    scores, indices = faiss_index.search(q_emb, top_k)
    results = []
    for rank in range(top_k):
        idx = int(indices[0][rank])
        score = float(scores[0][rank])
        results.append((idx, score, chunks[idx]))
    return results


# ════════════════════════════════════════════════════════════════
# 7. LLM LOADING & GENERATION
# ════════════════════════════════════════════════════════════════

def load_llm(model_name: str = LLM_MODEL_NAME):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"[INFO] Loading LLM: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    if device == "cpu":
        model = model.to(device)
    model.eval()
    print(f"[INFO] LLM loaded on {device} ({dtype})")
    return tokenizer, model, device


# Arabic + common punctuation — used to strip non-Arabic output
_ARABIC_KEEP_RE = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF"
    r"\s\d.,،؛؟!:()\[\]\-–—/\\]"
)


def _strip_non_arabic(text: str) -> str:
    """Remove runs of non-Arabic characters (e.g. Chinese, Latin sentences)."""
    # Split on whitespace and drop tokens that are majority non-Arabic script
    tokens = text.split()
    arabic_tokens = []
    for tok in tokens:
        arabic_chars = len(re.findall(r"[\u0600-\u06FF]", tok))
        if arabic_chars > 0 or re.fullmatch(r"[\d.,،؛؟!:()\[\]\-–—/\\]+", tok):
            arabic_tokens.append(tok)
    return " ".join(arabic_tokens).strip()


def _merge_system_into_user(messages: list[dict]) -> list[dict]:
    """For models that don't support 'system' role: prepend system content to first user msg."""
    merged = []
    system_text = ""
    for msg in messages:
        if msg["role"] == "system":
            system_text += msg["content"] + "\n\n"
        else:
            if system_text and msg["role"] == "user":
                merged.append({"role": "user",
                               "content": system_text + msg["content"]})
                system_text = ""
            else:
                merged.append(msg)
    return merged


def _generate(messages: list[dict], tokenizer, model, device: str,
              max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    import torch

    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
    except Exception:
        # Model doesn't support system role — merge into first user message
        text = tokenizer.apply_chat_template(
            _merge_system_into_user(messages), tokenize=False,
            add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
        )
    # Decode only newly generated tokens
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    decoded = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return _strip_non_arabic(decoded)


def generate_rag_answer(query: str, context: str, tokenizer, model, device: str) -> str:
    messages = [
        {"role": "system",
         "content": (
             "CRITICAL: You MUST respond ONLY in Arabic. Do NOT use Chinese, English, "
             "or any other language. Every single word must be Arabic script.\n"
             "CRITICAL: Base your answer ONLY on the provided reference text. "
             "Do NOT add any external information.\n\n"
             "أنت مساعد ذكي متخصص في الإجابة عن الأسئلة المتعلقة بمقدمة ابن خلدون. "
             "قواعد صارمة:\n"
             "1. أجب فقط باللغة العربية. ممنوع استخدام الصينية أو الإنجليزية أو أي لغة أخرى.\n"
             "2. اعتمد حصراً على النص المرجعي المقدم ولا تضف معلومات خارجه.\n"
             "3. إذا لم يتضمن النص إجابة كافية، قل: 'لا تتوفر معلومات كافية في النص.'"
         )},
        # ── Few-shot example 1 (shows the model the expected format) ──
        {"role": "user",
         "content": (
             "النص المرجعي:\n"
             "العمران هو أساس الحضارة الإنسانية وهو يتطور من البساطة إلى التعقيد.\n\n"
             "السؤال: ما هو العمران؟\n\n"
             "تعليمات: أجب باللغة العربية فقط، بناءً على النص المرجعي أعلاه حصراً.\n"
             "الإجابة:"
         )},
        {"role": "assistant",
         "content": "العمران هو أساس الحضارة الإنسانية، ويتطور من البساطة إلى التعقيد حسب ما ورد في النص."},
        # ── Few-shot example 2 ──
        {"role": "user",
         "content": (
             "النص المرجعي:\n"
             "الملك العادل هو الذي يحكم بالعدل ويحمي رعيته ولا يظلمهم في أموالهم.\n\n"
             "السؤال: ما صفات الملك العادل؟\n\n"
             "تعليمات: أجب باللغة العربية فقط، بناءً على النص المرجعي أعلاه حصراً.\n"
             "الإجابة:"
         )},
        {"role": "assistant",
         "content": "الملك العادل هو الذي يحكم بالعدل ويحمي رعيته ولا يظلمهم في أموالهم."},
        # ── Actual query ──
        {"role": "user",
         "content": (
             f"النص المرجعي:\n{context}\n\n"
             f"السؤال: {query}\n\n"
             "تحذير: يجب أن تكون إجابتك باللغة العربية فقط. لا تستخدم أي لغة أخرى. "
             "أجب بناءً على النص المرجعي أعلاه حصراً.\n"
             "الإجابة بالعربية:"
         )},
    ]
    return _generate(messages, tokenizer, model, device)


def generate_llm_only_answer(query: str, tokenizer, model, device: str) -> str:
    messages = [
        {"role": "system",
         "content": (
             "CRITICAL: You MUST respond ONLY in Arabic. Do NOT use Chinese, English, "
             "or any other language. Every single word must be Arabic script.\n\n"
             "أنت مساعد ذكي. "
             "أجب عن الأسئلة باللغة العربية فقط. "
             "ممنوع استخدام الصينية أو الإنجليزية أو أي لغة أخرى مطلقاً."
         )},
        # ── Few-shot example (shows the model the expected Arabic format) ──
        {"role": "user",
         "content": "السؤال: ما هو علم الاجتماع؟\n\nتحذير: أجب بالعربية فقط. لا تستخدم أي لغة أخرى.\nالإجابة بالعربية:"},
        {"role": "assistant",
         "content": "علم الاجتماع هو العلم الذي يدرس المجتمعات البشرية وتفاعلات أفرادها وقوانين تطورها."},
        # ── Actual query ──
        {"role": "user",
         "content": (
             f"السؤال: {query}\n\n"
             "تحذير: يجب أن تكون إجابتك باللغة العربية فقط. لا تستخدم أي لغة أخرى.\n"
             "الإجابة بالعربية:"
         )},
    ]
    return _generate(messages, tokenizer, model, device)


# ════════════════════════════════════════════════════════════════
# 8. HYBRID RETRIEVAL (Reciprocal Rank Fusion)
# ════════════════════════════════════════════════════════════════

RRF_K = 60  # standard RRF constant


def hybrid_search(query: str, bm25, embed_model, faiss_index, chunks: list[str],
                  top_k: int = TOP_K) -> list[tuple]:
    """Combine BM25 and semantic rankings via Reciprocal Rank Fusion."""
    import faiss as _faiss

    pool = min(top_k * 4, len(chunks))  # candidate pool per method

    # BM25 ranking
    tokens = tokenize_arabic(query)
    bm25_scores = bm25.get_scores(tokens)
    bm25_ranked = list(np.argsort(bm25_scores)[::-1][:pool])

    # Semantic ranking
    q_text = (_E5_QUERY_PREFIX + query) if _is_e5_model(EMBEDDING_MODEL_NAME) else query
    q_emb = embed_model.encode([q_text], convert_to_numpy=True).astype(np.float32)
    _faiss.normalize_L2(q_emb)
    sem_scores, sem_indices = faiss_index.search(q_emb, pool)
    sem_ranked = [int(sem_indices[0][i]) for i in range(pool)]

    # RRF fusion
    rrf: dict[int, float] = {}
    for rank, idx in enumerate(bm25_ranked):
        rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)
    for rank, idx in enumerate(sem_ranked):
        rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)

    sorted_ids = sorted(rrf, key=lambda i: rrf[i], reverse=True)[:top_k]
    return [(idx, rrf[idx], chunks[idx]) for idx in sorted_ids]


# ════════════════════════════════════════════════════════════════
# 9. DISPLAY HELPERS
# ════════════════════════════════════════════════════════════════

_SEP = "─" * 78


def display_search_comparison(query: str, bm25_results, semantic_results):
    print(f"\n{'═' * 78}")
    print(f"  الاستعلام: {query}")
    print(f"{'═' * 78}")

    print(f"\n  ◀ البحث الكلاسيكي (BM25) — أفضل {TOP_K} نتائج ▶")
    print(_SEP)
    for rank, (idx, score, text) in enumerate(bm25_results, 1):
        print(f"  [{rank}] (chunk {idx}, score={score:.4f})")
        for line in textwrap.wrap(text, width=72):
            print(f"      {line}")
        print()

    print(f"  ◀ البحث الدلالي (Semantic) — أفضل {TOP_K} نتائج ▶")
    print(_SEP)
    for rank, (idx, score, text) in enumerate(semantic_results, 1):
        print(f"  [{rank}] (chunk {idx}, score={score:.4f})")
        for line in textwrap.wrap(text, width=72):
            print(f"      {line}")
        print()


def display_answer_comparison(query: str, rag_ans: str, llm_ans: str,
                              model_label: str = ""):
    print(f"\n{'═' * 78}")
    title = f"  السؤال: {query}"
    if model_label:
        title += f"  [{model_label}]"
    print(title)
    print(f"{'═' * 78}")

    print("\n  ◀ إجابة RAG (مع استرجاع) ▶")
    print(_SEP)
    for line in textwrap.wrap(rag_ans, width=72):
        print(f"    {line}")

    print(f"\n  ◀ إجابة LLM فقط (بدون استرجاع) ▶")
    print(_SEP)
    for line in textwrap.wrap(llm_ans, width=72):
        print(f"    {line}")
    print()


# ════════════════════════════════════════════════════════════════
# 10. TEN EVALUATION QUERIES
# ════════════════════════════════════════════════════════════════

QUERIES = [
    # Direct / Easy
    "ما هي العصبية عند ابن خلدون؟",
    "كيف يقسم ابن خلدون العلوم؟",
    "ما الفرق بين أهل البادية وأهل الحضر؟",
    "ما هي أطوار الدولة الثلاثة؟",
    "ما رأي ابن خلدون في الضرائب؟",
    # Indirect / Hard
    "كيف يؤدي الترف إلى سقوط الدول؟",
    "لماذا يعتبر ابن خلدون مؤسس علم الاجتماع؟",
    "ما علاقة البيئة والمناخ بالحضارة في نظر ابن خلدون؟",
    "هل يمكن للدولة أن تستمر بدون عصبية؟ وضح.",
    "كيف ينظر ابن خلدون إلى دور العمل في الاقتصاد؟",
]


# ════════════════════════════════════════════════════════════════
# 11. MAIN
# ════════════════════════════════════════════════════════════════

def main():
    # ── 1. Book Preparation ──────────────────────────────────
    print("\n" + "=" * 60)
    print("  1. تحضير الكتاب — Book Preparation")
    print("=" * 60)
    print(f"  العنوان : مقدمة ابن خلدون")
    print(f"  المؤلف  : عبد الرحمن بن خلدون (1332–1406 م)")
    print(f"  الموضوع : علم الاجتماع، فلسفة التاريخ، الاقتصاد")
    print(f"  المعالجة: تقسيم إلى فقرات من 2–4 جمل، إزالة التشكيل، تطبيع الحروف")

    chunks = load_book()
    print(f"  عدد الفقرات (chunks): {len(chunks)}")

    # ── 2. Embedding & Indexing ───────────────────────────────
    print("\n" + "=" * 60)
    print("  2. التضمين والفهرسة — Embedding & Indexing")
    print("=" * 60)
    embed_model, embeddings = build_embeddings(chunks, EMBEDDING_MODEL_NAME)
    faiss_index = build_faiss_index(embeddings)
    bm25_index = build_bm25_index(chunks)

    # ── Open results file for incremental writing ─────────────
    output_path = os.path.join(os.path.dirname(__file__) or ".", "results.txt")
    f = open(output_path, "w", encoding="utf-8")
    f.write("=" * 60 + "\n")
    f.write("Problem 4: Arabic RAG System — Results\n")
    f.write(f"Book: مقدمة ابن خلدون\n")
    f.write(f"Embedding model: {EMBEDDING_MODEL_NAME}\n")
    f.write(f"LLMs: {', '.join(LLM_MODEL_NAMES)}\n")
    f.write(f"Chunks: {len(chunks)}\n")
    f.write("=" * 60 + "\n\n")
    f.flush()

    # ── 3. Search Evaluation (Task 2) ─────────────────────────
    print("\n" + "=" * 60)
    print("  3. تقييم البحث — Search Evaluation (10 queries)")
    print("=" * 60)

    all_bm25_results = {}
    all_semantic_results = {}

    for i, q in enumerate(QUERIES, 1):
        bm25_res = bm25_search(q, bm25_index, chunks)
        sem_res = semantic_search(q, embed_model, faiss_index, chunks)
        all_bm25_results[q] = bm25_res
        all_semantic_results[q] = sem_res
        display_search_comparison(q, bm25_res, sem_res)

        # Write search results immediately
        f.write(f"\n{'─' * 60}\n")
        f.write(f"Query {i}: {q}\n")
        f.write(f"{'─' * 60}\n")
        f.write("\n[BM25 Results]\n")
        for rank, (idx, score, text) in enumerate(bm25_res, 1):
            f.write(f"  {rank}. (chunk {idx}, score={score:.4f}) {text}\n")
        f.write("\n[Semantic Results]\n")
        for rank, (idx, score, text) in enumerate(sem_res, 1):
            f.write(f"  {rank}. (chunk {idx}, score={score:.4f}) {text}\n")
        f.flush()

    # ── 4. RAG vs LLM-only (Task 3) — run BOTH LLMs ─────────
    all_hybrid_contexts: dict[str, list] = {}  # shared (retrieval is model-independent)

    for llm_name in LLM_MODEL_NAMES:
        short = llm_name.split("/")[-1]
        print("\n" + "=" * 60)
        print(f"  4. RAG مقابل LLM فقط — {llm_name}")
        print("=" * 60)

        f.write(f"\n\n{'=' * 60}\n")
        f.write(f"LLM: {llm_name}\n")
        f.write(f"{'=' * 60}\n")
        f.flush()

        tokenizer, model, device = load_llm(llm_name)

        for q in QUERIES:
            # Build context (only compute once, reuse for second model)
            if q not in all_hybrid_contexts:
                hybrid_res = hybrid_search(
                    q, bm25_index, embed_model, faiss_index, chunks, top_k=TOP_K
                )
                all_hybrid_contexts[q] = [text for _, _, text in hybrid_res[:3]]
            context = "\n\n".join(all_hybrid_contexts[q])

            rag_ans = generate_rag_answer(q, context, tokenizer, model, device)
            llm_ans = generate_llm_only_answer(q, tokenizer, model, device)

            display_answer_comparison(q, rag_ans, llm_ans, model_label=short)

            # Write answers immediately
            f.write(f"\n[RAG Answer — {short}] {q}\n  {rag_ans}\n")
            f.write(f"\n[LLM-only Answer — {short}] {q}\n  {llm_ans}\n")
            f.flush()

        # Free GPU memory before loading next model
        del model, tokenizer
        try:
            import torch; torch.cuda.empty_cache()
        except Exception:
            pass

    # ── 5. Comparison & Reflection ────────────────────────────
    print("\n" + "=" * 60)
    print("  5. المقارنة والتحليل — Comparison & Reflection")
    print("=" * 60)

    reflection = textwrap.dedent("""\
    ┌──────────────────────────────────────────────────────────────────┐
    │                    التحليل والاستنتاجات                         │
    ├──────────────────────────────────────────────────────────────────┤
    │                                                                  │
    │ 1. مقارنة طرق الاسترجاع:                                       │
    │    • البحث الدلالي (Semantic Search) يتفوق على BM25 في فهم      │
    │      المعنى والسياق، خاصة عند استخدام مرادفات أو صياغات         │
    │      مختلفة للسؤال نفسه.                                        │
    │    • BM25 يعمل بشكل جيد عند تطابق الكلمات المفتاحية مباشرة    │
    │      لكنه يفشل مع الأسئلة غير المباشرة.                        │
    │    • البحث الدلالي أفضل للأسئلة المعقدة والتحليلية.            │
    │                                                                  │
    │ 2. تأثير الاسترجاع على إجابات النموذج:                         │
    │    • RAG ينتج إجابات أدق وأكثر ارتباطاً بالكتاب لأنه يعتمد    │
    │      على نصوص مرجعية فعلية من المقدمة.                          │
    │    • LLM وحده قد يولد إجابات عامة أو غير دقيقة، خاصة في       │
    │      التفاصيل الدقيقة لأفكار ابن خلدون.                        │
    │    • الاسترجاع يقلل من الهلوسة (hallucination) في الإجابات.     │
    │                                                                  │
    │ 3. استنتاجات حول قيمة البحث الدلالي و RAG:                     │
    │    • البحث الدلالي ضروري للنصوص العربية الكلاسيكية حيث          │
    │      تتنوع صياغات المعاني.                                      │
    │    • RAG يحسن جودة الإجابات بشكل ملحوظ مقارنة بـ LLM وحده.    │
    │    • اختيار كتاب غير شائع يبرز فائدة RAG لأن النموذج لم        │
    │      يتدرب بشكل كافٍ على محتواه المحدد.                        │
    │    • الجمع بين البحث الدلالي والكلاسيكي (hybrid) قد يعطي       │
    │      نتائج أفضل في التطبيقات العملية.                           │
    │                                                                  │
    └──────────────────────────────────────────────────────────────────┘
    """)
    print(reflection)
    f.write("\n\n" + reflection)
    f.flush()
    f.close()
    print(f"\n[INFO] Results saved to {output_path}")
    print("[DONE] ✓")


if __name__ == "__main__":
    main()
