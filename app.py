import os
import re
import base64
from pathlib import Path
from collections import defaultdict

import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# =========================================================
# 1. CONFIGURATION
# =========================================================
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
INDEX_DIR = BASE_DIR / "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MAX_IMAGE_BYTES = 10 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}

st.set_page_config(
    page_title="Uzhavar Nanban | Agriculture Portal",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# APP STYLING
# =========================================================
st.markdown("""
<style>
    .stApp {
        background-color: #F5F8F1;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    h1, h2, h3 {
        color: #174D32;
    }

    [data-testid="stSidebar"] {
        background-color: #EAF2E6;
    }

    .stButton > button {
        background-color: #176B45;
        color: white;
        border-radius: 10px;
        border: none;
        font-weight: 600;
    }

    .stButton > button:hover {
        background-color: #0F5132;
        color: white;
    }

    [data-testid="stChatMessage"] {
        background-color: white;
        border: 1px solid #DCE7D8;
        border-radius: 12px;
        padding: 12px;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# 2. UI
# =========================================================
st.title("🌾 Uzhavar Nanban")
st.subheader("Tamil Nadu Agriculture Information Portal")
st.write(
    "Explore Tamil Nadu agriculture schemes, farmer benefits, and answers "
    "supported by government source documents. You can ask questions in English or Tamil."
)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Knowledge Base", "Agriculture PDFs")
with col2:
    st.metric("Answer Sources", "Cited documents")
with col3:
    st.metric("Crop Support", "Image analysis")

st.caption(
    "Scheme answers are based on retrieved documents. Verify important details with "
    "official sources. Crop-image observations are preliminary, not a confirmed diagnosis."
)

with st.expander("🌐 Official Government Portals", expanded=False):
    link1, link2, link3 = st.columns(3)
    with link1:
        st.markdown("### 🌾 Agriculture")
        st.markdown("[Tamil Nadu Agriculture Department](https://www.tnagrisnet.tn.gov.in/home/index/en)")
    with link2:
        st.markdown("### 🌱 Schemes")
        st.markdown("[Agriculture Schemes and Subsidies](https://www.tnagrisnet.tn.gov.in/home/schemes/tm)")
    with link3:
        st.markdown("### 📝 Online Services")
        st.markdown("[Agri Online e-Sevai](https://www.tnagrisnet.tn.gov.in/esevai/)")

# =========================================================
# 3. FAISS
# =========================================================
@st.cache_resource
def load_vector_store():
    if not INDEX_DIR.exists():
        raise FileNotFoundError(f"FAISS index not found: {INDEX_DIR}. Run ingest.py first.")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.load_local(
        str(INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )

# =========================================================
# 4. NORMALIZATION AND QUERY EXPANSION
# =========================================================
def normalize_text(text):
    """Normalize English text for lexical matching."""
    text = (text or "").lower()
    text = re.sub(r"[–—−]", "-", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text):
    return normalize_text(text).split()


STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "what", "which",
    "who", "when", "where", "why", "how", "much", "many", "for",
    "of", "in", "on", "to", "from", "by", "with", "and", "or",
    "during", "under", "tell", "me", "please", "about", "give",
    "does", "do", "did", "can", "could", "would", "should", "it",
    "its", "this", "that", "i", "my", "we", "our", "you", "your",
    "there", "any", "some", "related", "information", "available",
    "mention", "only", "explicitly", "stated", "period", "provide",
}


def keyword_score(question, passage):
    """Lexical relevance score with phrase matching."""
    q_norm = normalize_text(question)
    p_norm = normalize_text(passage)
    q_words = {
        word for word in tokenize(question)
        if word not in STOP_WORDS and len(word) > 1
    }
    p_words = set(tokenize(passage))
    if not q_words:
        return 0.0

    overlap = len(q_words.intersection(p_words)) / len(q_words)
    phrase_bonus = 0.0
    phrases = (
        "rice bloom", "paddy", "rice", "sugarcane booster", "cotton plus",
        "soil fertility mission", "pradhan mantri fasal bima yojana",
        "pmfby", "soil health card", "threshing floor",
        "cotton renaissance mission", "palmyra protection mission",
    )
    for phrase in phrases:
        if phrase in q_norm and phrase in p_norm:
            phrase_bonus += 0.5
    return overlap + phrase_bonus


def detect_crop(text):
    text = normalize_text(text)
    crop_terms = [
        ("rice", ("rice", "paddy", "paddies")),
        ("sugarcane", ("sugarcane", "sugar cane")),
        ("cotton", ("cotton",)),
        ("banana", ("banana", "plantain")),
        ("coconut", ("coconut",)),
        ("mango", ("mango",)),
        ("guava", ("guava",)),
        ("tomato", ("tomato",)),
        ("moringa", ("moringa",)),
        ("small onion", ("small onion",)),
    ]
    for crop, aliases in crop_terms:
        if any(alias in text for alias in aliases):
            return crop
    return None


def crop_scheme_search_query(question, image_observation=None):
    """Expand crop-support questions toward relevant terms in the source PDF."""
    combined = normalize_text(f"{question} {image_observation or ''}")
    benefit_intent = any(
        term in combined
        for term in (
            "benefit", "scheme", "subsidy", "support", "incentive",
            "available", "help", "government", "farmer", "cultivation",
            "insurance", "allocation",
        )
    )
    if not benefit_intent:
        return None

    crop = detect_crop(combined)
    queries = {
        "rice": (
            "Tamil Nadu Agriculture Budget 2026-27 rice paddy farmer benefits. "
            "Rice Bloom for paddy flowering. Pradhan Mantri Fasal Bima Yojana "
            "PMFBY notified crops seasons farmers insurance. Soil Health Cards "
            "soil fertility mission. Find exact scheme names, allocations, "
            "eligibility and benefits explicitly stated in the document."
        ),
        "sugarcane": (
            "Sugarcane Booster Tamil Nadu sugarcane farmer schemes subsidies "
            "explicit benefits allocations eligibility Agriculture Budget 2026-27"
        ),
        "cotton": (
            "Cotton Plus Cotton Renaissance Mission Tamil Nadu cotton farmer "
            "schemes benefits allocations eligibility 2026-27"
        ),
        "banana": "Tamil Nadu banana plantain farmers schemes subsidies benefits",
        "coconut": "Tamil Nadu coconut farmers schemes subsidies benefits",
        "mango": "Tamil Nadu mango farmers schemes subsidies benefits",
        "guava": "Tamil Nadu guava farmers schemes subsidies benefits",
        "tomato": "Tamil Nadu tomato farmers schemes subsidies benefits",
        "moringa": "Tamil Nadu moringa farmers schemes subsidies benefits",
        "small onion": "Tamil Nadu small onion farmers schemes subsidies benefits",
    }
    return queries.get(crop)

# =========================================================
# 5. HYBRID RETRIEVAL
# =========================================================
def doc_key(doc):
    return (
        str(doc.metadata.get("source", "")),
        str(doc.metadata.get("page", -1)),
        re.sub(r"\s+", " ", doc.page_content).strip(),
    )


def retrieve_documents(db, question, semantic_k=12, final_k=5, extra_queries=None):
    """Combine vector search with lexical matching and diversify by source page."""
    queries = [question]
    for query in extra_queries or []:
        if query and query.strip() and query.strip() not in queries:
            queries.append(query.strip())

    candidate_map = {}
    semantic_rank = {}
    lexical_max = defaultdict(float)

    for query in queries:
        try:
            semantic_docs = db.similarity_search(query, k=semantic_k)
        except Exception as exc:
            st.warning(f"A semantic search query failed: {exc}")
            semantic_docs = []

        for rank, doc in enumerate(semantic_docs):
            key = doc_key(doc)
            candidate_map[key] = doc
            semantic_rank[key] = max(semantic_rank.get(key, 0.0), 1.0 / (rank + 1))
            lexical_max[key] = max(lexical_max[key], keyword_score(question, doc.page_content))

    # Scan the local FAISS docstore for exact keyword matches.
    try:
        all_docs = list(db.docstore._dict.values())
    except Exception:
        all_docs = []

    for doc in all_docs:
        best_lexical = max(
            (keyword_score(query, doc.page_content) for query in queries),
            default=0.0,
        )
        if best_lexical > 0:
            key = doc_key(doc)
            candidate_map[key] = doc
            lexical_max[key] = max(lexical_max[key], best_lexical)

    detected_crop = detect_crop(question)
    original_question = normalize_text(question)

    def final_score(item):
        key, doc = item
        content = normalize_text(doc.page_content)
        score = lexical_max.get(key, 0.0) + 0.25 * semantic_rank.get(key, 0.0)
        for phrase in (
            "rice bloom", "sugarcane booster", "cotton plus",
            "soil fertility mission", "pmfby",
            "pradhan mantri fasal bima yojana", "threshing floor",
        ):
            if phrase in original_question and phrase in content:
                score += 1.5
        if detected_crop and detected_crop in content:
            score += 0.1
        return score

    ranked = sorted(candidate_map.items(), key=final_score, reverse=True)
    selected, selected_keys, selected_pages = [], set(), set()

    # Prefer distinct source pages.
    for key, doc in ranked:
        page_key = (
            str(doc.metadata.get("source", "")),
            str(doc.metadata.get("page", -1)),
        )
        if page_key in selected_pages:
            continue
        selected.append(doc)
        selected_keys.add(key)
        selected_pages.add(page_key)
        if len(selected) >= final_k:
            break

    if len(selected) < final_k:
        for key, doc in ranked:
            if key not in selected_keys:
                selected.append(doc)
                selected_keys.add(key)
                if len(selected) >= final_k:
                    break
    return selected

# =========================================================
# 6. SOURCE CONTEXT AND CITATIONS
# =========================================================
def display_source_name(source, metadata=None):
    metadata = metadata or {}
    title = metadata.get("title")
    if title:
        return str(title)
    return Path(str(source)).name if source else "Unknown source"


def build_context(documents):
    context_parts, source_details = [], []
    for number, doc in enumerate(documents, start=1):
        metadata = doc.metadata or {}
        source = metadata.get("source", "Unknown source")
        source_type = metadata.get("source_type", "unknown")
        document_name = display_source_name(source, metadata)
        raw_page = metadata.get("page")
        page_number = "Not applicable"

        # Ingestion commonly stores zero-based page indices; display one-based PDF pages.
        if source_type in {"pdf", "pdf_image_ocr"}:
            try:
                page_number = int(raw_page) + 1 if raw_page is not None else "Unknown"
            except (TypeError, ValueError):
                page_number = "Unknown"

        context_parts.append(
            f"[SOURCE {number}]\n"
            f"Document: {document_name}\n"
            f"Source location: {source}\n"
            f"Source type: {source_type}\n"
            f"PDF page: {page_number}\n"
            f"Content:\n{doc.page_content}"
        )
        source_details.append({
            "number": number,
            "source": source,
            "display_name": document_name,
            "page": page_number,
            "source_type": source_type,
            "content": doc.page_content,
        })
    return "\n\n".join(context_parts), source_details

# =========================================================
# 7. IMAGE ANALYSIS AND TAMIL QUERY TRANSLATION
# =========================================================
def analyze_crop_image(uploaded_file, model_name):
    """Describe visible crop features without making a definitive diagnosis."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY was not found. Check your .env file.")

    image_bytes = uploaded_file.getvalue()
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError("Please upload an image smaller than 10 MB.")
    mime_type = uploaded_file.type
    if mime_type not in ALLOWED_IMAGE_TYPES:
        raise ValueError("Please upload a JPG, PNG, or WEBP image.")

    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    vision_llm = ChatOpenAI(model=model_name, temperature=0, api_key=api_key)
    message = HumanMessage(content=[
        {
            "type": "text",
            "text": (
                "You are assisting a farmer. State the likely crop only if identifiable; "
                "otherwise say you cannot identify it confidently. Describe only visible "
                "features such as spots, discoloration, damaged areas, or insects if clearly "
                "visible. Do not invent symptoms or give a definite disease diagnosis. "
                "Keep it brief and use plain language."
            ),
        },
        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_base64}"}},
    ])
    response = vision_llm.invoke([message])
    return response.content if isinstance(response.content, str) else str(response.content)


def is_tamil_text(text):
    return bool(re.search(r"[\u0B80-\u0BFF]", text or ""))


def make_english_retrieval_query(question, model_name):
    """Translate Tamil into English for searching English PDFs; do not answer."""
    if not is_tamil_text(question):
        return None
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        llm = ChatOpenAI(model=model_name, temperature=0, api_key=api_key)
        response = llm.invoke([
            SystemMessage(content=(
                "Translate the user's Tamil agriculture question into a concise English "
                "search query. Preserve scheme names, crop names, years, amounts, and intent. "
                "Output only the English query; do not answer or add facts."
            )),
            HumanMessage(content=question),
        ])
        query = response.content if isinstance(response.content, str) else str(response.content)
        return query.strip() or None
    except Exception as exc:
        st.warning(f"Tamil query translation failed; trying the original question. Details: {exc}")
        return None

# =========================================================
# 8. GROUNDED ANSWER GENERATION
# =========================================================
def generate_answer(question, context, model_name, image_observation=None):
    """
    Generate a source-grounded answer in the same language as the user's question.
    A language check triggers one repair attempt if the model uses the wrong script.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY was not found. Add it to your .env file.")

    llm = ChatOpenAI(model=model_name, temperature=0, api_key=api_key)

    # Use the script of the latest question to choose the answer language.
    question_is_tamil = is_tamil_text(question)
    if question_is_tamil:
        language_rule = """
STRICT OUTPUT LANGUAGE:
- The latest user question is in Tamil. Write the complete answer in natural Tamil.
- Use Tamil script for explanations and all field labels.
- Do not switch to English because the retrieved PDF passages are in English.
- Keep official scheme names, document titles, acronyms, amounts, page numbers, and [SOURCE N]
  unchanged when appropriate.
"""
        answer_format = """
For each financial figure, use these Tamil labels, one per line:
**புள்ளிவிவரம்:** [what the amount refers to]
**தொகை:** [amount]
**காலம்:** [period, only if supported]
**ஆவணம்:** [document title]
**PDF பக்க எண்:** [actual PDF page]
**ஆதாரம்:** [SOURCE N]
"""
    else:
        language_rule = """
STRICT OUTPUT LANGUAGE:
- The latest user question is in English. The entire answer MUST be in English.
- Do not output Tamil characters or Tamil words anywhere in the answer.
- The retrieved passages may be in Tamil; translate their meaning into clear English.
- Do not copy Tamil sentences or headings from the retrieved passages.
- Use English for every explanation and every field label.
"""
        answer_format = """
For each financial figure, use these exact English labels, one per line:
**Figure:** [what the amount refers to]
**Amount:** [amount and unit]
**Period:** [period, only if supported]
**Document:** [document title]
**Actual PDF page:** [actual PDF page number]
**Citation:** [SOURCE N]
The Figure field must describe the purpose, not repeat the amount.
The Amount field must contain the monetary amount, not a description.
Omit Period if the supplied passages do not state it.
"""

    system_prompt = f"""
You are Uzhavar Nanban, an AI assistant for Tamil Nadu farmers.

{language_rule}

SOURCE GROUNDING:
- Use only the supplied retrieved passages for government scheme facts.
- Never invent scheme names, benefits, eligibility, amounts, dates, or citations.
- Every scheme-specific factual claim must have an immediate [SOURCE N] citation.
- If the passages do not confirm a fact, say it could not be verified from the available document.
- Do not infer missing text from a passage that ends mid-sentence or mid-heading.
- Do not transfer an amount from one scheme or topic to another.
- Distinguish annual allocations from multi-year outlays and do not combine them.
- Do not call a general budget allocation a subsidy or individual farmer payment.
- Do not claim eligibility unless the source confirms it.
- Image observations are preliminary and must never be evidence for scheme facts.
- Do not describe a scheme as exclusively for paddy farmers merely because paddy may be covered.
- The ₹4,000-per-acre amount on PDF page 14 is described in the context of extending organic
  farming. Do not call it a paddy-specific subsidy unless a retrieved source explicitly confirms it.
- The ₹648.55 crore amount on PDF page 7 relates to Pradhan Mantri Fasal Bima Yojana (PMFBY).
  Describe it as a scheme allocation, not as direct payment or guaranteed benefit to each farmer.
- For every financial figure, state what it represents, its period if stated, the document title,
  actual PDF viewer page number, and matching [SOURCE N], only when supported by context.
- Use the actual PDF viewer page number supplied in the context; do not guess page metadata.
- If sources disagree, explain the discrepancy and cite both sources.

ANSWER FORMAT:
- Answer directly in simple language and include only relevant details.
- For multiple figures, use a numbered list.
{answer_format}
- Omit any field not supported by the supplied context. Never invent missing details.

RETRIEVED SOURCE PASSAGES:
{context}
"""

    user_prompt = (
        f"Farmer's question:\n{question}\n\n"
        f"Answer in {'Tamil' if question_is_tamil else 'English'} only. "
        "Use only the supplied source passages. Check that each figure is supported, "
        "each citation matches its source, the title and actual PDF page are supported, "
        "and no unrelated schemes or unsupported claims are included."
    )
    if image_observation:
        user_prompt += (
            "\n\nPreliminary image observation (not a confirmed diagnosis):\n"
            f"{image_observation}\n"
            "Use this only to understand the possible crop. Use source passages only for "
            "government scheme facts."
        )

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])
    answer = response.content if isinstance(response.content, str) else str(response.content)

    # Validate the output script. If the model ignores the language instruction,
    # make one correction call while keeping the original retrieved evidence.
    answer_is_tamil = is_tamil_text(answer)
    wrong_language = (not question_is_tamil and answer_is_tamil) or (
        question_is_tamil and not answer_is_tamil
    )
    if wrong_language:
        target_language = "Tamil" if question_is_tamil else "English"
        repair_prompt = (
            f"Rewrite the answer below entirely in {target_language}. "
            f"{'Use Tamil script and Tamil field labels.' if question_is_tamil else 'Use English words and English field labels only; do not include any Tamil characters.'} "
            "Preserve all supported facts, amounts, page numbers, and [SOURCE N] citations. "
            "Do not add facts or change the meaning. Follow the required answer format from the system instructions.\n\n"
            f"Answer to correct:\n{answer}"
        )
        repaired = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=repair_prompt),
        ])
        repaired_answer = repaired.content if isinstance(repaired.content, str) else str(repaired.content)
        repaired_is_tamil = is_tamil_text(repaired_answer)
        repaired_wrong_language = (not question_is_tamil and repaired_is_tamil) or (
            question_is_tamil and not repaired_is_tamil
        )
        if not repaired_wrong_language:
            answer = repaired_answer
        else:
            # Last-resort safe fallback: avoid showing an answer in the wrong script.
            if question_is_tamil:
                answer = "மன்னிக்கவும், இந்தக் கேள்விக்கான பதிலைத் தமிழில் உருவாக்க முடியவில்லை. தயவுசெய்து மீண்டும் முயற்சிக்கவும்."
            else:
                answer = (
                    "Sorry, I could not generate the answer in English correctly. "
                    "Please try asking the question again."
                )

    return answer

# =========================================================
# 9. SIDEBAR SETTINGS
# =========================================================
with st.sidebar:
    st.markdown("## 🌾 Uzhavar Nanban")
    st.caption("Your agriculture knowledge assistant")
    st.divider()
    st.markdown("### ⚙️ Model Settings")
    model_name = st.text_input(
        "OpenAI model",
        value=DEFAULT_MODEL,
        help="Uses OPENAI_MODEL from your .env file by default. Choose a vision-capable model for image analysis.",
    )
    show_debug = st.checkbox("Show retrieved passages", value=False)
    st.divider()
    st.markdown("### 🔎 Retrieval Configuration")
    st.write("Semantic candidates per query: **12**")
    st.write("Keyword matching: **Full local index**")
    st.write("Final context passages: **5**")
    st.caption("Semantic retrieval and keyword matching are combined to rank relevant passages.")
    st.divider()
    st.markdown("### 🌱 How to use")
    st.markdown(
        "1. Select your question mode.\n"
        "2. Type or select a suggested question.\n"
        "3. Read the generated answer.\n"
        "4. Open the source cards.\n"
        "5. Verify important details in official documents."
    )

# =========================================================
# 10. LOAD INDEX
# =========================================================
try:
    db = load_vector_store()
except Exception as error:
    st.error(f"Could not load the FAISS index: {error}")
    st.info("Check that ingest.py completed successfully and faiss_index exists.")
    st.stop()

# =========================================================
# 11. CHAT HISTORY
# =========================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


def set_suggested_question(question_text):
    st.session_state.pending_question = question_text


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# =========================================================
# 12. SUGGESTED QUESTIONS
# =========================================================
st.header("💬 Ask Your Agriculture Question")
st.write(
    "Choose a sample question or type your own question in English or Tamil. "
    "Tamil questions are translated for searching English source documents, while "
    "the original question is used to generate the answer in Tamil."
)

suggestions = [
    "How much funding is allocated to the Soil Fertility Mission for 2026–27?",
    "What government schemes are available for paddy farmers?",
    "What agriculture schemes are available to farmers in Tamil Nadu?",
]
suggestion_cols = st.columns(3)
for index, suggestion in enumerate(suggestions):
    with suggestion_cols[index]:
        st.button(
            suggestion,
            key=f"suggested_question_{index}",
            use_container_width=True,
            on_click=set_suggested_question,
            args=(suggestion,),
        )

# =========================================================
# 13. QUESTION MODE AND IMAGE UPLOAD
# =========================================================
question_mode = st.radio(
    "Choose question mode",
    options=["Text question", "Ask about a crop image"],
    horizontal=True,
    index=0,
    key="question_mode",
)

uploaded_image = None
if question_mode == "Ask about a crop image":
    uploaded_image = st.file_uploader(
        "🌱 Upload a crop or leaf image",
        type=["jpg", "jpeg", "png", "webp"],
        key="crop_image",
        help="Supported formats: JPG, PNG, WEBP. Maximum size: 10 MB.",
    )
    if uploaded_image is not None:
        st.image(uploaded_image, caption="Image selected for crop analysis", width=350)
    else:
        st.info("Please upload a crop image before asking your question.")
else:
    st.caption("Text question mode: no crop image will be used.")

# =========================================================
# 14. QUESTION INPUT
# =========================================================
question = st.chat_input(
    "Ask in English or Tamil about schemes, subsidies, funding, or farmer benefits..."
)
pending_question = st.session_state.pending_question
if pending_question:
    st.session_state.pending_question = None
if not question:
    question = pending_question

# =========================================================
# 15. PROCESS QUESTION
# =========================================================
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
        if question_mode == "Ask about a crop image" and uploaded_image is not None:
            st.caption("🌱 Crop image attached for this question")

    with st.chat_message("assistant"):
        try:
            is_image_question = question_mode == "Ask about a crop image" and uploaded_image is not None
            with st.spinner(
                "Analyzing crop image and searching documents..."
                if is_image_question else "Searching agriculture documents..."
            ):
                image_observation = None
                if is_image_question:
                    image_observation = analyze_crop_image(uploaded_image, model_name)

                english_retrieval_query = make_english_retrieval_query(question, model_name)
                search_text = " ".join(
                    part for part in [question, english_retrieval_query or ""] if part
                )
                extra_queries = []
                if english_retrieval_query:
                    extra_queries.append(english_retrieval_query)
                if image_observation:
                    extra_queries.append(
                        f"{search_text}. Crop image observation: {image_observation}"
                    )

                crop = detect_crop(search_text)
                scheme_query = crop_scheme_search_query(search_text, image_observation)
                if scheme_query:
                    extra_queries.append(scheme_query)

                if crop == "rice":
                    extra_queries.append(
                        "Tamil Nadu Agriculture Budget 2026-2027 paddy rice farmer schemes "
                        "benefits and subsidies. Rice Bloom flowering booster. Pradhan Mantri "
                        "Fasal Bima Yojana PMFBY crop insurance. Soil Health Cards. "
                        "Find only details explicitly stated in the source."
                    )

                # Use English translation for lexical retrieval when available.
                retrieval_query = english_retrieval_query or question
                documents = retrieve_documents(
                    db,
                    retrieval_query,
                    semantic_k=12,
                    final_k=5,
                    extra_queries=extra_queries,
                )

                if not documents:
                    if is_tamil_text(question):
                        answer = (
                            "குறியிடப்பட்ட வேளாண்மை ஆவணங்களில் தொடர்புடைய தகவலைக் கண்டறிய முடியவில்லை. "
                            "மேலும் குறிப்பான கேள்வியைக் கேட்கவும் அல்லது அதிகாரப்பூர்வ அரசு "
                            "ஆதாரத்தில் விவரங்களைச் சரிபார்க்கவும்."
                        )
                    else:
                        answer = (
                            "I could not find relevant information in the indexed agriculture "
                            "documents. Please try a more specific question or verify the details "
                            "using an official government source."
                        )
                    source_details = []
                else:
                    context, source_details = build_context(documents)
                    answer = generate_answer(
                        question=question,
                        context=context,
                        model_name=model_name,
                        image_observation=image_observation,
                    )

            if image_observation:
                with st.expander(
                    "🌱 படத்தில் காணப்படும் அம்சங்கள் — நோயறிதல் அல்ல"
                    if is_tamil_text(question)
                    else "🌱 Image observations — not a diagnosis",
                    expanded=True,
                ):
                    st.write(image_observation)

            st.markdown(answer)

            if source_details:
                with st.expander(
                    "📚 இந்தப் பதிலுக்குப் பயன்படுத்திய ஆதாரங்கள்"
                    if is_tamil_text(question)
                    else "📚 Sources used for this answer",
                    expanded=True,
                ):
                    if is_tamil_text(question):
                        st.caption(
                            "பதிலுக்குப் பயன்படுத்தப்பட்ட ஆதாரப் பகுதிகள். முக்கியமான தகவல்களை "
                            "உறுதிப்படுத்த அசல் ஆவணத்தைத் திறந்து சரிபார்க்கவும்."
                        )
                    else:
                        st.caption(
                            "Retrieved passages supporting the answer. Open the original "
                            "document to verify important claims."
                        )

                    for source in source_details:
                        with st.container(border=True):
                            st.caption(f"SOURCE {source['number']}")
                            st.write(f"📄 **{source['display_name']}**")
                            if is_tamil_text(question):
                                st.write(f"வகை: {source.get('source_type', 'unknown')}")
                                st.write(f"PDF பக்க எண்: {source['page']}")
                                st.write(f"மூல இருப்பிடம்: {source['source']}")
                            else:
                                st.write(f"Type: {source.get('source_type', 'unknown')}")
                                st.write(f"PDF page: {source['page']}")
                                st.write(f"Source location: {source['source']}")

            if show_debug and source_details:
                with st.expander(
                    "🛠️ பிழைத்திருத்தம்: பெறப்பட்ட உரைப் பகுதிகள்"
                    if is_tamil_text(question)
                    else "🛠️ Debug: Retrieved passages",
                    expanded=False,
                ):
                    for source in source_details:
                        st.markdown(f"### SOURCE {source['number']}")
                        st.write(
                            "ஆவணம்:" if is_tamil_text(question) else "Document:",
                            source["display_name"],
                        )
                        st.write(
                            "PDF பக்க எண்:" if is_tamil_text(question) else "PDF page:",
                            source["page"],
                        )
                        st.write(
                            "கேள்விக்கான முக்கியச்சொல் மதிப்பெண்:"
                            if is_tamil_text(question)
                            else "Keyword score against question:",
                            round(keyword_score(question, source["content"]), 3),
                        )
                        st.text(source["content"])

            st.session_state.messages.append({"role": "assistant", "content": answer})

        except Exception as error:
            st.error(f"An error occurred while generating the answer: {error}")
            st.info("Check your API key, model name, dependencies, image type, and FAISS index.")

# =========================================================
# 16. FOOTER
# =========================================================

st.divider()

st.markdown("🌾 **Uzhavar Nanban**")
st.caption(
    "Agriculture information and farmer support for Tamil Nadu."
)
st.caption(
    "Developed using Streamlit, LangChain, FAISS, "
    "Hugging Face embeddings and OpenAI."
)