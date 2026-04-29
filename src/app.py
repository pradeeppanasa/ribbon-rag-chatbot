"""
Streamlit UI Application
RUBRIC: Streamlit UI Application
- Page config and layout implemented
- Search integrated correctly
- Results and sources displayed (1 mark)
- UI/UX design and examples (1 mark)

TASK: Create Streamlit web interface for Ribbon chatbot
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import streamlit as st

# ====================
# Page Config (must be first Streamlit call)
# ====================
st.set_page_config(
    page_title="Ribbon AI Assistant",
    page_icon="https://www.ribbonglobal.com/hubfs/logo.svg",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject actual Ribbon favicon (overrides Streamlit default)
st.markdown(
    '<link rel="icon" href="https://www.ribbonglobal.com/hubfs/logo.svg" type="image/svg+xml">',
    unsafe_allow_html=True,
)

# Custom CSS — Ribbon green palette
st.markdown("""
<style>
    .ribbon-header {
        background: linear-gradient(135deg, #1a7a4a 0%, #25a865 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .ribbon-header h1 { margin: 0; font-size: 2rem; }
    .ribbon-header p  { margin: 0.3rem 0 0; opacity: 0.9; font-size: 1rem; }
    .category-badge {
        display: inline-block;
        background: #e8f5ee;
        color: #1a7a4a;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .source-card {
        border-left: 3px solid #25a865;
        padding: 0.5rem 1rem;
        margin: 0.5rem 0;
        background: #f8fdf9;
        border-radius: 0 8px 8px 0;
    }
    div[data-testid="stMetric"] { background: #f0faf4; border-radius: 8px; padding: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ── Cached engine — loads once per server process, survives reruns ────────────
@st.cache_resource(show_spinner="Loading Ribbon AI engine — please wait...")
def _load_engine():
    try:
        from src.search_engine import RibbonSearchEngine
        return RibbonSearchEngine(), None
    except Exception as e:
        return None, str(e)

engine, engine_error = _load_engine()

# ====================
# Session-state bootstrap — must run before ANY widget renders
# ====================
if "_qv"  not in st.session_state: st.session_state["_qv"]  = 0
if "ribbon_query" not in st.session_state: st.session_state["ribbon_query"] = ""

# ====================
# Header
# ====================
st.markdown("""
<div class="ribbon-header">
    <img src="https://www.ribbonglobal.com/hubfs/logo.svg"
         style="height:48px; margin-bottom:0.6rem; filter:brightness(0) invert(1);">
    <p style="margin:0; font-size:1rem; opacity:0.9;">
        Grow together — ask anything about Ribbon accounts, cards, transfers, budgeting &amp; sustainability
    </p>
</div>
""", unsafe_allow_html=True)

# ====================
# Callbacks (defined before any widget that references them)
# ====================
_TOPIC_QUERIES = {
    "All Topics":           "",
    "Personal Accounts":    "What is included in a Ribbon student account?",
    "Business Accounts":    "Can freelancers open a Ribbon business account?",
    "Cards & Payments":     "How do I get a Ribbon virtual debit card?",
    "Money Transfers":      "How does Ribbon find the best exchange rates?",
    "Budgeting & Analytics":"How does Ribbon track my spending by category?",
    "Piggy Bank & Goals":   "How does the Ribbon piggy bank feature work?",
    "Sustainability":       "Are Ribbon cards made from recycled materials?",
    "Investments":          "What investment products is Ribbon planning to launch?",
}

def _on_topic_change():
    selected = st.session_state["topic_radio"]
    new_query = _TOPIC_QUERIES.get(selected, "")
    st.session_state["_qv"] += 1
    st.session_state[f"ribbon_query_{st.session_state['_qv']}"] = new_query
    st.session_state["ribbon_query"] = new_query

def _pick(q):
    st.session_state["_qv"] += 1
    st.session_state[f"ribbon_query_{st.session_state['_qv']}"] = q
    st.session_state["ribbon_query"] = q
    st.session_state["_auto_search"] = True

# ====================
# Sidebar
# ====================
with st.sidebar:
    st.image("https://www.ribbonglobal.com/hubfs/logo.svg", width=160)
    st.caption("_Grow together_")
    st.divider()

    # Topic filter
    st.markdown("### Filter by Topic")
    topic_filter = st.radio(
        "Select a topic",
        key="topic_radio",
        on_change=_on_topic_change,
        options=[
            "All Topics",
            "Personal Accounts",
            "Business Accounts",
            "Cards & Payments",
            "Money Transfers",
            "Budgeting & Analytics",
            "Piggy Bank & Goals",
            "Sustainability",
            "Investments",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    # Topic-to-hint mapping shown below filter
    topic_hints = {
        "Personal Accounts":    "e.g. standard, student, or multi-currency accounts",
        "Business Accounts":    "e.g. freelancer or sole proprietor accounts",
        "Cards & Payments":     "e.g. virtual/physical Visa debit cards",
        "Money Transfers":      "e.g. remittance marketplace, exchange rates",
        "Budgeting & Analytics":"e.g. smart spend tracking, bill splitting",
        "Piggy Bank & Goals":   "e.g. spare change, savings targets",
        "Sustainability":       "e.g. recycled cards, CO2 offset, Ribbon coins",
        "Investments":          "e.g. equity, crypto, forex, deposits (coming soon)",
    }
    if topic_filter != "All Topics":
        st.info(f"**{topic_filter}**\n\n{topic_hints[topic_filter]}")

    st.divider()

    # Statistics
    st.markdown("### Session Stats")
    if "query_count" not in st.session_state:
        st.session_state.query_count = 0
    st.metric("Queries this session", st.session_state.query_count)

    st.divider()

    # Engine status indicator
    st.markdown("### Engine Status")
    if engine_error:
        st.error(f"Load failed: {engine_error}")
    else:
        st.success("AI Ready")

    st.divider()
    if st.button("Reload Engine", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

# ====================
# Quick-Question Buttons (2 rows × 3)
# ====================
st.markdown("### Quick Questions")

QUICK_QUESTIONS = [
    ("🏦", "Personal Account",  "What features does the Ribbon personal account offer?"),
    ("💼", "Business Account",  "What does the Ribbon business account offer for freelancers?"),
    ("💳", "Debit Cards",       "What types of debit cards does Ribbon provide?"),
    ("💸", "Money Transfers",   "How does Ribbon's remittance marketplace work?"),
    ("📊", "Budget Tools",      "How does Ribbon's budget management and spend analytics work?"),
    ("♻️", "Sustainability",    "What sustainability features does Ribbon offer?"),
]

row1 = st.columns(3)
row2 = st.columns(3)
for idx, (icon, label, query) in enumerate(QUICK_QUESTIONS):
    col = (row1 if idx < 3 else row2)[idx % 3]
    with col:
        st.button(f"{icon} {label}", use_container_width=True,
                  on_click=_pick, args=(query,))

st.divider()

# ====================
# Search Input
# ====================
st.markdown("### Ask Ribbon")

placeholder_map = {
    "All Topics":           "e.g. 'How do I open a Ribbon account?'",
    "Personal Accounts":    "e.g. 'What is included in a Ribbon student account?'",
    "Business Accounts":    "e.g. 'Can freelancers open a Ribbon business account?'",
    "Cards & Payments":     "e.g. 'How do I get a Ribbon virtual debit card?'",
    "Money Transfers":      "e.g. 'How does Ribbon find the best exchange rates?'",
    "Budgeting & Analytics":"e.g. 'How does Ribbon track my spending by category?'",
    "Piggy Bank & Goals":   "e.g. 'How does the Ribbon piggy bank feature work?'",
    "Sustainability":       "e.g. 'Are Ribbon cards made from recycled materials?'",
    "Investments":          "e.g. 'What investment products is Ribbon planning to launch?'",
}

query_text = st.text_input(
    "Your question",
    key=f"ribbon_query_{st.session_state['_qv']}",
    placeholder=placeholder_map[topic_filter],
    label_visibility="collapsed",
)

_auto_search = st.session_state.pop("_auto_search", False)

# Show engine status near the search button
if engine_error:
    st.error(f"Engine error: {engine_error}")
else:
    st.success("Ready for search")

search_button = st.button("Search Ribbon Knowledge Base", use_container_width=True, type="primary", disabled=engine is None)

active_query = query_text.strip()
if not active_query and _auto_search:
    active_query = st.session_state.get("ribbon_query", "").strip()

effective_active_query = active_query
if active_query and topic_filter != "All Topics":
    effective_active_query = f"[Topic: {topic_filter}] {active_query}"

trigger = search_button or _auto_search

# ====================
# Results
# ====================
def display_results(results, query_text, generated_response):
    col_left, col_right = st.columns([3, 1])

    with col_left:
        st.markdown("#### AI Response")
        with st.container(border=True):
            st.markdown(generated_response)

    with col_right:
        st.markdown("#### Result Info")
        st.metric("Documents found", len(results))

    if results:
        with st.expander("View Source Documents", expanded=False):
            for i, doc in enumerate(results):
                source   = doc.metadata.get("source", "Unknown")
                category = doc.metadata.get("category", "general")
                st.markdown(f"""
<div class="source-card">
    <span class="category-badge">{category.replace('_',' ').title()}</span>
    <strong> {i+1}. {source}</strong><br>
    <small>{doc.page_content[:350]}...</small>
</div>
""", unsafe_allow_html=True)
    else:
        st.warning("No relevant documents found. Try rephrasing or selecting a different topic filter.")


if trigger and engine and active_query:
    st.session_state.query_count += 1
    st.markdown("---")
    with st.spinner("Searching Ribbon knowledge base..."):
        start_time = time.time()
        try:
            results, _ = engine.search_by_text(effective_active_query, k=5)
            generated_response = engine.synthesize_response(results, active_query)
            latency = time.time() - start_time
            st.caption(f"Answered in {latency:.2f}s")
            display_results(results, active_query, generated_response)
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Please try rephrasing your question or contact Ribbon support.")

elif trigger and not active_query:
    st.warning("Please enter a question.")
