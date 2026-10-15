"""Streamlit interface for the Hiligaynon grammar checker.

Run from the project root:
    streamlit run hiligaynon_checker/app.py
"""
from __future__ import annotations

import random
import sys
from html import escape as escape_html
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from nltk import Tree

from hiligaynon_checker.grammar_engine.grammar import (
    PLAIN_LABELS,
    TEMPLATES,
    GrammarChecker,
    leaf_roles,
    result_summary,
    sentence_kind,
    tree_to_dot,
)
from hiligaynon_checker.helpers.lexicon import CATEGORY_META, Lexicon, load_lexicon, normalize


st.set_page_config(
    page_title="Hiligaynon Grammar Checker",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = False


def toggle_theme() -> None:
    st.session_state["dark_mode"] = not st.session_state["dark_mode"]


dark_mode = st.session_state["dark_mode"]


st.markdown(
    """
    <style>
    :root {
        --ink: #292b27;
        --muted: #6e716b;
        --forest: #355743;
        --paper: #faf9f6;
        --line: #e5e2da;
        --primary-color: #355743 !important;
    }
    [data-testid="stApp"] { --primary-color: #355743 !important; }
    html, body, [class*="css"] {
        font-family: Inter, "Segoe UI", sans-serif;
        color: var(--ink);
    }
    [data-testid="stAppViewContainer"], [data-testid="stMain"] { background: var(--paper) !important; }
    [data-testid="stHeader"] {
        background: linear-gradient(to bottom, rgba(250,249,246,.95) 0 2rem, transparent 2rem 100%) !important;
    }
    .block-container {
        max-width: 1160px;
        padding-top: .9rem;
        padding-bottom: 3.5rem;
    }
    [data-testid="stSidebar"] { background: #f1f0eb !important; }
    [data-testid="stSidebar"] > div:first-child { border-right: 1px solid var(--line); }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: var(--ink); }
    h1 {
        color: var(--ink) !important;
        font-family: Georgia, "Times New Roman", serif;
        font-weight: 500;
        letter-spacing: -.025em;
    }
    h2, h3 { color: var(--ink) !important; letter-spacing: -.015em; }
    p, label, [data-testid="stCaptionContainer"] { color: var(--muted) !important; }
    .brand-banner {
        display: inline-flex;
        align-items: center;
        gap: .45rem;
        margin: .25rem 0 .35rem;
        padding: .48rem .72rem;
        border-left: 3px solid var(--forest);
        background: #eeede7;
        color: var(--forest);
        font-size: .68rem;
        font-weight: 700;
        letter-spacing: .1em;
    }
    .brand-banner .brand-slash { color: #97988f; font-weight: 400; }
    [data-testid="stSidebar"] div[data-testid="stButton"] button {
        width: 36px !important;
        height: 36px !important;
        min-height: 36px !important;
        padding: 0 !important;
        border-radius: 50% !important;
        border: 1px solid #bcd3b6 !important;
        background: #deeed9 !important;
        color: #2f5138 !important;
        font-family: "Segoe UI Symbol", sans-serif;
        font-size: 1rem !important;
        display: flex;
        align-items: center;
        justify-content: center;
        line-height: 1;
        box-shadow: 0 2px 7px rgba(34,55,40,.12);
    }
    [data-testid="stSidebar"] div[data-testid="stButton"] button:hover {
        background: #c8dfc2 !important;
        transform: rotate(12deg) scale(1.05);
        box-shadow: 0 4px 11px rgba(34,55,40,.2);
    }
    [data-testid="stSidebar"] div[data-testid="stButton"] button,
    [data-testid="stSidebar"] div[data-testid="stButton"] button * {
        color: #2f5138 !important;
    }
    [data-testid="stSidebar"] {
        width: 18rem !important;
        min-width: 18rem !important;
        max-width: 18rem !important;
    }
    .rail-mark { display: none; }
    [data-testid="stSidebar"][aria-expanded="false"] {
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        width: 58px !important;
        min-width: 58px !important;
        max-width: 58px !important;
        height: 100vh !important;
        transform: none !important;
        overflow: hidden !important;
        z-index: 1 !important;
        background: #f1f0eb !important;
        border-right: 1px solid var(--line);
        box-shadow: 1px 0 8px rgba(41,43,39,.1);
    }
    [data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarContent"] > * {
        visibility: hidden;
    }
    [data-testid="stSidebar"][aria-expanded="false"]
        [data-testid="stElementContainer"]:has(.rail-mark) + [data-testid="stElementContainer"] {
        visibility: visible !important;
    }
    [data-testid="stSidebar"][aria-expanded="false"]
        [data-testid="stElementContainer"]:has(.rail-mark-0) + [data-testid="stElementContainer"] [data-testid="stButton"] {
        position: fixed !important;
        left: 9px !important;
        width: 40px !important;
        height: 40px !important;
        top: 72px !important;
    }
    [data-testid="stSidebar"][aria-expanded="false"]
        [data-testid="stElementContainer"]:has(.rail-mark-1) + [data-testid="stElementContainer"] [data-testid="stButton"] {
        position: fixed !important;
        left: 9px !important;
        width: 40px !important;
        height: 40px !important;
        top: 120px !important;
    }
    [data-testid="stSidebar"][aria-expanded="false"]
        [data-testid="stElementContainer"]:has(.rail-mark-2) + [data-testid="stElementContainer"] [data-testid="stButton"] {
        position: fixed !important;
        left: 9px !important;
        width: 40px !important;
        height: 40px !important;
        top: 168px !important;
    }
    [data-testid="stSidebar"][aria-expanded="false"] div[data-testid="stButton"] button {
        width: 40px !important;
        height: 40px !important;
        min-height: 40px !important;
    }
    .sidebar-title {
        font-family: Georgia, "Times New Roman", serif;
        font-size: .98rem;
        font-weight: 600;
        color: var(--ink);
        letter-spacing: -.01em;
        margin: 0;
        padding-left: .2rem;
        line-height: 1.25;
    }
    [data-testid="stSidebar"] [data-testid="stElementContainer"]:has(.sidebar-title) {
        display: flex;
        align-items: center;
        min-height: 46px;
    }
    .logo-anchor { display: none; }
    [data-testid="stSidebar"] [data-testid="stElementContainer"]:has(.logo-anchor) + [data-testid="stElementContainer"] [data-testid="stButton"] button {
        width: 46px !important;
        height: 46px !important;
        min-height: 46px !important;
        border-radius: 13px !important;
        font-family: Georgia, "Times New Roman", serif;
        font-size: 1.5rem !important;
        font-weight: 700;
        box-shadow: 0 3px 10px rgba(34,55,40,.18);
    }
    [data-testid="stSidebar"] [data-testid="stElementContainer"]:has(.logo-anchor) + [data-testid="stElementContainer"] [data-testid="stButton"] button:hover {
        background: #c8dfc2 !important;
        transform: scale(1.06) !important;
    }
    [data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding: .7rem 1rem 1.1rem; }
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: .8rem !important; }
    [data-testid="stSidebar"] p { margin: .2rem 0 !important; line-height: 1.35; }
    [data-testid="stSidebar"] hr { margin: .65rem 0 !important; }
    [data-testid="stSidebar"] [data-testid="stMetric"] { padding: .45rem 0 !important; border-top: 0 !important; }
    [data-testid="stSidebar"] [data-testid="stMetricValue"] { font-size: 1.4rem !important; line-height: 1.2; }
    [data-testid="stSidebar"] [data-testid="stMetricLabel"] { font-size: .75rem !important; line-height: 1.25; }
    [data-testid="stSidebar"] [data-testid="stMarkdown"] p { font-size: .78rem; }
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] { gap: .25rem !important; }
    div[data-testid="stTextArea"] textarea,
    div[data-testid="stTextInput"] input,
    div[data-testid="stSelectbox"] > div > div {
        border-radius: 6px;
        border-color: #d8e1d7;
        background: #fff;
        color: #18342c !important;
    }
    div[data-testid="stTextArea"] textarea::placeholder,
    div[data-testid="stTextInput"] input::placeholder {
        color: #68766d !important;
        opacity: 1;
    }
    div[data-baseweb="tag"] {
        background: #e7efe6 !important;
        color: #245442 !important;
    }
    div[data-testid="stRadio"] label { color: #18342c !important; }
    div[data-testid="stTextArea"] textarea:focus,
    div[data-testid="stTextInput"] input:focus {
        border-color: #57836a;
        box-shadow: 0 0 0 2px rgba(63,112,82,.12);
    }
    .stButton > button, [data-testid="stFormSubmitButton"] button {
        border-radius: 6px;
        font-weight: 600;
        min-height: 2.65rem;
        background: #deeed9;
        border: 1px solid #bcd3b6;
        color: #2f5138;
        transition: transform .16s ease, box-shadow .16s ease, background-color .16s ease;
    }
    .stButton > button:hover, [data-testid="stFormSubmitButton"] button:hover {
        background: #c8dfc2;
        border-color: #a8c3a1;
        transform: translateY(-1px);
        box-shadow: 0 3px 8px rgba(41,43,39,.12);
    }
    button[kind="primary"], [data-testid="stFormSubmitButton"] button[kind="primary"] {
        background: var(--forest) !important;
        border-color: var(--forest) !important;
        color: white !important;
    }
    button[kind="primary"]:hover, [data-testid="stFormSubmitButton"] button[kind="primary"]:hover {
        background: #2c4a38 !important;
        border-color: #2c4a38 !important;
        color: white !important;
    }
    div[data-testid="stTabs"] [role="tablist"] {
        position: sticky;
        top: 0;
        z-index: 6;
        gap: 0;
        background: var(--paper);
        border-bottom: 1px solid var(--line);
    }
    div[data-testid="stTabs"] [role="tablist"]::after {
        content: "";
        position: absolute;
        bottom: -1px;
        left: 0;
        width: 25%;
        height: 2px;
        background: var(--forest) !important;
        transition: transform .28s cubic-bezier(.2,.75,.25,1);
    }
    div[data-testid="stTabs"] [role="tablist"]:has(> button:nth-child(2)[aria-selected="true"])::after { transform: translateX(100%); }
    div[data-testid="stTabs"] [role="tablist"]:has(> button:nth-child(3)[aria-selected="true"])::after { transform: translateX(200%); }
    div[data-testid="stTabs"] [role="tablist"]:has(> button:nth-child(4)[aria-selected="true"])::after { transform: translateX(300%); }
    div[data-testid="stTabs"]:has([role="tablist"] > button:nth-child(1)[aria-selected="true"]) [role="tabpanel"]:nth-of-type(1),
    div[data-testid="stTabs"]:has([role="tablist"] > button:nth-child(2)[aria-selected="true"]) [role="tabpanel"]:nth-of-type(2),
    div[data-testid="stTabs"]:has([role="tablist"] > button:nth-child(3)[aria-selected="true"]) [role="tabpanel"]:nth-of-type(3),
    div[data-testid="stTabs"]:has([role="tablist"] > button:nth-child(4)[aria-selected="true"]) [role="tabpanel"]:nth-of-type(4) {
        animation: panel-slide-in .26s ease-out both;
    }
    div[data-testid="stTabs"] [role="tablist"],
    div[data-testid="stTabs"] [role="tablist"] *,
    div[data-testid="stTabs"] [data-baseweb="tab-list"],
    div[data-testid="stTabs"] [data-baseweb="tab-list"] * {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        opacity: 1 !important;
        filter: none !important;
        font-weight: 600 !important;
    }
    div[data-testid="stTabs"] [role="tab"],
        div[data-testid="stTabs"] [data-baseweb="tab"] {
            flex: 1;
            border-radius: 0;
            padding: .7rem .55rem;
            background: transparent;
            transition: color .16s ease, background-color .16s ease, transform .16s ease;
        }
    div[data-testid="stTabs"] [role="tab"]:hover,
        div[data-testid="stTabs"] [data-baseweb="tab"]:hover { background: #f1f0eb; transform: translateY(-1px); }
    div[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
        color: var(--forest);
        background: transparent;
        border-bottom: 0 !important;
        box-shadow: none !important;
    }
    .section-banner {
        display: flex;
        align-items: center;
        gap: .9rem;
        min-height: 52px;
        margin: .65rem 0 1.25rem;
        padding: .6rem .9rem;
        border-left: 3px solid var(--forest);
        background: #f1f0eb;
        animation: banner-slide-in .28s ease-out both;
    }
    .banner-tag { color: var(--forest); font-size: .68rem; font-weight: 700; letter-spacing: .12em; }
    .banner-copy { color: var(--muted); font-size: .86rem; }
    .result-line {
        display: flex;
        align-items: center;
        gap: .5rem;
        min-height: 44px;
        margin: .3rem 0 .9rem;
        padding: .55rem .85rem;
        border-left: 3px solid var(--forest);
        background: #f1f0eb;
        color: var(--ink);
        font-size: .95rem;
    }
    .result-line b { color: var(--forest); letter-spacing: .06em; font-weight: 700; }
    .result-line-bad { border-left-color: #a4503f; }
    .result-line-bad b { color: #a4503f; }
    @keyframes banner-slide-in {
        from { opacity: 0; transform: translateX(8px); }
        to { opacity: 1; transform: translateX(0); }
    }
    @keyframes panel-slide-in {
        from { opacity: 0; transform: translateX(10px); }
        to { opacity: 1; transform: translateX(0); }
    }
    div[data-testid="stMetric"] {
        border: 0;
        border-top: 1px solid var(--line);
        border-radius: 0;
        background: transparent;
        padding: .65rem 0;
    }
    div[data-testid="stMetricValue"] { color: var(--forest) !important; }
    div[data-testid="stMetricLabel"] { color: var(--muted) !important; }
    div[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 4px; overflow: hidden; }
    .word-chip {
        display: inline-block; margin: .18rem .15rem .18rem 0; padding: .25rem .58rem;
        border-radius: 4px; background: #eeede7; color: var(--forest); font-size: .82rem;
    }
    .fly-stage {
        position: fixed;
        left: 7rem;
        top: 5rem;
        width: 0;
        height: 0;
        z-index: 99999;
        pointer-events: none;
    }
    .fly-word {
        position: absolute;
        left: 0;
        top: 0;
        margin: 0;
        padding: .32rem .72rem;
        border-radius: 999px;
        background: #deeed9;
        border: 1px solid #bcd3b6;
        color: #2f5138;
        font-size: .92rem;
        font-weight: 600;
        white-space: nowrap;
        box-shadow: 0 4px 12px rgba(34,55,40,.16);
        animation: word-fly 2s cubic-bezier(.18,.65,.32,1) both;
    }
    .fly-word.fly-alt {
        background: #f4f0e4;
        border-color: #ddd6bf;
        color: #5a5232;
    }
    @keyframes word-fly {
        0% { opacity: 0; transform: translate(-50%,-50%) scale(.5) rotate(0deg); }
        10% { opacity: 1; transform: translate(-50%,-50%) scale(1) rotate(0deg); }
        100% { opacity: 0; transform: translate(calc(-50% + var(--dx)), calc(-50% + var(--dy))) scale(1.08) rotate(var(--rot)); }
    }
    @media (max-width: 700px) {
        .block-container { padding: 1rem 1rem 2.5rem; }
    }
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after { animation-duration: .01ms !important; transition-duration: .01ms !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if dark_mode:
    st.markdown(
        """
        <style>
        :root {
            --ink: #e8ebe7;
            --muted: #b4beb6;
            --forest: #9bc9a7;
            --paper: #171b18;
            --line: #39443c;
            --primary-color: #83b892 !important;
        }
        [data-testid="stApp"] { --primary-color: #83b892 !important; }
        [data-testid="stAppViewContainer"], [data-testid="stMain"] { background: #171b18 !important; }
        [data-testid="stHeader"] { background: rgba(23,27,24,.96) !important; }
        [data-testid="stSidebar"] { background: #202622 !important; }
        [data-testid="stSidebar"][aria-expanded="false"] {
            background: #202622 !important;
            border-right-color: #39443c !important;
            box-shadow: 1px 0 8px rgba(0,0,0,.35) !important;
        }
        h1, h2, h3, p, label, [data-testid="stCaptionContainer"],
        [data-testid="stMarkdownContainer"] { color: var(--ink) !important; }
        [data-testid="stCaptionContainer"] p, .banner-copy { color: var(--muted) !important; }
        .brand-banner, .section-banner, .result-line { background: #252d27 !important; color: #b5d5ba !important; }
        .result-line b { color: #b5d5ba !important; }
        .result-line-bad { border-left-color: #c47a68 !important; }
        .result-line-bad b { color: #e0a898 !important; }
        .brand-banner .brand-slash { color: #87938a !important; }
        div[data-testid="stSidebar"] div[data-testid="stButton"] button {
            background: #dcebd7 !important;
            border-color: #a9c4a3 !important;
            color: #24543c !important;
        }
        div[data-testid="stSidebar"] div[data-testid="stButton"] button:hover {
            background: #c8dfc2 !important;
            color: #1d4630 !important;
        }
        div[data-testid="stSidebar"] div[data-testid="stButton"] button,
        div[data-testid="stSidebar"] div[data-testid="stButton"] button * {
            color: #24543c !important;
        }
        div[data-testid="stSidebar"] div[data-testid="stButton"] button:hover,
        div[data-testid="stSidebar"] div[data-testid="stButton"] button:hover * {
            color: #1d4630 !important;
        }
        div[data-testid="stTextArea"] textarea, div[data-testid="stTextInput"] input,
        div[data-testid="stSelectbox"] > div > div, [data-baseweb="select"] > div {
            background: #222a24 !important;
            color: #edf0ec !important;
            border-color: #4a574d !important;
        }
        div[data-testid="stTextArea"] textarea::placeholder,
        div[data-testid="stTextInput"] input::placeholder { color: #aab5ac !important; }
        div[data-testid="stTabs"] [role="tablist"],
        div[data-testid="stTabs"] [role="tablist"] *,
        div[data-testid="stTabs"] [data-baseweb="tab-list"],
        div[data-testid="stTabs"] [data-baseweb="tab-list"] * {
            color: #d0d4cc !important;
            -webkit-text-fill-color: #d0d4cc !important;
            opacity: 1 !important;
            filter: none !important;
        }
        div[data-testid="stTabs"] [role="tab"]:hover,
        div[data-testid="stTabs"] [role="tab"]:hover * { background: #252d27 !important; color: #e8ebe7 !important; }
        div[data-testid="stTabs"] [aria-selected="true"],
        div[data-testid="stTabs"] [aria-selected="true"] * {
            color: #b5d5ba !important;
            -webkit-text-fill-color: #b5d5ba !important;
        }
        div[data-testid="stTabs"] [role="tablist"]::after { background: #9bc9a7 !important; }
        div[data-testid="stMetricValue"] { color: #b5d5ba !important; }
        div[data-testid="stMetricLabel"] { color: #b4beb6 !important; }
        div[data-testid="stDataFrame"] { border-color: #39443c !important; }
        div[data-baseweb="tag"] { background: #33453a !important; color: #dce9de !important; }
        .word-chip { background: #303a32 !important; color: #c9e0ce !important; }
        .fly-word { background: #33453a !important; border-color: #536b58 !important; color: #dce9de !important; }
        .fly-word.fly-alt { background: #3d4034 !important; border-color: #5c6049 !important; color: #dedfc4 !important; }
        [data-testid="stAlert"] { background: #252d27 !important; color: #e8ebe7 !important; }
        .stButton > button, [data-testid="stFormSubmitButton"] button {
            background: #33453a !important;
            border-color: #536b58 !important;
            color: #dce9de !important;
        }
        .stButton > button:hover, [data-testid="stFormSubmitButton"] button:hover {
            background: #3f5546 !important;
            border-color: #6a8770 !important;
        }
        button[kind="primary"], [data-testid="stFormSubmitButton"] button[kind="primary"] {
            background: #477653 !important;
            border-color: #477653 !important;
            color: #fff !important;
        }
        button[kind="primary"]:hover, [data-testid="stFormSubmitButton"] button[kind="primary"]:hover {
            background: #3a6245 !important;
            border-color: #3a6245 !important;
            color: #fff !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def get_lexicon() -> Lexicon:
    return load_lexicon()


@st.cache_resource
def get_checker(extra_names: tuple[str, ...] = ()) -> GrammarChecker:
    lexicon = get_lexicon()
    return GrammarChecker(lexicon.build_categories(extra_names=list(extra_names)))


if "extra_names" not in st.session_state:
    st.session_state.extra_names = []

lex = get_lexicon()
checker = get_checker(tuple(st.session_state.extra_names))
grammar_word_count = sum(len(category.words) for category in checker.categories.values())


def tree_word_paths(tree: Tree) -> list[dict[str, object]]:
    """Collect each terminal word with the grammar states on its parse path."""
    paths: list[dict[str, object]] = []

    def walk(node, ancestors: list[str]) -> None:
        if isinstance(node, Tree):
            branch = [*ancestors, node.label()]
            for child in node:
                walk(child, branch)
        else:
            role = ancestors[-1] if ancestors else ""
            paths.append(
                {
                    "word": str(node),
                    "states": ancestors,
                    "role": PLAIN_LABELS.get(role, role),
                }
            )

    walk(tree, [])
    return paths


def render_rule_trace(result: dict, view_key: str) -> None:
    """Animate per-word grammar paths for the selected parse or a rejected input."""
    replay_key = f"trace_replay_{view_key}"
    replay_count = st.session_state.get(replay_key, 0)
    if st.button("Replay rule path", key=f"replay_button_{view_key}"):
        replay_count += 1
        st.session_state[replay_key] = replay_count

    accepted = result["valid"]
    if accepted:
        tree = result["trees"][0]
        token_paths = tree_word_paths(tree)
    else:
        token_paths = []
        for token in result["tokens"]:
            categories = [letter for letter, words in checker.vocab.items() if token in words]
            token_paths.append(
                {
                    "word": token,
                    "states": categories,
                    "role": " · ".join(PLAIN_LABELS.get(letter, letter) for letter in categories)
                    if categories
                    else "Not in vocabulary",
                    "known": bool(categories),
                }
            )

    def state_node(symbol: str, delay_ms: int, extra_class: str = "") -> str:
        label = PLAIN_LABELS.get(symbol, symbol)
        return (
            f'<span class="trace-node {extra_class}" style="--delay:{delay_ms}ms">'
            f'<b>{escape_html(symbol)}</b><small>{escape_html(label)}</small></span>'
        )

    lanes = []
    max_state_count = 0
    for index, item in enumerate(token_paths):
        word = str(item["word"])
        states = list(item["states"])
        max_state_count = max(max_state_count, len(states))
        lane_delay = index * 860
        nodes = [state_node("START", lane_delay, "trace-start")]
        for step, symbol in enumerate(states, start=1):
            nodes.append(f'<span class="trace-edge" style="--delay:{lane_delay + step * 125}ms">›</span>')
            nodes.append(state_node(symbol, lane_delay + step * 125))
        terminal_delay = lane_delay + (len(states) + 1) * 125
        nodes.extend(
            [
                f'<span class="trace-edge" style="--delay:{terminal_delay}ms">›</span>',
                f'<span class="trace-node trace-token" style="--delay:{terminal_delay}ms">'
                f'<b>{escape_html(word)}</b><small>{escape_html(str(item["role"]))}</small></span>',
            ]
        )
        if not accepted and not item.get("known", False):
            nodes.append(f'<span class="trace-badge trace-missing" style="--delay:{terminal_delay + 120}ms">UNKNOWN</span>')
        lanes.append(
            '<div class="trace-lane"><div class="trace-lane-label">'
            f'WORD {index + 1:02d}</div><div class="trace-flow">{"".join(nodes)}</div></div>'
        )

    if not token_paths:
        lanes.append('<div class="trace-empty">Enter a sentence to see its rule path.</div>')

    final_delay = max(450, max(0, len(token_paths) - 1) * 860 + (max_state_count + 2) * 125 + 220)
    result_class = "trace-accept" if accepted else "trace-reject"
    result_word = "ACCEPT" if accepted else "REJECT"
    html = f"""
    <html><head><meta charset="utf-8"><style>
      * {{ box-sizing: border-box; }}
      body {{ margin: 0; background: transparent; color: {'#e8ebe7' if dark_mode else '#292b27'};
        font-family: Inter, 'Segoe UI', sans-serif; }}
      .trace-shell {{ border: 1px solid {'#39443c' if dark_mode else '#e5e2da'}; border-radius: 10px;
        padding: 14px; background: {'#202622' if dark_mode else '#fff'}; overflow: hidden; }}
      .trace-top {{ display:flex; align-items:center; justify-content:space-between; gap:12px; margin-bottom:12px; }}
      .trace-title {{ font-size:12px; letter-spacing:.12em; font-weight:700; color:{'#b5d5ba' if dark_mode else '#355743'}; }}
      .trace-run {{ font-size:11px; color:{'#aab5ac' if dark_mode else '#73786f'}; }}
      .trace-lanes {{ display:grid; gap:8px; }}
      .trace-lane {{ display:flex; align-items:center; gap:12px; min-width:0; padding:8px 0;
        border-top:1px solid {'#354139' if dark_mode else '#eeede7'}; }}
      .trace-lane-label {{ flex:0 0 54px; color:{'#9ba89e' if dark_mode else '#858980'};
        font-size:9px; font-weight:700; letter-spacing:.08em; }}
      .trace-flow {{ display:flex; align-items:center; gap:7px; min-width:0; overflow-x:auto; padding:4px 3px 7px; }}
      .trace-node {{ flex:0 0 auto; display:flex; flex-direction:column; justify-content:center; gap:2px;
        min-width:68px; min-height:48px; padding:6px 9px; border:1px solid {'#506054' if dark_mode else '#d7ded6'};
        border-radius:7px; background:{'#29332c' if dark_mode else '#f7f8f5'};
        animation: node-arrive .48s ease both; animation-delay:var(--delay); }}
      .trace-node b {{ font-size:12px; line-height:1.1; color:{'#e8ebe7' if dark_mode else '#293b30'}; }}
      .trace-node small {{ font-size:9px; line-height:1.15; color:{'#aab5ac' if dark_mode else '#73786f'}; }}
      .trace-start {{ min-width:60px; border-color:{'#6c9274' if dark_mode else '#9ab39d'}; }}
      .trace-token {{ border-color:{'#70a47a' if dark_mode else '#8baa8d'}; background:{'#304535' if dark_mode else '#eaf2e9'}; }}
      .trace-token b {{ color:{'#d4ead7' if dark_mode else '#254a32'}; }}
      .trace-edge {{ flex:0 0 auto; color:{'#8fbd99' if dark_mode else '#63826b'}; font-size:20px; line-height:1;
        animation: edge-arrive .36s ease both; animation-delay:var(--delay); }}
      .trace-badge {{ flex:0 0 auto; padding:6px 9px; border-radius:999px; font-size:9px; font-weight:800;
        animation: node-arrive .48s ease both; animation-delay:var(--delay); }}
      .trace-missing {{ color:#9b493c; background:#fae9e5; }}
      .trace-empty {{ padding:18px 4px; color:{'#aab5ac' if dark_mode else '#73786f'}; font-size:13px; }}
      .trace-outcome {{ display:flex; align-items:center; justify-content:center; gap:10px; margin-top:13px;
        padding:11px; border-radius:7px; font-size:12px; font-weight:800; letter-spacing:.12em;
        animation: outcome-arrive .6s ease both; animation-delay:{final_delay}ms; }}
      .trace-accept {{ color:{'#d4ead7' if dark_mode else '#245432'}; background:{'#304535' if dark_mode else '#eaf2e9'}; }}
      .trace-reject {{ color:#9b493c; background:{'#492d29' if dark_mode else '#fae9e5'}; }}
      @keyframes node-arrive {{ 0% {{ opacity:.26; transform:translateY(5px) scale(.97); box-shadow:none; }}
        55% {{ opacity:1; transform:translateY(-1px) scale(1.02); box-shadow:0 0 18px {'rgba(134,190,145,.40)' if dark_mode else 'rgba(81,139,91,.27)'}; }}
        100% {{ opacity:1; transform:translateY(0) scale(1); }} }}
      @keyframes edge-arrive {{ from {{ opacity:.12; }} to {{ opacity:1; }} }}
      @keyframes outcome-arrive {{ from {{ opacity:.25; transform:scale(.97); }} to {{ opacity:1; transform:scale(1); }} }}
      @media (max-width:600px) {{ .trace-shell {{ padding:10px; }} .trace-lane {{ gap:7px; }}
        .trace-lane-label {{ flex-basis:44px; }} .trace-node {{ min-width:60px; }} }}
      @media (prefers-reduced-motion:reduce) {{ *,*::before,*::after {{ animation:none !important; opacity:1 !important; }} }}
    </style></head><body>
      <div class="trace-shell" data-run="{replay_count}">
        <div class="trace-top"><span class="trace-title">WORD-BY-WORD GRAMMAR TRACE</span>
          <span class="trace-run">{'Reading 1 of ' + str(len(result['trees'])) if accepted and len(result['trees']) > 1 else 'Sentence path'}</span></div>
        <div class="trace-lanes">{''.join(lanes)}</div>
        <div class="trace-outcome {result_class}">{result_word} · {'ALL RULES MATCHED' if accepted else 'NO COMPLETE RULE MATCH'}</div>
      </div>
    </body></html>
    """
    components.html(html, height=min(660, 148 + len(token_paths) * 74), scrolling=True)


def render_result(result: dict, view_key: str) -> None:
    """Show the animated grammar path and details for a sentence check."""
    if not result["valid"]:
        st.markdown(
            f'<div class="result-line result-line-bad"><b>Result :</b> {escape_html(result_summary(result))}</div>',
            unsafe_allow_html=True,
        )
        render_rule_trace(result, view_key)
        if result["unknown"]:
            st.markdown("**Words we could not find in the vocabulary**")
            for token, suggestions in result["unknown"].items():
                if suggestions:
                    st.markdown(f"- `{token}` — possible matches: " + ", ".join(f"`{word}`" for word in suggestions))
                else:
                    st.markdown(f"- `{token}`")
            st.caption("The other words may fit the pattern. Try checking spelling or using the dictionary search.")
        else:
            st.caption("The words are recognized, but their order does not fit a pattern the checker knows.")
            st.info("Try the Sentence builder for examples of supported word orders.")
        return

    tree = result["trees"][0]
    st.markdown(
        f'<div class="result-line"><b>Result :</b> {escape_html(result_summary(result))} | {escape_html(sentence_kind(tree))}</div>',
        unsafe_allow_html=True,
    )
    render_rule_trace(result, view_key)
    with st.expander("View parse tree and word roles"):
        show_symbols = st.checkbox("Show grammar symbols", key=f"show_symbols_{view_key}")
        for index, parsed_tree in enumerate(result["trees"], 1):
            if len(result["trees"]) > 1:
                st.caption(f"Reading {index} of {len(result['trees'])}")
            st.graphviz_chart(
                tree_to_dot(parsed_tree, plain=not show_symbols, dark=dark_mode),
                use_container_width=True,
            )

        rows = [
            {"Word": word, "Role": role, "Meaning": lex.meaning_of(word)}
            for word, role in leaf_roles(tree)
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=210)


def text_value(value) -> str:
    """Convert CSV cells to clean display text."""
    if value is None or pd.isna(value):
        return ""
    value = str(value).strip()
    return "" if value.lower() in {"nan", "none", "unknown"} else value


def set_dictionary_word(word: str) -> None:
    """Open a selected word from either the dictionary or word-bank results."""
    st.session_state["dictionary_query"] = word
    st.session_state["dictionary_selected_word"] = normalize(word)
    st.session_state["dictionary_direction"] = "Hiligaynon → English"


def clear_dictionary_selection() -> None:
    st.session_state.pop("dictionary_selected_word", None)


def render_dictionary_entry(word: str) -> None:
    """Present a headword and its available definitions as a dictionary entry."""
    result = lex.lookup(word)
    merged = result["lexicon"]
    pinoy = result["pinoy"]
    motus = result["motus"]

    display_word = word
    for frame in (merged, pinoy, motus):
        if len(frame):
            display_word = text_value(frame.iloc[0]["word"]) or word
            break

    st.subheader(display_word)

    metadata = []
    normalized_word = normalize(word)
    grammar_roles = [
        category.label
        for letter, category in checker.categories.items()
        if normalized_word in checker.vocab.get(letter, set())
    ]
    if grammar_roles:
        metadata.append("Grammar: " + " · ".join(grammar_roles))
    if len(merged):
        parts_of_speech = list(dict.fromkeys(text_value(value) for value in merged["pos_word"] if text_value(value)))
        sentiments = list(dict.fromkeys(text_value(value) for value in merged["sentiment"] if text_value(value)))
        if parts_of_speech:
            metadata.append(" · ".join(parts_of_speech))
        if sentiments:
            metadata.append("Sentiment: " + ", ".join(sentiments))
    if metadata:
        st.caption("  |  ".join(metadata))

    meanings = list(dict.fromkeys(text_value(value) for value in merged["meaning"] if text_value(value))) if len(merged) else []
    if meanings:
        st.markdown("**Quick meaning**")
        for meaning in meanings[:5]:
            st.write(meaning)

    source_count = sum(bool(len(frame)) for frame in (merged, pinoy, motus))
    if source_count:
        st.caption(f"Found in {source_count} local source{'s' if source_count != 1 else ''}.")

    if len(pinoy):
        with st.expander(f"PinoyDictionary · {len(pinoy)} entr{'y' if len(pinoy) == 1 else 'ies'}", expanded=not meanings):
            for _, row in pinoy.head(5).iterrows():
                definition = text_value(row.get("definition"))
                if definition:
                    st.write(definition)
                link = text_value(row.get("link"))
                if link.startswith("https://") or link.startswith("http://"):
                    st.markdown(f"[Open source entry]({link})")

    if len(motus):
        with st.expander(f"Cecile Motus · {len(motus)} entr{'y' if len(motus) == 1 else 'ies'}", expanded=not meanings and not len(pinoy)):
            for _, row in motus.head(5).iterrows():
                part_of_speech = text_value(row.get("part_of_speech"))
                if part_of_speech:
                    st.caption(part_of_speech)
                definition = text_value(row.get("definitions"))
                if definition:
                    st.write(definition)
                examples = text_value(row.get("examples"))
                if examples:
                    st.markdown("**Examples**")
                    st.write(examples)

    if not meanings and not len(pinoy) and not len(motus):
        meaning = lex.meaning_of(word)
        st.write(meaning or "This word is in the grammar vocabulary, but no definition is available in the included dictionaries.")

def render_word_card(
    word: str,
    key: str,
    pos: str = "",
    source: str = "",
    open_in_dialog: bool = True,
) -> None:
    """Show a compact, clickable dictionary-style word card."""
    meaning = lex.meaning_of(word)
    display_word = word
    for frame in lex.lookup(word).values():
        if len(frame):
            display_word = text_value(frame.iloc[0]["word"]) or word
            break
    with st.container(border=True):
        top, action = st.columns([3, 1])
        with top:
            st.markdown(f"**{escape_html(display_word)}**")
            detail = " · ".join(part for part in (pos, source) if part)
            if detail:
                st.caption(detail)
        with action:
            st.button(
                "Open",
                key=key,
                on_click=open_word_card if open_in_dialog else set_dictionary_word,
                args=(word,),
                use_container_width=True,
            )
        if meaning:
            st.write(meaning)
        else:
            st.caption("Definition available in the dictionary tab, if included.")


def set_example(example: str) -> None:
    st.session_state["sentence_input"] = example
    st.session_state["run_check"] = True


def render_section_banner(tag: str, copy: str) -> None:
    st.markdown(
        f'<div class="section-banner"><span class="banner-tag">{tag}</span>'
        f'<span class="banner-copy">{copy}</span></div>',
        unsafe_allow_html=True,
    )


HOW_IT_WORKS_TEXT = (
    "The checker matches each word to a vocabulary class, then looks for "
    "a supported sentence pattern. The animation follows the accepted grammar "
    "parse path; the underlying checker uses a context-free grammar, so rules "
    "can branch and nest."
)


def open_info(kind: str) -> None:
    st.session_state["info_dialog"] = kind


def trigger_logo_burst() -> None:
    st.session_state["logo_burst_pending"] = True


def render_logo_burst() -> None:
    pool = sorted({word for words in checker.vocab.values() for word in words if word != "."})
    if not pool:
        return
    picks = random.sample(pool, min(16, len(pool)))
    chips = []
    for index, word in enumerate(picks):
        dx = random.randint(-420, 420)
        dy = random.randint(-380, -90)
        rot = random.randint(-28, 28)
        tone = "fly-alt" if index % 2 else ""
        chips.append(
            f'<span class="fly-word {tone}" style="--dx:{dx}px;--dy:{dy}px;--rot:{rot}deg;'
            f'animation-delay:{index * 55}ms">{escape_html(word)}</span>'
        )
    st.markdown(f'<div class="fly-stage">{"".join(chips)}</div>', unsafe_allow_html=True)


def open_word_card(word: str) -> None:
    st.session_state["word_card_word"] = word
    st.session_state["word_card_pending"] = True


@st.dialog("How a check works", width="large")
def how_it_works_dialog() -> None:
    st.markdown(HOW_IT_WORKS_TEXT)
    st.caption("Patterns are limited to the rules included in this project.")
    if st.button("Close", type="primary", use_container_width=True):
        st.rerun()


@st.dialog("Grammar labels", width="large")
def grammar_labels_dialog() -> None:
    labels = [
        f"**{escape_html(symbol)}** — {escape_html(label)}"
        for symbol, label in PLAIN_LABELS.items()
    ]
    grid = st.columns(3)
    for index, label in enumerate(labels):
        with grid[index % 3]:
            st.markdown(label)


@st.dialog("Word details", width="large")
def word_detail_dialog(word: str) -> None:
    render_dictionary_entry(word)
    if st.button("Close", type="primary", use_container_width=True):
        st.session_state.pop("word_card_word", None)
        st.rerun()


with st.sidebar:
    logo_col, title_col = st.columns([30, 70], gap="small")
    with logo_col:
        st.markdown('<span class="logo-anchor"></span>', unsafe_allow_html=True)
        st.button(
            "H",
            key="logo_button",
            help="Hiligaynon Grammar Checker — click for a surprise",
            on_click=trigger_logo_burst,
        )
    with title_col:
        st.markdown(
            '<div class="sidebar-title">Hiligaynon Grammar<br>Checker</div>',
            unsafe_allow_html=True,
        )
    rail = st.columns(3)
    with rail[0]:
        st.markdown('<span class="rail-mark rail-mark-0"></span>', unsafe_allow_html=True)
        st.button(
            "☼" if dark_mode else "☾",
            key="rail_theme",
            help="Switch to light mode" if dark_mode else "Switch to dark mode",
            on_click=toggle_theme,
            use_container_width=True,
        )
    with rail[1]:
        st.markdown('<span class="rail-mark rail-mark-1"></span>', unsafe_allow_html=True)
        st.button(
            "ⓘ",
            key="rail_how",
            help="How a check works",
            on_click=open_info,
            args=("how",),
            use_container_width=True,
        )
    with rail[2]:
        st.markdown('<span class="rail-mark rail-mark-2"></span>', unsafe_allow_html=True)
        st.button(
            "Aa",
            key="rail_labels",
            help="Grammar labels",
            on_click=open_info,
            args=("labels",),
            use_container_width=True,
        )
    st.divider()
    st.metric("Grammar vocabulary", f"{grammar_word_count:,}")
    st.metric("Lexicon entries", f"{len(lex.merged):,}")
    st.caption("Patterns are limited to the rules included in this project.")

pending_word = st.session_state.pop("word_card_pending", False)
pending_info = st.session_state.pop("info_dialog", None)
pending_burst = st.session_state.pop("logo_burst_pending", False)
if pending_burst:
    render_logo_burst()
if pending_word:
    word_detail_dialog(st.session_state.get("word_card_word", ""))
elif pending_info == "how":
    how_it_works_dialog()
elif pending_info == "labels":
    grammar_labels_dialog()


check_tab, build_tab, words_tab, dictionary_tab = st.tabs(
    ["Check", "Build", "Word bank", "Dictionary"]
)


with check_tab:
    left, right = st.columns([1.55, 1], gap="large")
    with left:
        st.subheader("Enter a sentence")
        st.caption("Include a final period. Accent marks are optional.")
        with st.form("sentence_check_form", clear_on_submit=False):
            sentence = st.text_area(
                "Sentence",
                value="naghampang ang tawo .",
                key="sentence_input",
                placeholder="e.g. naghampang ang tawo .",
                height=110,
                label_visibility="collapsed",
            )
            submitted = st.form_submit_button("Check sentence", type="primary", use_container_width=True)
    with right:
        st.subheader("Examples")
        examples = [
            "naghampang ang tawo .",
            "nagahampang sang mahinay ang tawo .",
            "si hans nagkaon .",
        ]
        for example in examples:
            st.button(example, key=f"example_{example}", on_click=set_example, args=(example,), use_container_width=True)
            st.caption("Choosing an example fills the sentence box and starts the grammar trace.")

    run_example = st.session_state.pop("run_check", False)
    if submitted or run_example:
        if sentence.strip():
            st.session_state["check_result"] = checker.check(sentence)
        else:
            st.warning("Enter a sentence first.")

    if "check_result" in st.session_state:
        st.divider()
        render_result(st.session_state["check_result"], "main")


with build_tab:
    render_section_banner("BUILD", "Choose a pattern and fill each slot with a word the grammar knows.")
    st.subheader("Build a sentence")
    st.caption("Choose a pattern, then fill each slot with a word the grammar knows.")

    template_names = [name for name, _ in TEMPLATES]
    choice = st.selectbox("Sentence pattern", template_names, key="builder_pattern")
    slots = dict(TEMPLATES)[choice]
    st.markdown("**Pattern slots**")
    st.markdown(" ".join(f'<span class="word-chip">{CATEGORY_META[letter][0]}</span>' for letter in slots), unsafe_allow_html=True)

    chosen = []
    slot_columns = st.columns(min(len(slots), 4), gap="medium")
    for index, letter in enumerate(slots):
        category = checker.categories[letter]
        options = [category.display[word] for word in category.words]
        with slot_columns[index % len(slot_columns)]:
            selected = st.selectbox(category.label, options, key=f"slot_{choice}_{index}")
        chosen.append(selected)

    built_sentence = " ".join(chosen)
    st.markdown("**Your sentence**")
    st.code(built_sentence, language=None)
    if st.button("Validate sentence", key="validate_built_sentence", type="primary"):
        st.session_state["build_result"] = checker.check(built_sentence)
    if "build_result" in st.session_state:
        st.divider()
        render_result(st.session_state["build_result"], "builder")


with words_tab:
    render_section_banner("WORD BANK", "Browse grammar words as quick meaning cards, then open a full dictionary entry.")
    st.subheader("Explore words")
    mode = st.radio("Browse", ["Grammar word classes", "Explore all words"], horizontal=True, label_visibility="collapsed")

    if mode == "Grammar word classes":
        st.caption("Choose a word class to see the words the grammar can recognize.")
        selected_letter = st.selectbox(
            "Word class",
            list(CATEGORY_META),
            format_func=lambda letter: f"{CATEGORY_META[letter][0]}  ·  {letter}",
            key="wordbank_category",
        )
        category = checker.categories[selected_letter]
        st.markdown(f"### {category.label}")
        st.caption(category.description)
        st.caption(f"{len(category.words):,} words in this class")

        if selected_letter == "b":
            with st.form("add_name_form", clear_on_submit=True):
                new_name = st.text_input("Add a name for this visit", placeholder="e.g. Maria")
                add_name = st.form_submit_button("Add name")
            if add_name and new_name.strip():
                if normalize(new_name.strip()) not in {normalize(name) for name in st.session_state.extra_names}:
                    st.session_state.extra_names.append(new_name.strip())
                    get_checker.clear()
                    st.rerun()
                else:
                    st.info("That name is already in your list.")

        word_query = st.text_input(
            "Find a word in this class",
            placeholder="Type a Hiligaynon word",
            key=f"wordbank_class_search_{selected_letter}",
        )
        filtered_words = [
            word for word in category.words
            if not word_query.strip() or normalize(category.display[word]).find(normalize(word_query.strip())) >= 0
        ]
        page_size = 12
        page_count = max(1, (len(filtered_words) + page_size - 1) // page_size)
        page = st.selectbox(
            "Page",
            range(1, page_count + 1),
            format_func=lambda number: f"{number} of {page_count}",
            key=f"wordbank_class_page_{selected_letter}",
        )
        page_words = filtered_words[(page - 1) * page_size : page * page_size]
        st.caption(f"Showing {len(page_words):,} of {len(filtered_words):,} matching words")
        card_columns = st.columns(2)
        for index, word in enumerate(page_words):
            with card_columns[index % 2]:
                render_word_card(
                    category.display[word],
                    key=f"wordbank_card_{selected_letter}_{page}_{index}",
                    pos=category.label,
                    source=category.source[word],
                )
    else:
        direction = st.radio(
            "Search direction",
            ["Hiligaynon → English", "English → Hiligaynon"],
            horizontal=True,
            key="wordbank_direction",
        )
        query = st.text_input("Search words or meanings", placeholder="Try tubig or water", key="wordbank_query")
        st.caption("Open a card to view its full dictionary entry in an overlay.")
        if query.strip():
            candidates = lex.search(query, direction="meaning" if direction.startswith("English") else "word")
            if candidates:
                st.caption(f"Showing {min(12, len(candidates))} of up to {len(candidates)} matches")
                card_columns = st.columns(2)
                for index, word in enumerate(candidates[:12]):
                    lookup = lex.lookup(word)["lexicon"]
                    pos = text_value(lookup.iloc[0]["pos_word"]) if len(lookup) else "Dictionary entry"
                    with card_columns[index % 2]:
                        render_word_card(word, key=f"wordbank_all_{index}", pos=pos)
            else:
                st.info("No matches yet. Try a shorter spelling or a different English meaning.")
        else:
            st.info("Search in either direction to browse matching words and meanings.")


with dictionary_tab:
    render_section_banner("DICTIONARY", "Look up Hiligaynon words or search English meanings across the local dictionaries.")
    st.subheader("Hiligaynon ↔ English dictionary")
    st.caption("Search in either direction. Entries combine the local lexicon with PinoyDictionary and Cecile Motus when available; accent marks are ignored.")
    st.radio(
        "Search direction",
        ["Hiligaynon → English", "English → Hiligaynon"],
        horizontal=True,
        key="dictionary_direction",
        on_change=clear_dictionary_selection,
    )
    query = st.text_input(
        "Word or meaning",
        placeholder="Try tubig or water",
        key="dictionary_query",
        on_change=clear_dictionary_selection,
    )

    if query.strip():
        direction = "meaning" if st.session_state.dictionary_direction.startswith("English") else "word"
        candidates = lex.search(query, direction=direction)
        normalized_query = normalize(query)
        if direction == "word" and any(normalized_query in words for words in checker.vocab.values()):
            candidates = list(dict.fromkeys([normalized_query, *candidates]))
        if candidates:
            selected_word = st.session_state.get("dictionary_selected_word")
            if selected_word not in candidates:
                selected_word = candidates[0]
            st.markdown("### Dictionary entry")
            render_dictionary_entry(selected_word)

            other_matches = [word for word in candidates if word != selected_word]
            if other_matches:
                st.markdown("#### More matches")
                st.caption("Choose a result to open its full entry.")
                result_columns = st.columns(2)
                for index, word in enumerate(other_matches[:10]):
                    lookup = lex.lookup(word)["lexicon"]
                    pos = text_value(lookup.iloc[0]["pos_word"]) if len(lookup) else "Dictionary entry"
                    with result_columns[index % 2]:
                        render_word_card(word, key=f"dictionary_match_{index}", pos=pos, open_in_dialog=False)
        else:
            st.warning(f"No dictionary entries matched “{escape_html(query)}”.")
            if direction == "word":
                suggestions = lex.suggest(query)
                if suggestions:
                    st.markdown("Try one of these Hiligaynon words:")
                    suggestion_columns = st.columns(min(3, len(suggestions)))
                    for index, word in enumerate(suggestions):
                        with suggestion_columns[index % len(suggestion_columns)]:
                            st.button(
                                word,
                                key=f"dictionary_suggestion_{index}",
                                on_click=set_dictionary_word,
                                args=(word,),
                                use_container_width=True,
                            )
    else:
        st.info("Start with a Hiligaynon word like tubig, or switch direction and search an English meaning like water.")
        quick_columns = st.columns(3)
        for index, word in enumerate(("tubig", "adlaw", "balay")):
            with quick_columns[index]:
                st.button(
                    word,
                    key=f"dictionary_quick_{word}",
                    on_click=set_dictionary_word,
                    args=(word,),
                    use_container_width=True,
                )
