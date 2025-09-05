# avails_bot.py — Smart vertical (game/games), Vertical_Truth, Jounce fixed, sums-after-filters, intents
import os, json, re, unicodedata, difflib, traceback
from typing import List, Optional, Any, Dict, Set

import pandas as pd
import streamlit as st
import openai

# ========= Config =========
USE_LLM = True  # set False to force heuristic parser for every prompt
st.set_page_config(page_title="Avails Bot — Smart Filters", layout="wide")
st.caption(f"🔧 Parser mode: **{'LLM mode' if USE_LLM else 'Heuristic mode'}**")

# OpenAI client (expects OPENAI_API_KEY in Streamlit secrets)
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ========= Canon & Hints =========
NUMERIC_HINTS = {
    "Monthly Traffic", "Ad Impressions Served", "Valid Wins",
    "Ad Impressions Rendered", "Total Burn",
    # eCPM computed after aggregation; not a base numeric hint
}
BOOL_LIKE_HINTS = {"Is Rewarded Slot","Coppa Enabled","Certified as Green Media"}
POLICY_COLUMNS = {
    "Gambling/Card Games","Wine/Cocktails & Beer","Medicinal Drugs/Alternative Medicines",
    "Dating","Politics","Sexuality","Cigars","Binance.com","Crypto.com","Coinbase.com","KuCoin.com",
}

COLUMN_ALIASES = {
    # Country / OS
    "country":"Country Name","country name":"Country Name",
    "os":"Operating System Name","operating system":"Operating System Name","operating system name":"Operating System Name",
    # App / publisher
    "app name":"Inmobi App Name","inmobi app name":"Inmobi App Name",
    "bundle":"Forwarded Bundle ID","bundle id":"Forwarded Bundle ID","package":"Forwarded Bundle ID",
    "publisher id":"Publisher Account GUID","publisher guid":"Publisher Account GUID","publisher":"Publisher Account Name",
    # Format / placement
    "format":"Placement Type","placement":"Placement Type","placement type":"Placement Type","final format":"Final Format",
    "rewarded":"Is Rewarded Slot","is rewarded":"Is Rewarded Slot",
    "integration":"Integration Method","integration method":"Integration Method",
    "content rating":"Content Rating Id",
    # Environment
    "inventory channel":"Inventory Channel","device type":"Device Type Name","device type name":"Device Type Name",
    # Policy shortcuts
    "gambling":"Gambling/Card Games","alcohol":"Wine/Cocktails & Beer","cannabis":"Medicinal Drugs/Alternative Medicines",
    "dating":"Dating","politics":"Politics","sexuality":"Sexuality","tobacco":"Cigars","crypto":"Crypto.com",
    # Categories
    "category":"Primary Category","primary category":"Primary Category","vertical":"Vertical","iab":"Primary Category",
    "url":"URL","jounce":"Jounce Media","green":"Certified as Green Media",
}

COUNTRY_CANON = {
    "usa":"united states","us":"united states","u.s.":"united states","u.s":"united states",
    "united states":"united states","united states of america":"united states",
    "uk":"united kingdom","united kingdom":"united kingdom",
}
COUNTRY_SYNONYMS = {"usa":"United States","us":"United States","u.s.":"United States","u.s":"United States","uk":"United Kingdom"}

# Aggregation grain
DEFAULT_GROUP_GRAIN = "app_os"  # "app_os" or "app_only"
APP_KEYS_APP_OS = [
    "Publisher Account GUID","Publisher Account Name","Publisher Account Type",
    "Inmobi App Inc ID","Inmobi App Name",
    "Forwarded Bundle ID","Operating System Name","URL",
]
APP_KEYS_APP_ONLY = [
    "Publisher Account GUID","Publisher Account Name","Publisher Account Type",
    "Inmobi App Inc ID","Inmobi App Name",
    "Forwarded Bundle ID","URL",
]
AGG_SUM_COLS = ["Monthly Traffic","Ad Impressions Rendered","Total Burn"]

# Default table columns (no eCPM by default)
DISPLAY_COLS_ORDER = [
    "Publisher Account GUID","Publisher Account Name","Publisher Account Type",
    "Inmobi App Inc ID","Inmobi App Name",
    "Forwarded Bundle ID","Operating System Name","URL",
    "Monthly Traffic","Ad Impressions Rendered","Total Burn",
]

CATEGORY_SYNONYMS = {
    "cars":"Automotive","auto":"Automotive","automobile":"Automotive",
    "beauty":"Beauty & Fitness","makeup":"Beauty & Fitness",
    "finance":"Finance","fintech":"Finance","food":"Food & Drink","restaurants":"Food & Drink",
    "travel":"Travel","education":"Education","health":"Health & Fitness","fitness":"Health & Fitness",
    "parenting":"Parenting","kids":"Parenting","sports":"Sports","news":"News","shopping":"Shopping","ecommerce":"Shopping",
    "entertainment":"Entertainment","music":"Music","video":"Video","productivity":"Productivity","business":"Business",
    "lifestyle":"Lifestyle","photography":"Photography","art":"Arts & Design","weather":"Weather","maps":"Maps & Navigation",
    "beauty & fitness":"Beauty & Fitness","food & drink":"Food & Drink",
}

# ========= Regex for vertical from text =========
GAMEY_PAT = re.compile(r"\bgame(s|r)?\b|\bgaming\b", re.I)
NON_GAMING_PAT = re.compile(r"\bnon[-_\s]*gaming\b", re.I)

# ========= Helpers =========
def clean_app_name(s: str) -> str:
    if pd.isna(s): return s
    s = unicodedata.normalize("NFKC", str(s))
    s = "".join(ch for ch in s if ch.isprintable())
    return re.sub(r"\s+"," ",s).strip()

def clean_header(h: Any) -> str:
    s = "" if h is None else str(h)
    s = unicodedata.normalize("NFKC", s)
    s = "".join(ch for ch in s if ch.isprintable())
    s = s.replace("\xa0", " ")
    s = re.sub(r"[\r\n\t]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def canonicalize_country_value(v: Any) -> str:
    s = str(v or "").strip().lower()
    return COUNTRY_CANON.get(s, s)

def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    for c in BOOL_LIKE_HINTS:
        if c in df.columns:
            df[c] = df[c].astype(str)
    for c in NUMERIC_HINTS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# ---- Vertical canonicalization ----
def canonicalize_vertical_value(v: Any) -> Optional[str]:
    s = str(v or "").lower()
    s = re.sub(r"[^a-z]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s: return None
    if s == "gaming": return "Gaming"
    if re.fullmatch(r"non\s*gaming", s): return "Non Gaming"
    return None

def normalize_vertical_column(df: pd.DataFrame) -> pd.DataFrame:
    if "Vertical" in df.columns:
        canon = df["Vertical"].map(canonicalize_vertical_value)
        df["Vertical"] = canon.fillna(df["Vertical"])
    return df

def apply_final_format(df: pd.DataFrame) -> pd.DataFrame:
    def derive_final_format(row) -> str:
        placement = str(row.get("Placement Type","")).strip().lower()
        rewarded = str(row.get("Is Rewarded Slot","")).strip().lower() in {"true","1","yes"}
        if placement == "banner":
            return "Banner"
        elif placement == "interstitial":
            return "Rewarded Video" if rewarded else "FSI"
        elif placement == "video":
            return "Rewarded Video" if rewarded else "API Video"
        elif placement == "native":
            return "Native"
        return "Unknown"
    df["Final Format"] = df.apply(derive_final_format, axis=1)
    return df

def derive_vertical(df: pd.DataFrame) -> pd.DataFrame:
    if "Vertical" in df.columns:
        return normalize_vertical_column(df)
    if "Primary Category" in df.columns:
        mask = df["Primary Category"].astype(str).str.contains(r"\bgam(e|es|ing)\b", case=False, na=False)
    elif "Inmobi App Categories" in df.columns:
        mask = df["Inmobi App Categories"].astype(str).str.contains(r"\bgam(e|es|ing)\b", case=False, na=False)
    else:
        df["Vertical"] = "Non Gaming"; return df
    df["Vertical"] = mask.map({True: "Gaming", False: "Non Gaming"})
    return df

def derive_vertical_from_categories(df: pd.DataFrame) -> pd.Series:
    cat_hit = pd.Series(False, index=df.index)
    if "Primary Category" in df.columns:
        cat_hit = cat_hit | df["Primary Category"].astype(str).str.contains(r"\bgam
