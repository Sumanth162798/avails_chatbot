# avails_bot.py — Smart vertical truth, Jounce fixed, video-supply override, sums-after-filters, intents
import os, json, re, unicodedata, difflib, traceback
from typing import List, Optional, Any, Dict, Set

import pandas as pd
import streamlit as st
import openai

# ========= Config =========
USE_LLM = True  # set False to force heuristic parser for every prompt
st.set_page_config(page_title="Avails Bot (Beta Testing version)", layout="wide")
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

# ========= Regex cues =========
GAMEY_PAT = re.compile(r"\bgame(s|r)?\b|\bgaming\b", re.I)
NON_GAMING_PAT = re.compile(r"\bnon[-_\s]*gaming\b", re.I)
VIDEO_SUPPLY_PAT = re.compile(r"\bvideo\s+supply\b|\bvideo\b.*\bsupply\b", re.I)

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
        cat_hit = cat_hit | df["Primary Category"].astype(str).str.contains(r"\bgam(e|es|ing)\b", case=False, na=False)
    if "Inmobi App Categories" in df.columns:
        cat_hit = cat_hit | df["Inmobi App Categories"].astype(str).str.contains(r"\bgam(e|es|ing)\b", case=False, na=False)
    return cat_hit.map({True: "Gaming", False: "Non Gaming"})

def compute_vertical_truth(df: pd.DataFrame) -> pd.DataFrame:
    v = df["Vertical"] if "Vertical" in df.columns else pd.Series(index=df.index, dtype="object")
    v = v.map(canonicalize_vertical_value).fillna(v)
    cat_v = derive_vertical_from_categories(df)
    need_cat = v.isna() | ~v.isin(["Gaming","Non Gaming"]) | (cat_v.eq("Gaming") & ~v.eq("Gaming"))
    v_truth = v.where(~need_cat, cat_v)
    df["Vertical_Truth"] = v_truth.fillna(cat_v)
    return df

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    cols_lower = {clean_header(c).lower(): c for c in df.columns}
    for alias, target in COLUMN_ALIASES.items():
        if target not in df.columns and alias in cols_lower:
            rename_map[cols_lower[alias]] = target
    for c in df.columns:
        if c in rename_map: continue
        lc = clean_header(c).lower()
        if re.fullmatch(r"(os|operating\s*system(\s*name)?)", lc) and "Operating System Name" not in df.columns:
            rename_map[c] = "Operating System Name"; continue
        if ("bundle" in lc or "package" in lc) and "id" in lc and "Forwarded Bundle ID" not in df.columns:
            rename_map[c] = "Forwarded Bundle ID"; continue
        if re.fullmatch(r"(content\s*rating(\s*id)?)", lc) and "Content Rating Id" not in df.columns:
            rename_map[c] = "Content Rating Id"; continue
    if rename_map: df = df.rename(columns=rename_map)
    return df

# ========= Region / load =========
def detect_region(prompt: str) -> str:
    p = (prompt or "").lower()
    if any(t in p for t in ["global","worldwide","all regions"]): return "GLOBAL"
    na_terms = ["us","usa","united states","canada","north america","na"]
    apac_terms = ["india","indonesia","philippines","vietnam","apac","thailand","sg","singapore","malaysia"]
    if any(t in p for t in na_terms): return "NA"
    if any(t in p for t in apac_terms): return "APAC"
    return "NA"

@st.cache_data
def load_df(region: str) -> pd.DataFrame:
    paths = []
    if region in ("NA","GLOBAL"):  paths.append("na_avails.xlsx")
    if region in ("APAC","GLOBAL"): paths.append("apac_avails.xlsx")
    frames = []
    for path in paths:
        df = pd.read_excel(path, engine="openpyxl")
        df.columns = [clean_header(c) for c in df.columns]
        df = normalize_columns(df)
        if "Inmobi App Name" in df.columns:
            df["Inmobi App Name"] = df["Inmobi App Name"].apply(clean_app_name)
        df = apply_final_format(df)
        df = derive_vertical(df)
        df = normalize_vertical_column(df)
        df = coerce_types(df)
        df = compute_vertical_truth(df)  # Vertical_Truth at load
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

# ========= Categories =========
def build_category_vocab(df: pd.DataFrame) -> List[str]:
    vocab: Set[str] = set()
    if "Primary Category" in df.columns:
        vocab.update([str(x).strip() for x in df["Primary Category"].dropna().unique()])
    if "Inmobi App Categories" in df.columns:
        for s in df["Inmobi App Categories"].dropna().astype(str):
            for piece in re.split(r"[;,|/]", s):
                piece = piece.strip()
                if piece: vocab.add(piece)
    vocab = {v for v in vocab if v and v.lower() != "nan"}
    return sorted(vocab)

def map_categories_with_llm(raw_terms: List[str], vocab: List[str]) -> List[str]:
    if not raw_terms: return []
    try:
        payload = {"instruction": "Map user terms to the closest items in CATEGORY_VOCAB. Return JSON array of exact vocab strings.",
                   "user_terms": raw_terms, "CATEGORY_VOCAB": vocab[:400]}
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role":"system","content":"Return only a JSON array or an object with key 'mapped'."},
                      {"role":"user","content":json.dumps(payload)}],
            temperature=0.0,
            response_format={"type":"json_object"},
        )
        raw = (resp.choices[0].message.content or "").strip()
        obj = json.loads(raw)
        out = obj if isinstance(obj, list) else obj.get("mapped", [])
        return [x for x in out if x in vocab]
    except Exception:
        return []

def fuzzy_map_categories(raw_terms: List[str], vocab: List[str]) -> List[str]:
    mapped: List[str] = []
    for t in raw_terms:
        t_norm = t.strip().lower()
        t_syn = CATEGORY_SYNONYMS.get(t_norm)
        if t_syn and t_syn in vocab:
            mapped.append(t_syn); continue
        best = difflib.get_close_matches(t, vocab, n=1, cutoff=0.6)
        if best: mapped.append(best[0])
    seen = set(); out = []
    for x in mapped:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

def resolve_categories(raw_terms: List[str], df: pd.DataFrame) -> List[str]:
    vocab = build_category_vocab(df)
    if not raw_terms: return []
    llm = map_categories_with_llm(raw_terms, vocab)
    if llm: return llm
    return fuzzy_map_categories(raw_terms, vocab)

# ========= Parser (LLM + fallback) =========
LLM_SYSTEM = """You are a filter compiler for an avails dataset.
Return ONLY a JSON object with keys:
- filters: array of {column, op, value?}, op in ["equals","contains","in","not_in","gte","lte","gt","lt","is_true","is_false","allowed","blocked"]
- macros: optional array from ["premium","green","local_apps"]
- thresholds: optional {"min_requests": int, "min_rendered": int, "min_burn": number}
- final_format: optional in ["Banner","FSI","Native","Rewarded Video","API Video"]
- categories_raw: optional array of user category terms
- include_cols: optional array of column names to include in the final display
- intent: optional in ["list","count","breakdown","summary"]
Guidance:
- "USA"/"US" => "United States" for column "Country Name".
- "SDK integration" => {"column":"Integration Method","op":"equals","value":"SDK"}.
- "rewarded video" => final_format "Rewarded Video" and/or {"column":"Is Rewarded Slot","op":"is_true"}.
- If text says "gaming"/"game(s)/gamer", add a Vertical filter = "Gaming", unless it says "non-gaming".
- If "premium"/"safe", add macro "premium".
- If "green media", add macro "green".
- "local apps" => macro "local_apps".
- Categories must match against "Primary Category" or "Inmobi App Categories" (NOT Vertical).
Return compact JSON. No prose.
"""

COUNT_PAT = re.compile(r"\bhow many\b|\bcount\b|\bnumber of\b|\bunique\b", re.I)
BREAKDOWN_PAT = re.compile(r"\bbreak\s*down\b|\bby os\b|\bby country\b|\bby category\b|\bsplit by\b", re.I)
SUMMARY_PAT = re.compile(r"\bsummary\b|\bsummarize\b|\btl;dr\b|\binsights?\b", re.I)

def detect_intent(user_text: str, qobj: Dict[str, Any]) -> str:
    intent = (qobj or {}).get("intent")
    if intent in {"count","summary","list","breakdown"}:
        return intent
    t = (user_text or "")
    if COUNT_PAT.search(t): return "count"
    if BREAKDOWN_PAT.search(t): return "breakdown"
    if SUMMARY_PAT.search(t): return "summary"
    return "list"

def heuristic_query(prompt: str) -> Dict[str, Any]:
    p = (prompt or "").lower()
    out = {"filters": [], "macros": [], "thresholds": {}, "categories_raw": [], "include_cols": [], "intent": "list"}
    if any(k in p for k in ["usa","us","united states"]):
        out["filters"].append({"column":"Country Name","op":"equals","value":"United States"})
    if "india" in p: out["filters"].append({"column":"Country Name","op":"equals","value":"India"})
    if "uk" in p or "united kingdom" in p: out["filters"].append({"column":"Country Name","op":"equals","value":"United Kingdom"})
    if "android" in p: out["filters"].append({"column":"Operating System Name","op":"equals","value":"Android"})
    if "ios" in p or "iphone" in p: out["filters"].append({"column":"Operating System Name","op":"equals","value":"iOS"})
    if "sdk" in p: out["filters"].append({"column":"Integration Method","op":"equals","value":"SDK"})
    if "rewarded" in p: out["filters"].append({"column":"Is Rewarded Slot","op":"is_true"})
    if "banner" in p: out["filters"].append({"column":"Final Format","op":"equals","value":"Banner"})
    if "interstitial" in p or "fullscreen" in p or "fsi" in p:
        out["filters"].append({"column":"Final Format","op":"equals","value":"FSI"})
    if "native" in p: out["filters"].append({"column":"Final Format","op":"equals","value":"Native"})
    if "video" in p and "rewarded" not in p:
        # legacy heuristic; will be overridden by video supply override if "video supply" is asked
        out["filters"].append({"column":"Final Format","op":"equals","value":"API Video"})
    if NON_GAMING_PAT.search(p):
        out["filters"].append({"column":"Vertical","op":"equals","value":"Non Gaming"})
    elif GAMEY_PAT.search(p):
        out["filters"].append({"column":"Vertical","op":"equals","value":"Gaming"})
    if "premium" in p or "safe" in p: out["macros"].append("premium")
    if "green" in p: out["macros"].append("green")
    if "local" in p: out["macros"].append("local_apps")
    for k in CATEGORY_SYNONYMS.keys():
        if k in p: out["categories_raw"].append(k)
    if COUNT_PAT.search(p): out["intent"] = "count"
    if BREAKDOWN_PAT.search(p): out["intent"] = "breakdown"
    if SUMMARY_PAT.search(p): out["intent"] = "summary"
    inc = re.findall(r"(?:with|including|along with|also show|show)\s+([a-z0-9 _/&-]+)", p)
    out["include_cols"] = [x.strip() for x in inc]
    return out

def extract_query_with_errors(prompt: str) -> Dict[str, Any]:
    info = {"parsed": None, "error": None, "raw": None, "request_id": None}
    if not USE_LLM:
        info["parsed"] = heuristic_query(prompt); return info
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role":"system","content": LLM_SYSTEM},
                      {"role":"user","content": prompt}],
            temperature=0.1,
            response_format={"type":"json_object"},
        )
        info["request_id"] = getattr(resp, "id", None)
        raw = (resp.choices[0].message.content or "").strip()
        info["raw"] = raw
        if not raw: raise ValueError("Empty LLM response")
        info["parsed"] = json.loads(raw)
        return info
    except Exception as e:
        info["error"] = f"{e.__class__.__name__}: {e}\n{traceback.format_exc()}"
        info["parsed"] = heuristic_query(prompt)
        return info

# ========= Filtering / macros / agg =========
def alias_to_canonical(col: str) -> str:
    key = (col or "").strip().lower()
    return COLUMN_ALIASES.get(key, col)

def apply_one_filter(df: pd.DataFrame, column: str, op: str, value: Any = None) -> pd.DataFrame:
    col = alias_to_canonical(column)
    if col not in df.columns:
        return df

    # Normalize for Vertical & Country
    if col == "Vertical" and op == "equals":
        v_can = canonicalize_vertical_value(value)
        if v_can: value = v_can

    if op in {"allowed","blocked"} and col in POLICY_COLUMNS:
        s = df[col].astype(str).str.lower()
        truthy = s.str.contains("allow|allowed|yes|true|1|clean", na=False)
        falsy  = s.str.contains("block|blocked|no|false|0|deny|restricted", na=False)
        return df[truthy] if op == "allowed" else df[falsy]

    series = df[col].astype(str)

    if op == "equals":
        if pd.api.types.is_numeric_dtype(df[col]):
            return df[df[col] == pd.to_numeric(value, errors="coerce")]
        if col == "Vertical":
            series_norm = df[col].map(canonicalize_vertical_value).fillna(series)
            return df[series_norm.str.casefold() == str(value).casefold()]
        if col == "Country Name":
            series_norm = df[col].apply(canonicalize_country_value)
            value_norm = canonicalize_country_value(value)
            return df[series_norm == value_norm]
        return df[series.str.casefold() == str(value).casefold()]

    if op == "contains":
        return df[series.str.contains(str(value), case=False, na=False)]

    if op in {"in","not_in"}:
        vals = value if isinstance(value, list) else [value]
        vals_lc = set(str(v).casefold() for v in vals)
        mask = series.str.casefold().isin(vals_lc)
        return df[mask] if op == "in" else df[~mask]

    if op in {"gte","lte","gt","lt"}:
        s = pd.to_numeric(df[col], errors="coerce")
        v = pd.to_numeric(value, errors="coerce")
        if pd.isna(v): return df
        if op == "gte": return df[s >= v]
        if op == "lte": return df[s <= v]
        if op == "gt":  return df[s > v]
        if op == "lt":  return df[s < v]

    if op == "is_true":
        return df[series.str.contains("true|1|yes", case=False, na=False)]
    if op == "is_false":
        return df[~series.str.contains("true|1|yes", case=False, na=False)]
    return df

def _norm_jounce(s: str) -> str:
    s = str(s or "").lower().strip()
    s = re.sub(r"[\s/_-]+", " ", s)
    return s

BAD_JOUNCE = {
    "cheap reach",
    "made for advertising",
    "made for ads",
    "made for advertizing",
    "mfa",
    "untrusted",
    "untrsuted",  # common typo
}

def apply_macros(df: pd.DataFrame, macros: List[str]) -> pd.DataFrame:
    d = df
    macros = macros or []
    if "premium" in macros:
        # Jounce rule: drop ONLY bad classes; blanks are allowed/kept
        if "Jounce Media" in d.columns:
            s = d["Jounce Media"].astype(str).map(_norm_jounce)
            bad_mask = s.isin(BAD_JOUNCE) | s.str.contains(r"\bmfa\b", na=False)
            d = d[~bad_mask]
        # (Optional stricter toggles can be enabled later: require SDK, exclude MA)

    if "green" in macros:
        green_cols = ["Certified as Green Media","Scope3","Scope 3","Scope3 Certified","Scope 3 Certified"]
        any_mask = None
        for gc in green_cols:
            if gc in d.columns:
                s = d[gc].astype(str).str.lower()
                m = s.str.contains(r"true|yes|1|green|certified|scope\s*3", na=False)
                any_mask = m if any_mask is None else (any_mask | m)
        if any_mask is not None:
            d = d[any_mask]

    if "local_apps" in macros and {"Publisher Origin Country Name","Country Name"} <= set(d.columns):
        d = d[d["Publisher Origin Country Name"].astype(str).str.lower() == d["Country Name"].astype(str).str.lower()]

    return d

def aggregate_app_level(df: pd.DataFrame, grain: str = DEFAULT_GROUP_GRAIN) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    group_cols = APP_KEYS_APP_OS if grain == "app_os" else APP_KEYS_APP_ONLY
    group_cols = [c for c in group_cols if c in df.columns]
    if not group_cols:
        return pd.DataFrame()
    agg_map = {c: "sum" for c in AGG_SUM_COLS if c in df.columns}
    g = (df.groupby(group_cols, dropna=False).agg(agg_map).reset_index())
    # eCPM from aggregated sums (hidden unless user asks)
    if {"Total Burn","Ad Impressions Rendered"} <= set(g.columns):
        denom = g["Ad Impressions Rendered"].replace(0, pd.NA)
        g["eCPM"] = (g["Total Burn"] * 1000.0) / denom
    return g

def normalize_include_cols(include_cols: List[str], df: pd.DataFrame) -> List[str]:
    if not include_cols: return []
    known = set(df.columns); out = []
    for raw in include_cols:
        cand = raw.strip()
        if not cand: continue
        if cand in known: out.append(cand); continue
        alias = alias_to_canonical(cand)
        if alias in known: out.append(alias); continue
        best = difflib.get_close_matches(cand, list(known), n=1, cutoff=0.7)
        if best: out.append(best[0])
    seen = set(); deduped = []
    for c in out:
        if c not in seen:
            seen.add(c); deduped.append(c)
    return deduped

# ========= Conversational answer helpers =========
def answer_count(g: pd.DataFrame):
    total_bundles = g["Forwarded Bundle ID"].nunique() if "Forwarded Bundle ID" in g.columns else 0
    if "Operating System Name" in g.columns:
        an = g[g["Operating System Name"].astype(str).str.casefold()=="android"]["Forwarded Bundle ID"].nunique()
        io = g[g["Operating System Name"].astype(str).str.casefold()=="ios"]["Forwarded Bundle ID"].nunique()
        st.success(f"Unique bundles — Total: **{total_bundles}** | Android: **{an}** | iOS: **{io}**")
    else:
        st.success(f"Unique bundles — Total: **{total_bundles}**")

def _uniq_by(g: pd.DataFrame, dim: str, topn: int = 10) -> pd.DataFrame:
    if dim not in g.columns or "Forwarded Bundle ID" not in g.columns:
        return pd.DataFrame()
    out = (g.groupby(dim)["Forwarded Bundle ID"]
             .nunique().sort_values(ascending=False).head(topn)
             .reset_index(name="Unique Bundles"))
    return out

def answer_breakdown(g: pd.DataFrame):
    # Prefer OS, else Country, else Final Format
    for dim in ["Operating System Name","Country Name","Final Format"]:
        out = _uniq_by(g, dim)
        if not out.empty:
            st.write(f"**Bundles by {dim} (unique Forwarded Bundle ID):**")
            st.dataframe(out, use_container_width=True)
            return
    st.info("No obvious breakdown dimensions available.")

def answer_summary(g: pd.DataFrame):
    rows = len(g)
    uniq = g["Forwarded Bundle ID"].nunique() if "Forwarded Bundle ID" in g.columns else 0
    mt  = int(g["Monthly Traffic"].sum()) if "Monthly Traffic" in g.columns else 0
    burn = float(g["Total Burn"].sum()) if "Total Burn" in g.columns else 0.0
    bits = [f"rows: **{rows}**", f"unique bundles: **{uniq}**"]
    if mt: bits.append(f"Monthly Traffic Σ: **{mt:,}**")
    if burn: bits.append(f"Burn Σ: **${burn:,.2f}**")
    st.success(" | ".join(bits))
    top_cols = [c for c in ["Monthly Traffic","Total Burn","Ad Impressions Rendered"] if c in g.columns]
    if top_cols:
        small = g.sort_values(top_cols, ascending=[False]*len(top_cols)).head(20)
        keep = [c for c in ["Inmobi App Name","Forwarded Bundle ID","Operating System Name"] if c in small.columns]
        keep += [c for c in top_cols if c in small.columns]
        st.dataframe(small[keep], use_container_width=True)

# ========= UI =========
st.title("Avails Bot (Beta Testing version)")

st.sidebar.header("Run options")
safe_mode = st.sidebar.checkbox("Safe mode (ignore AI & thresholds)", value=False)
grain = st.sidebar.selectbox("Aggregation grain", ["app_os","app_only"], index=0,
                             help="app_os = split Android/iOS; app_only = combine OS per bundle")
rewarded_mode = st.sidebar.selectbox(
    "Rewarded matching", ["slot_or_final","strict (Final Format)","slot_only"], index=0,
    help="How to match 'rewarded' asks."
)
if st.sidebar.button("🔁 Clear cached data"):
    st.cache_data.clear(); st.experimental_rerun()

user_input = st.text_input("Ask e.g. 'Game apps in India' or 'Video supply in Canada' or 'Premium rewarded gaming apps in US on SDK, Android; include Primary Category'")

# Debug collector
def _dbg_count(label: str, d: pd.DataFrame):
    st.session_state.setdefault("_debug_counts", [])
    st.session_state["_debug_counts"].append((label, len(d)))
    return d

if user_input:
    region = detect_region(user_input)
    df = load_df(region)
    st.caption(f"Region detected: **{region}**")

    with st.expander("Debug: key counts"):
        if "Final Format" in df.columns: st.write("Final Format:", df["Final Format"].value_counts(dropna=False))
        if "Integration Method" in df.columns: st.write("Integration Method:", df["Integration Method"].value_counts(dropna=False))
        if "Country Name" in df.columns: st.write("Top countries:", df["Country Name"].value_counts(dropna=False).head(20))
        if "Vertical_Truth" in df.columns: st.write("Vertical_Truth:", df["Vertical_Truth"].value_counts(dropna=False))

    required_for_group = ["Inmobi App Name","Forwarded Bundle ID","Operating System Name"]
    missing = [c for c in required_for_group if c not in df.columns]
    if missing:
        st.error(f"Missing required columns for grouping: {missing}. Check your Excel headers or update COLUMN_ALIASES.")

    if safe_mode:
        st.info("Safe mode ON — unfiltered, aggregated app list.")
        g = aggregate_app_level(df.copy(), grain=grain)
        for _col, _default in [("Monthly Traffic",0),("Ad Impressions Rendered",0),("Total Burn",0.0)]:
            if _col not in g.columns: g[_col] = _default
        if g.empty:
            st.error("Aggregation returned empty. Check group keys exist.")
        else:
            sort_cols = [c for c in ["Monthly Traffic","Total Burn","Ad Impressions Rendered"] if c in g.columns]
            if sort_cols: g = g.sort_values(sort_cols, ascending=[False]*len(sort_cols))
            base_cols = [c for c in DISPLAY_COLS_ORDER if c in g.columns]
            st.dataframe(g[base_cols].head(500), use_container_width=True)
            st.download_button("Download CSV", g.to_csv(index=False).encode("utf-8"), "avails_app_level.csv", "text/csv")
        st.stop()

    # Parse
    parse_info = extract_query_with_errors(user_input)
    q = parse_info.get("parsed") or {}

    # --- Hard-enforce vertical from text (robust) ---
    t = (user_input or "")
    q["filters"] = [f for f in (q.get("filters") or []) if (f.get("column","").strip().lower() != "vertical")]
    if NON_GAMING_PAT.search(t):
        q.setdefault("filters", []).append({"column":"Vertical","op":"equals","value":"Non Gaming"})
    elif GAMEY_PAT.search(t):
        q.setdefault("filters", []).append({"column":"Vertical","op":"equals","value":"Gaming"})
    # Macros via text cues
    macs = set(q.get("macros") or [])
    if re.search(r"\bpremium\b|\bsafe\b", t, re.I): macs.add("premium")
    if re.search(r"\bgreen\b", t, re.I): macs.add("green")
    if re.search(r"\blocal apps?\b", t, re.I): macs.add("local_apps")
    q["macros"] = list(macs)
    # Rewarded cue
    if re.search(r"\brewarded( video)?\b", t, re.I):
        q["final_format"] = "Rewarded Video"
        fs = q.get("filters", [])
        if not any((f.get("column","").strip().lower()=="is rewarded slot") for f in fs):
            fs.append({"column":"Is Rewarded Slot","op":"is_true"})
        q["filters"] = fs

    # --- Video supply override (Placement Type = Interstitial or Video) ---
    video_supply_mode = False
    t_lower = (user_input or "").lower()
    if VIDEO_SUPPLY_PAT.search(user_input or "") and not re.search(r"\brewarded\b", t_lower):
        video_supply_mode = True
        # remove any Final Format filter the parser/heuristics might have added
        q["filters"] = [f for f in (q.get("filters") or []) if (f.get("column","").strip().lower() != "final format")]
        # add Placement Type IN {Interstitial, Video}
        q["filters"].append({"column": "Placement Type", "op": "in", "value": ["Interstitial","Video"]})

    with st.expander("Parsed query (debug)"):
        st.json(q)
    with st.expander("LLM errors & raw (if any)"):
        if parse_info.get("request_id"): st.write(f"request_id: {parse_info['request_id']}")
        if parse_info.get("error"): st.error(parse_info["error"])
        if parse_info.get("raw"): st.code(parse_info["raw"], language="json")

    # Start filtering
    d = _dbg_count("start", df.copy())

    # Vertical first — use Vertical_Truth instead of raw Vertical
    d = compute_vertical_truth(d)  # idempotent if already present
    for filt in q.get("filters", []):
        if (filt.get("column") or "").strip().lower() == "vertical":
            val = filt.get("value")
            d = d[d["Vertical_Truth"].astype(str).str.casefold() == str(val).casefold()]
    d = _dbg_count("after vertical (truth)", d)

    # Final format / rewarded mode
    ff = q.get("final_format")
    if ff and "Final Format" in d.columns:
        if re.search(r"\brewarded", str(ff), re.I):
            if rewarded_mode == "strict (Final Format)":
                d = d[d["Final Format"].astype(str).str.casefold().eq("rewarded video")]
            elif rewarded_mode == "slot_only":
                if "Is Rewarded Slot" in d.columns:
                    d = d[d["Is Rewarded Slot"].astype(str).str.contains("true|1|yes", case=False, na=False)]
            else:  # slot_or_final (default)
                mask = d["Final Format"].astype(str).str.casefold().eq("rewarded video")
                if "Is Rewarded Slot" in d.columns:
                    mask = mask | d["Is Rewarded Slot"].astype(str).str.contains("true|1|yes", case=False, na=False)
                d = d[mask]
        else:
            d = apply_one_filter(d, "Final Format", "equals", ff)
    d = _dbg_count("after final_format/rewarded", d)

    # General filters (non-Vertical)
    for filt in q.get("filters", []):
        col = (filt.get("column") or "").strip()
        if col.lower() == "vertical": continue
        d = apply_one_filter(d, col or "", filt.get("op") or "equals", filt.get("value"))
    d = _dbg_count("after general filters", d)

    # Categories (if any)
    cat_terms = q.get("categories_raw", []) or q.get("categories", [])
    mapped_cats = resolve_categories(cat_terms, df) if cat_terms else []
    if mapped_cats:
        if "Primary Category" in d.columns:
            mask = d["Primary Category"].astype(str).isin(mapped_cats)
            if not mask.any() and "Inmobi App Categories" in d.columns:
                patt = "|".join(map(re.escape, mapped_cats))
                mask = d["Inmobi App Categories"].astype(str).str.contains(patt, case=False, na=False)
            d = d[mask]
        elif "Inmobi App Categories" in d.columns:
            patt = "|".join(map(re.escape, mapped_cats))
            d = d[d["Inmobi App Categories"].astype(str).str.contains(patt, case=False, na=False)]
    d = _dbg_count("after categories", d)

    # Macros (premium/green/local)
    d = apply_macros(d, q.get("macros", []))
    d = _dbg_count("after macros (premium/green/local)", d)

    # Aggregate (SUMS)
    d = _dbg_count("before aggregation", d)
    g = aggregate_app_level(d, grain=grain)
    st.session_state["_debug_counts"].append(("after aggregation (groups)", len(g)))

    # Ensure defaults present
    for _col, _default in [("Monthly Traffic",0),("Ad Impressions Rendered",0),("Total Burn",0.0)]:
        if _col not in g.columns: g[_col] = _default

    # Thresholds (post-agg)
    st.sidebar.header("Thresholds (post-aggregation)")
    vol_col = "Monthly Traffic" if "Monthly Traffic" in g.columns else None
    min_vol = st.sidebar.number_input(f"Min {vol_col or 'Volume'}", 0, 1_000_000_000, 0, 1000) if vol_col else 0
    min_rend = st.sidebar.number_input("Min Ad Impressions Rendered", 0, 1_000_000_000, 0, 1000)
    min_burn = st.sidebar.number_input("Min Total Burn ($)", 0.0, 1e12, 0.0, 1.0)
    if vol_col and min_vol: g = g[g[vol_col] >= min_vol]
    if "Ad Impressions Rendered" in g.columns and min_rend: g = g[g["Ad Impressions Rendered"] >= min_rend]
    if "Total Burn" in g.columns and min_burn: g = g[g["Total Burn"] >= min_burn]

    # Sort for display
    sort_cols = [c for c in ["Monthly Traffic","Total Burn","Ad Impressions Rendered"] if c in g.columns]
    if sort_cols: g = g.sort_values(sort_cols, ascending=[False]*len(sort_cols))

    # Zero-result handling
    if g.empty:
        with st.expander("Why no results? (row counts per gate)"):
            for lbl, n in st.session_state.get("_debug_counts", []):
                st.write(f"{lbl}: {n}")
        st.warning("No results. Try removing 'premium', relaxing rewarded mode, or lowering thresholds.")
        st.stop()

    # Intent handling
    intent = detect_intent(user_input, q)
    if intent == "count":
        if video_supply_mode:
            st.info("Note: 65% of request ad slots for **FSI** inventory on exchange are **video-ready** (based on the requested ad format). "
                    "Normalization on this basis is applied today; an exact breakdown is coming in the next versions.")
        answer_count(g)
        st.download_button("Download CSV", g.to_csv(index=False).encode("utf-8"), "avails_app_level.csv", "text/csv")
        st.stop()
    if intent == "breakdown":
        if video_supply_mode:
            st.info("Note: 65% of request ad slots for **FSI** inventory on exchange are **video-ready** (based on the requested ad format). "
                    "Normalization on this basis is applied today; an exact breakdown is coming in the next versions.")
        answer_breakdown(g)
        st.download_button("Download CSV", g.to_csv(index=False).encode("utf-8"), "avails_app_level.csv", "text/csv")
        st.stop()
    if intent == "summary":
        if video_supply_mode:
            st.info("Note: 65% of request ad slots for **FSI** inventory on exchange are **video-ready** (based on the requested ad format). "
                    "Normalization on this basis is applied today; an exact breakdown is coming in the next versions.")
        answer_summary(g)
        st.download_button("Download CSV", g.to_csv(index=False).encode("utf-8"), "avails_app_level.csv", "text/csv")
        st.stop()

    # Include extra columns on demand (incl. eCPM if explicitly asked)
    include_cols = normalize_include_cols(q.get("include_cols", []), g)
    if any(re.search(r"\becpm\b", str(x), re.I) for x in q.get("include_cols", [])):
        if "eCPM" not in include_cols and "eCPM" in g.columns:
            include_cols.append("eCPM")

    # Render table
    base_cols = [c for c in DISPLAY_COLS_ORDER if c in g.columns]
    show_cols = base_cols + [c for c in include_cols if c not in base_cols and c in g.columns]
    st.success(f"Apps matched: {len(g)}")
    if video_supply_mode:
        st.info("Note: 65% of request ad slots for **FSI** inventory on exchange are **video-ready** (based on the requested ad format). "
                "Normalization on this basis is applied today; an exact breakdown is coming in the next versions.")
    st.dataframe(g[show_cols].head(500), use_container_width=True)
    st.download_button("Download CSV", g.to_csv(index=False).encode("utf-8"), "avails_app_level.csv", "text/csv")

    with st.expander("Why no results? (row counts per gate)"):
        for lbl, n in st.session_state.get("_debug_counts", []):
            st.write(f"{lbl}: {n}")

    with st.expander("Available columns in dataset"):
        st.write(sorted(df.columns.tolist()))
