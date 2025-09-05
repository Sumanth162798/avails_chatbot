# avails_bot.py
import os, json, re, unicodedata, difflib, traceback
from typing import List, Optional, Any, Dict, Set

import pandas as pd
import streamlit as st
import openai

# =========================
# Config / Constants
# =========================
USE_LLM = True  # set False to force heuristic parser for every prompt

st.set_page_config(page_title="Avails Bot — Smart Filters", layout="wide")
mode_badge = "LLM mode" if USE_LLM else "Heuristic mode"
st.caption(f"🔧 Parser mode: **{mode_badge}**")

# OpenAI client (expects OPENAI_API_KEY in Streamlit secrets)
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# NOTE: No "Valid Ad Request" anywhere
# (Removed eCPM from NUMERIC_HINTS so it's not a "necessary" column.)
NUMERIC_HINTS = {
    "Monthly Traffic",
    "Ad Impressions Served",
    "Valid Wins",
    "Ad Impressions Rendered",
    "Total Burn",
    # "eCPM",  # ← no longer treated as a base/numeric hint
}

BOOL_LIKE_HINTS = {
    "Is Rewarded Slot",
    "Coppa Enabled",
    "Certified as Green Media",
}

POLICY_COLUMNS = {
    "Gambling/Card Games",
    "Wine/Cocktails & Beer",
    "Medicinal Drugs/Alternative Medicines",
    "Dating",
    "Politics",
    "Sexuality",
    "Cigars",
    "Binance.com",
    "Crypto.com",
    "Coinbase.com",
    "KuCoin.com",
}

# natural aliases → real columns (no Valid Ad Request aliases)
COLUMN_ALIASES = {
    # Country / OS
    "country": "Country Name",
    "country name": "Country Name",
    "os": "Operating System Name",
    "operating system": "Operating System Name",
    "operating system name": "Operating System Name",

    # App / publisher
    "app name": "Inmobi App Name",
    "inmobi app name": "Inmobi App Name",
    "bundle": "Forwarded Bundle ID",
    "bundle id": "Forwarded Bundle ID",
    "package": "Forwarded Bundle ID",
    "publisher id": "Publisher Account GUID",
    "publisher guid": "Publisher Account GUID",
    "publisher": "Publisher Account Name",

    # Format / placement
    "format": "Placement Type",
    "placement": "Placement Type",
    "placement type": "Placement Type",
    "final format": "Final Format",
    "rewarded": "Is Rewarded Slot",
    "is rewarded": "Is Rewarded Slot",
    "integration": "Integration Method",
    "integration method": "Integration Method",
    "content rating": "Content Rating Id",

    # Environment
    "inventory channel": "Inventory Channel",
    "device type": "Device Type Name",
    "device type name": "Device Type Name",

    # Policy shortcuts
    "gambling": "Gambling/Card Games",
    "alcohol": "Wine/Cocktails & Beer",
    "cannabis": "Medicinal Drugs/Alternative Medicines",
    "dating": "Dating",
    "politics": "Politics",
    "sexuality": "Sexuality",
    "tobacco": "Cigars",
    "crypto": "Crypto.com",

    # Categories
    "category": "Primary Category",
    "primary category": "Primary Category",
    "vertical": "Vertical",
    "iab": "Primary Category",

    "url": "URL",
    "jounce": "Jounce Media",
    "green": "Certified as Green Media",
}

# Canon for country comparisons (lowercased)
COUNTRY_CANON = {
    "usa": "united states",
    "us": "united states",
    "u.s.": "united states",
    "u.s": "united states",
    "united states": "united states",
    "united states of america": "united states",
    "uk": "united kingdom",
    "united kingdom": "united kingdom",
}

COUNTRY_SYNONYMS = {  # kept for display/normalization
    "usa": "United States",
    "us": "United States",
    "u.s.": "United States",
    "u.s": "United States",
    "uk": "United Kingdom",
}

# App-level aggregation keys
APP_GROUP_KEYS = [
    "Publisher Account GUID",
    "Publisher Account Name",
    "Publisher Account Type",
    "Inmobi App Inc ID",
    "Inmobi App Name",
    "Forwarded Bundle ID",
    "Operating System Name",
    "URL",
]

# Default display order (Removed eCPM from default.)
DISPLAY_COLS_ORDER = [
    "Publisher Account GUID","Publisher Account Name","Publisher Account Type",
    "Inmobi App Inc ID","Inmobi App Name",
    "Forwarded Bundle ID","Operating System Name","URL",
    "Monthly Traffic",
    "Ad Impressions Rendered","Total Burn",
    # "eCPM",  # not by default; included only if asked
]

CATEGORY_SYNONYMS = {
    "cars": "Automotive","auto": "Automotive","automobile": "Automotive",
    "beauty": "Beauty & Fitness","makeup": "Beauty & Fitness",
    "finance": "Finance","fintech": "Finance",
    "food": "Food & Drink","restaurants": "Food & Drink",
    "travel": "Travel","education": "Education",
    "health": "Health & Fitness","fitness": "Health & Fitness",
    "parenting": "Parenting","kids": "Parenting",
    "sports": "Sports","news": "News","shopping": "Shopping","ecommerce": "Shopping",
    "entertainment": "Entertainment","music": "Music","video": "Video",
    "productivity": "Productivity","business": "Business","lifestyle": "Lifestyle",
    "photography": "Photography","art": "Arts & Design","weather": "Weather",
    "maps": "Maps & Navigation","beauty & fitness": "Beauty & Fitness",
    "food & drink": "Food & Drink",
}

# =========================
# Helpers
# =========================
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

def normalize_country_display(value: Optional[str]) -> Optional[str]:
    if not value: return value
    return COUNTRY_SYNONYMS.get(value.strip().lower(), value)

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

# ---- Canonicalize Vertical values everywhere ----
def canonicalize_vertical_value(v: Any) -> Optional[str]:
    s = str(v or "").lower()
    s = re.sub(r"[^a-z]+", " ", s)     # keep letters; others → spaces
    s = re.sub(r"\s+", " ", s).strip()
    if not s: return None
    if s == "gaming": return "Gaming"
    if re.fullmatch(r"non\s*gaming", s): return "Non Gaming"
    return None  # unknown stays as-is

def normalize_vertical_column(df: pd.DataFrame) -> pd.DataFrame:
    if "Vertical" in df.columns:
        canon = df["Vertical"].map(canonicalize_vertical_value)
        df["Vertical"] = canon.fillna(df["Vertical"])
    return df
# -------------------------------------------------

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
    # Prefer existing "Vertical" but canonicalize it
    if "Vertical" in df.columns:
        return normalize_vertical_column(df)
    # Otherwise derive from categories
    if "Primary Category" in df.columns:
        mask = df["Primary Category"].astype(str).str.contains(r"\bgam(e|es|ing)\b", case=False, na=False)
    elif "Inmobi App Categories" in df.columns:
        mask = df["Inmobi App Categories"].astype(str).str.contains(r"\bgam(e|es|ing)\b", case=False, na=False)
    else:
        df["Vertical"] = "Non Gaming"
        return df
    df["Vertical"] = mask.map({True: "Gaming", False: "Non Gaming"})
    return df

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    cols_lower = {clean_header(c).lower(): c for c in df.columns}
    # 1) alias-based renames
    for alias, target in COLUMN_ALIASES.items():
        if target not in df.columns and alias in cols_lower:
            rename_map[cols_lower[alias]] = target

    # 2) regex safety nets
    for c in df.columns:
        if c in rename_map: continue
        lc = clean_header(c).lower()

        # OS
        if re.fullmatch(r"(os|operating\s*system(\s*name)?)", lc):
            if "Operating System Name" not in df.columns:
                rename_map[c] = "Operating System Name"
                continue

        # Forwarded Bundle ID
        if ("bundle" in lc or "package" in lc) and "id" in lc:
            if "Forwarded Bundle ID" not in df.columns:
                rename_map[c] = "Forwarded Bundle ID"
                continue

        # Content Rating
        if re.fullmatch(r"(content\s*rating(\s*id)?)", lc):
            if "Content Rating Id" not in df.columns:
                rename_map[c] = "Content Rating Id"
                continue

    if rename_map:
        df = df.rename(columns=rename_map)

    return df

# =========================
# Region selection
# =========================
def detect_region(prompt: str) -> str:
    p = (prompt or "").lower()
    na_terms = ["us","usa","united states","canada","north america","na"]
    apac_terms = ["india","indonesia","philippines","vietnam","apac","thailand","sg","singapore","malaysia"]
    if any(t in p for t in na_terms): return "NA"
    if any(t in p for t in apac_terms): return "APAC"
    return "NA"

@st.cache_data
def load_df(region: str) -> pd.DataFrame:
    path = "na_avails.xlsx" if region == "NA" else "apac_avails.xlsx"
    df = pd.read_excel(path, engine="openpyxl")
    df.columns = [clean_header(c) for c in df.columns]  # deep clean headers
    df = normalize_columns(df)
    if "Inmobi App Name" in df.columns:
        df["Inmobi App Name"] = df["Inmobi App Name"].apply(clean_app_name)
    df = apply_final_format(df)
    df = derive_vertical(df)
    df = normalize_vertical_column(df)  # ensure canonical after derivation
    df = coerce_types(df)
    return df

# =========================
# Category vocabulary & mapping (for specific asks)
# =========================
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

# =========================
# LLM parsing with explicit error surfacing
# =========================
LLM_SYSTEM = """You are a filter compiler for an avails dataset.
Return ONLY a JSON object with keys:
- filters: array of {column, op, value?}, op in ["equals","contains","in","not_in","gte","lte","gt","lt","is_true","is_false","allowed","blocked"]
- macros: optional array from ["premium","green","local_apps"]
- thresholds: optional {"min_requests": int, "min_rendered": int, "min_burn": number}
- final_format: optional in ["Banner","FSI","Native","Rewarded Video","API Video"]
- categories_raw: optional array of user category terms (e.g., ["cars","beauty","fintech"])
- include_cols: optional array of column names to include in the final display (e.g., ["Primary Category","Vertical","URL"])
- intent: optional in ["list","count"]; set "count" when the user asks "how many" or "count"
Guidance:
- "USA"/"US" => "United States" for column "Country Name".
- "SDK integration" => {"column":"Integration Method","op":"equals","value":"SDK"}.
- "rewarded video" => final_format "Rewarded Video" and/or {"column":"Is Rewarded Slot","op":"is_true"}.
- If the user says "gaming" or "non gaming", you MUST add {"column":"Vertical","op":"equals","value":"Gaming"/"Non Gaming"}.
  Do NOT treat "gaming" only as a free-text category.
- If user says "premium"/"safe", add macro "premium".
- If user says "green media", add macro "green".
- "local apps" => macro "local_apps".
- If user implies categories (e.g., "cars","fashion","fintech","education"), put them into categories_raw.
  Categories_raw must be matched against "Inmobi App Categories" or "Primary Category" (NOT against Vertical).
Return compact JSON. No prose, no code fences.
"""

def heuristic_query(prompt: str) -> Dict[str, Any]:
    p = (prompt or "").lower()
    out = {"filters": [], "macros": [], "thresholds": {}, "categories_raw": [], "include_cols": [], "intent": "list"}
    if any(k in p for k in ["usa","us","united states"]):
        out["filters"].append({"column":"Country Name","op":"equals","value":"United States"})
    if "india" in p: out["filters"].append({"column":"Country Name","op":"equals","value":"India"})
    if "android" in p: out["filters"].append({"column":"Operating System Name","op":"equals","value":"Android"})
    if "ios" in p or "iphone" in p: out["filters"].append({"column":"Operating System Name","op":"equals","value":"iOS"})
    if "sdk" in p: out["filters"].append({"column":"Integration Method","op":"equals","value":"SDK"})
    if "rewarded" in p: out["filters"].append({"column":"Is Rewarded Slot","op":"is_true"})
    if "banner" in p: out["filters"].append({"column":"Final Format","op":"equals","value":"Banner"})
    if "interstitial" in p or "fsi" in p: out["filters"].append({"column":"Final Format","op":"equals","value":"FSI"})
    if "video" in p and "rewarded" not in p: out["filters"].append({"column":"Final Format","op":"equals","value":"API Video"})
    if re.search(r"\bnon[-_\s]?gaming\b", p):
        out["filters"].append({"column":"Vertical","op":"equals","value":"Non Gaming"})
    elif re.search(r"\bgaming\b", p):
        out["filters"].append({"column":"Vertical","op":"equals","value":"Gaming"})
    if "premium" in p or "safe" in p: out["macros"].append("premium")
    if "green" in p: out["macros"].append("green")
    if "local" in p: out["macros"].append("local_apps")
    for k in ["cars","auto","beauty","finance","fintech","food","travel","education","health","fitness","parenting","sports","news","shopping","entertainment","music","video","business","lifestyle"]:
        if k in p: out["categories_raw"].append(k)
    if re.search(r"\bhow many\b|\bcount\b|\bnumber of\b", p):
        out["intent"] = "count"
    # crude “include” catcher
    include_hints = re.findall(r"(?:with|including|along with|also show|show)\s+([a-z0-9 _/&-]+)", p)
    for hint in include_hints:
        out["include_cols"].append(hint.strip())
    return out

def extract_query_with_errors(prompt: str) -> Dict[str, Any]:
    info = {"parsed": None, "error": None, "raw": None, "request_id": None}
    if not USE_LLM:
        info["parsed"] = heuristic_query(prompt)
        return info
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": LLM_SYSTEM},
                      {"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"},
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

# =========================
# Dynamic filter engine
# =========================
def alias_to_canonical(col: str) -> str:
    key = (col or "").strip().lower()
    return COLUMN_ALIASES.get(key, col)

def apply_one_filter(df: pd.DataFrame, column: str, op: str, value: Any = None) -> pd.DataFrame:
    col = alias_to_canonical(column)
    if col not in df.columns:
        return df

    # Canonicalize expected value for Vertical comparisons
    if col == "Vertical" and op == "equals":
        v_can = canonicalize_vertical_value(value)
        if v_can:
            value = v_can

    if op in {"allowed", "blocked"} and col in POLICY_COLUMNS:
        s = df[col].astype(str).str.lower()
        truthy = s.str.contains("allow|allowed|yes|true|1|clean", na=False)
        falsy  = s.str.contains("block|blocked|no|false|0|deny|restricted", na=False)
        return df[truthy] if op == "allowed" else df[falsy]

    series = df[col].astype(str)

    if op == "equals":
        if pd.api.types.is_numeric_dtype(df[col]):
            return df[df[col] == pd.to_numeric(value, errors="coerce")]

        # Special: normalize Vertical & Country Name comparisons
        if col == "Vertical":
            series_norm = df[col].map(canonicalize_vertical_value).fillna(series)
            return df[series_norm.str.casefold() == str(value).casefold()]

        if col == "Country Name":
            series_norm = df[col].apply(canonicalize_country_value)
            value_norm = canonicalize_country_value(value)
            return df[series_norm == value_norm]

        # default string equals
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

def apply_macros(df: pd.DataFrame, macros: List[str]) -> pd.DataFrame:
    d = df
    macros = macros or []
    if "premium" in macros:
        # SDK only
        if "Integration Method" in d.columns:
            d = d[d["Integration Method"].astype(str).str.contains(r"\bsdk\b", case=False, na=False)]
        # Exclude MA
        if "Content Rating Id" in d.columns:
            d = d[~d["Content Rating Id"].astype(str).str.fullmatch(r"\s*MA\s*", case=False)]
        # Jounce clean variants
        if "Jounce Media" in d.columns:
            def _norm(s: str) -> str:
                s = str(s or "").lower().strip()
                s = re.sub(r"\s+", " ", s)
                s = s.replace(" / ", "/").replace(" /", "/").replace("/ ", "/")
                return s
            CLEAN_CANON = {
                _norm("Clean"),
                _norm("Clean/no joiner score attached"),
                _norm("Clean/no jounce score attached"),
            }
            s = d["Jounce Media"].astype(str).map(_norm)
            d = d[s.isin(CLEAN_CANON)]
        # Basic app-name sanity (optional)
        if "Inmobi App Name" in d.columns:
            def ok(s):
                if not isinstance(s,str) or not s: return False
                bad = sum(1 for ch in s if not ch.isalnum() and ch not in " ._-&()")
                return bad / max(1,len(s)) <= 0.3
            d = d[d["Inmobi App Name"].apply(ok)]
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

def aggregate_app_level(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [c for c in APP_GROUP_KEYS if c in df.columns]
    if not group_cols:
        return pd.DataFrame()
    agg = {}
    for c in NUMERIC_HINTS:
        if c in df.columns:
            agg[c] = "sum"
    g = df.groupby(group_cols, dropna=False).agg(agg).reset_index()
    # Compute eCPM if you ever include it explicitly
    if "Total Burn" in g.columns and "Ad Impressions Rendered" in g.columns:
        denom = g["Ad Impressions Rendered"].replace(0, pd.NA)
        g["eCPM"] = (g["Total Burn"] * 1000.0) / denom
    return g

# ======= INCLUDE-COLS handling =======
def normalize_include_cols(include_cols: List[str], df: pd.DataFrame) -> List[str]:
    if not include_cols: return []
    known = set(df.columns)
    out = []
    for raw in include_cols:
        cand = raw.strip()
        # exact
        if cand in known:
            out.append(cand); continue
        # alias
        alias = alias_to_canonical(cand)
        if alias in known:
            out.append(alias); continue
        # fuzzy
        best = difflib.get_close_matches(cand, list(known), n=1, cutoff=0.7)
        if best: out.append(best[0])
    # de-dup, keep order
    seen = set(); deduped = []
    for c in out:
        if c not in seen:
            seen.add(c); deduped.append(c)
    return deduped

# ======= INTENT helpers =======
def is_count_intent(user_text: str, qobj: Dict[str, Any]) -> bool:
    if (qobj or {}).get("intent") == "count":
        return True
    return bool(re.search(r"\bhow many\b|\bcount\b|\bnumber of\b", (user_text or "").lower()))

# =========================
# UI
# =========================
st.title("Avails Bot — App-level List (Smart, Schema-Aware)")

st.sidebar.header("Run options")
safe_mode = st.sidebar.checkbox("Safe mode (ignore AI & thresholds)", value=False)
if st.sidebar.button("🔁 Clear cached data"):
    st.cache_data.clear()
    st.experimental_rerun()

user_input = st.text_input("Ask anything (e.g., 'Premium rewarded video gaming apps in United States on SDK, Android; include Primary Category')")

if user_input:
    region = detect_region(user_input)
    df = load_df(region)
    st.caption(f"Region detected: **{region}**")

    with st.expander("Debug: key counts"):
        if "Final Format" in df.columns: st.write("Final Format:", df["Final Format"].value_counts(dropna=False))
        if "Integration Method" in df.columns: st.write("Integration Method:", df["Integration Method"].value_counts(dropna=False))
        if "Country Name" in df.columns: st.write("Top countries:", df["Country Name"].value_counts(dropna=False).head(20))
        if "Vertical" in df.columns: st.write("Vertical (raw/canonical):", df["Vertical"].value_counts(dropna=False))

    required_for_group = ["Inmobi App Name","Forwarded Bundle ID","Operating System Name"]
    missing = [c for c in required_for_group if c not in df.columns]
    if missing:
        st.error(f"Missing required columns for grouping: {missing}. Check your Excel headers or update COLUMN_ALIASES.")

    if safe_mode:
        st.info("Safe mode ON — unfiltered, aggregated app list.")
        g = aggregate_app_level(df.copy())
        # ensure defaults
        for _col, _default in [
            ("Monthly Traffic", 0),
            ("Ad Impressions Rendered", 0),
            ("Total Burn", 0.0),
        ]:
            if _col not in g.columns: g[_col] = _default
        if g.empty:
            st.error("Aggregation returned empty. Check APP_GROUP_KEYS columns exist.")
        else:
            sort_cols = [c for c in ["Monthly Traffic","Total Burn","Ad Impressions Rendered"] if c in g.columns]
            if sort_cols: g = g.sort_values(sort_cols, ascending=[False]*len(sort_cols))
            base_cols = [c for c in DISPLAY_COLS_ORDER if c in g.columns]
            st.dataframe(g[base_cols].head(500), use_container_width=True)
            st.download_button("Download CSV", g.to_csv(index=False).encode("utf-8"), "avails_app_level.csv", "text/csv")
        st.stop()

    # ==== Parse with LLM (and show exact errors/raw if any) ====
    parse_info = extract_query_with_errors(user_input)
    q = parse_info.get("parsed") or {}

    # --- HARD ENFORCEMENT: derive the Vertical + Macros directly from user text ---
    text_lc = (user_input or "").lower()

    def ensure_vertical_filter(qobj, text):
        qobj["filters"] = [f for f in (qobj.get("filters") or []) if f.get("column","").strip().lower() != "vertical"]
        if re.search(r"\bnon[-_\s]?gaming\b", text):
            qobj.setdefault("filters", []).append({"column":"Vertical","op":"equals","value":"Non Gaming"})
        elif re.search(r"\bgaming\b", text):
            qobj.setdefault("filters", []).append({"column":"Vertical","op":"equals","value":"Gaming"})
        return qobj

    def ensure_macros(qobj, text):
        macs = set((qobj.get("macros") or []))
        if re.search(r"\bpremium\b|\bsafe\b", text): macs.add("premium")
        if re.search(r"\bgreen\b", text): macs.add("green")
        if re.search(r"\blocal apps?\b", text): macs.add("local_apps")
        qobj["macros"] = list(macs)
        return qobj

    def ensure_rewarded(qobj, text):
        # make rewarded stricter/explicit if the text says so
        if re.search(r"\brewarded( video)?\b", text):
            # prefer final_format; also mark slot
            qobj["final_format"] = "Rewarded Video"
            existing = qobj.get("filters", [])
            has_slot = any((f.get("column","").strip().lower()=="is rewarded slot") for f in existing)
            if not has_slot:
                existing.append({"column":"Is Rewarded Slot","op":"is_true"})
            qobj["filters"] = existing
        return qobj

    q = ensure_vertical_filter(q, text_lc)
    q = ensure_macros(q, text_lc)
    q = ensure_rewarded(q, text_lc)

    with st.expander("Parsed query (debug)"):
        st.json(q)
    with st.expander("LLM errors & raw (if any)"):
        if parse_info.get("request_id"): st.write(f"request_id: {parse_info['request_id']}")
        if parse_info.get("error"): st.error(parse_info["error"])
        if parse_info.get("raw"): st.code(parse_info["raw"], language="json")

    d = df.copy()

    # Apply Vertical first (authoritative split, after normalizing column)
    d = normalize_vertical_column(d)
    for filt in q.get("filters", []):
        if (filt.get("column") or "").strip().lower() == "vertical" and "Vertical" in d.columns:
            d = apply_one_filter(d, "Vertical", filt.get("op","equals"), filt.get("value"))

    # final_format if present (robust rewarded handling)
    ff = q.get("final_format")
    if ff and "Final Format" in d.columns:
        if ff == "Rewarded Video":
            mask = d["Final Format"].astype(str).str.casefold().eq("rewarded video")
            if "Is Rewarded Slot" in d.columns:
                mask = mask | d["Is Rewarded Slot"].astype(str).str.contains("true|1|yes", case=False, na=False)
            d = d[mask]
        else:
            d = apply_one_filter(d, "Final Format", "equals", ff)

    # general filters (non-Vertical)
    for filt in q.get("filters", []):
        col = (filt.get("column") or "").strip()
        if col.lower() == "vertical": continue
        op = filt.get("op"); val = filt.get("value")
        if alias_to_canonical(col) in {"Country Name"} and isinstance(val, str):
            pass
        d = apply_one_filter(d, col or "", op or "equals", val)

    # categories mapping (ONLY for specific asks)
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

    # macros (now reliably enforced)
    d = apply_macros(d, q.get("macros", []))

    # Debug after filters
    with st.expander("Debug after filters"):
        if "Vertical" in d.columns:
            st.write("Vertical counts (post-filter):", d["Vertical"].value_counts(dropna=False))

    # Aggregate
    g = aggregate_app_level(d)

    # Guarantee defaults (no eCPM default)
    for _col, _default in [
        ("Monthly Traffic", 0),
        ("Ad Impressions Rendered", 0),
        ("Total Burn", 0.0),
    ]:
        if _col not in g.columns: g[_col] = _default

    # thresholds — NONE by default; use sidebar (kept simple)
    st.sidebar.header("Thresholds (post-aggregation)")
    vol_col = "Monthly Traffic" if "Monthly Traffic" in g.columns else None
    min_vol = st.sidebar.number_input(f"Min {vol_col or 'Volume'}", 0, 1_000_000_000, 0, 1000) if vol_col else 0
    min_rend = st.sidebar.number_input("Min Ad Impressions Rendered", 0, 1_000_000_000, 0, 1000)
    min_burn = st.sidebar.number_input("Min Total Burn ($)", 0.0, 1e12, 0.0, 1.0)

    if vol_col and min_vol: g = g[g[vol_col] >= min_vol]
    if "Ad Impressions Rendered" in g.columns and min_rend: g = g[g["Ad Impressions Rendered"] >= min_rend]
    if "Total Burn" in g.columns and min_burn: g = g[g["Total Burn"] >= min_burn]

    # Sort — Monthly Traffic → Burn → Rendered
    sort_cols = [c for c in ["Monthly Traffic","Total Burn","Ad Impressions Rendered"] if c in g.columns]
    if sort_cols:
        g = g.sort_values(sort_cols, ascending=[False]*len(sort_cols))

    # ===== Intent: COUNT (unique bundles/apps) =====
    if is_count_intent(user_input, q):
        # Unique Forwarded Bundle ID after filters/aggregation scope
        uniq_bundles = g["Forwarded Bundle ID"].nunique() if "Forwarded Bundle ID" in g.columns else 0
        uniq_apps = g["Inmobi App Inc ID"].nunique() if "Inmobi App Inc ID" in g.columns else uniq_bundles
        st.success(f"Unique apps (by bundle): **{uniq_bundles}**  |  Unique apps (by Inmobi App Inc ID): **{uniq_apps}**")
        # Still show a small preview for context
        preview_cols = [c for c in ["Inmobi App Name","Forwarded Bundle ID","Operating System Name","Monthly Traffic"] if c in g.columns]
        if preview_cols:
            st.dataframe(g[preview_cols].head(50), use_container_width=True)
        st.download_button("Download CSV", g.to_csv(index=False).encode("utf-8"), "avails_app_level.csv", "text/csv")
        st.stop()

    # ===== Include extra columns on demand =====
    include_cols = normalize_include_cols(q.get("include_cols", []), g)
    # Allow users to explicitly ask for eCPM
    if any(re.search(r"\becpm\b", str(x), re.I) for x in q.get("include_cols", [])):
        if "eCPM" not in include_cols and "eCPM" in g.columns:
            include_cols.append("eCPM")

    # Graceful fallbacks if empty
    if g.empty:
        # 1) Try relaxing strict rewarded to rewarded slot
        if ff == "Rewarded Video" and "Final Format" in df.columns:
            d_relaxed = df.copy()
            d_relaxed = normalize_vertical_column(d_relaxed)
            for filt in q.get("filters", []):
                if (filt.get("column") or "").strip().lower() == "vertical":
                    d_relaxed = apply_one_filter(d_relaxed, "Vertical", filt.get("op","equals"), filt.get("value"))
            for filt in q.get("filters", []):
                col = (filt.get("column") or "").strip().lower()
                if col in {"vertical","final format"}: 
                    continue
                d_relaxed = apply_one_filter(d_relaxed, filt.get("column") or "", filt.get("op") or "equals", filt.get("value"))
            if "Is Rewarded Slot" in d_relaxed.columns:
                d_relaxed = d_relaxed[d_relaxed["Is Rewarded Slot"].astype(str).str.contains("true|1|yes", case=False, na=False)]
            d_relaxed = apply_macros(d_relaxed, q.get("macros", []))
            g_relaxed = aggregate_app_level(d_relaxed)
            for _col, _default in [("Monthly Traffic",0),("Ad Impressions Rendered",0),("Total Burn",0.0)]:
                if _col not in g_relaxed.columns: g_relaxed[_col] = _default
            if not g_relaxed.empty:
                st.info("No rows with strict 'Rewarded Video'. Showing **rewarded slots** (relaxed) with your other filters.")
                sort_cols3 = [c for c in ["Monthly Traffic","Total Burn","Ad Impressions Rendered"] if c in g_relaxed.columns]
                if sort_cols3: g_relaxed = g_relaxed.sort_values(sort_cols3, ascending=[False]*len(sort_cols3))
                base_cols3 = [c for c in DISPLAY_COLS_ORDER if c in g_relaxed.columns]
                show_cols3 = base_cols3 + [c for c in include_cols if c not in base_cols3 and c in g_relaxed.columns]
                st.dataframe(g_relaxed[show_cols3].head(500), use_container_width=True)
                st.download_button("Download CSV (rewarded relaxed)", g_relaxed.to_csv(index=False).encode("utf-8"), "avails_rewarded_relaxed.csv", "text/csv")
                st.stop()

        st.warning("No results. Try lowering thresholds or removing 'rewarded'/'premium'.")
        st.stop()

    # Success table
    base_cols = [c for c in DISPLAY_COLS_ORDER if c in g.columns]
    show_cols = base_cols + [c for c in include_cols if c not in base_cols and c in g.columns]

    st.success(f"Apps matched: {len(g)}")
    st.dataframe(g[show_cols].head(500), use_container_width=True)
    st.download_button("Download CSV", g.to_csv(index=False).encode("utf-8"), "avails_app_level.csv", "text/csv")

    with st.expander("Available columns in dataset"):
        st.write(sorted(df.columns.tolist()))
