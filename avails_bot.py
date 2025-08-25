import os, json, re, unicodedata
import pandas as pd
import streamlit as st
import openai
from typing import List, Optional

# ---------- OpenAI (kept as in your app) ----------
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ---------- Helpers ----------
NUMERIC_COLS = ["Valid Ad Request","Ad Impressions Served","Valid Wins",
                "Ad Impressions Rendered","Total Burn","eCPM"]

def clean_app_name(s: str) -> str:
    if pd.isna(s): return s
    s = unicodedata.normalize("NFKC", str(s))
    s = "".join(ch for ch in s if ch.isprintable())
    return re.sub(r"\s+"," ",s).strip()

def derive_final_format(row) -> str:
    placement = str(row.get("Placement Type","")).strip().lower()
    rewarded = str(row.get("Is Rewarded Slot","")).strip().lower() in {"true","1","yes"}
    if placement == "banner":
        return "Banner"
    elif placement == "interstitial":
        return "Rewarded Video" if rewarded else "FSI"
    elif placement == "native":
        return "Native"
    elif placement == "video":
        return "API Video"
    return "Unknown"

def map_environment(inv: str, dev: str) -> str:
    dt, ic = (dev or "").lower(), (inv or "").lower()
    if dt in {"is_ott","is_connected_tv","is_console"}: return "CTV"
    if "web" in ic: return "Web"
    return "App"

# ---------- Region Detection (kept) ----------
def detect_region(prompt: str) -> str:
    p = (prompt or "").lower()
    na_terms = ["us","usa","canada","north america","na"]
    apac_terms = ["india","indonesia","philippines","vietnam","apac","thailand","sg","singapore","malaysia"]
    if any(t in p for t in na_terms): return "NA"
    if any(t in p for t in apac_terms): return "APAC"
    return "NA"

# ---------- Load Data ----------
@st.cache_data
def load_data(region: str) -> pd.DataFrame:
    path = "na_avails.xlsx" if region == "NA" else "apac_avails.xlsx"
    df = pd.read_excel(path, engine="openpyxl")

    # Derivations / normalization
    if "Inmobi App Name" in df.columns:
        df["Inmobi App Name"] = df["Inmobi App Name"].apply(clean_app_name)

    df["Final Format"] = df.apply(derive_final_format, axis=1)

    if {"Inventory Channel","Device Type Name"} <= set(df.columns):
        df["Environment"] = df.apply(lambda r: map_environment(
            r.get("Inventory Channel",""), r.get("Device Type Name","")), axis=1)
    else:
        df["Environment"] = "App"

    # consistent types
    for c in NUMERIC_COLS:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["Is Rewarded Slot","Coppa Enabled","Certified as Green Media","Jounce Media",
              "Content Rating Id","Integration Method"]:
        if c in df.columns: df[c] = df[c].astype(str)

    return df

# ---------- OpenAI: extract filters (kept, but expanded keys) ----------
def extract_filters(prompt: str) -> dict:
    system = """You extract structured filters for a supply availability bot.
Return ONLY a valid JSON object (no text) with any of these keys:
country, os, final_format, rewarded, premium, green, local_apps,
categories (array of strings), policy_flags (array like "Gambling/Card Games:Allowed"),
min_requests, min_rendered, min_burn.
Example:
{"country":"India","os":"Android","final_format":"Rewarded Video","rewarded":true,"premium":true,"green":false,"local_apps":false,"categories":["Gaming"],"policy_flags":["Gambling/Card Games:Allowed"],"min_requests":100000,"min_rendered":10000,"min_burn":10}"""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role":"system","content":system},
                      {"role":"user","content":prompt}],
            temperature=0.2
        )
        return json.loads(resp.choices[0].message.content.strip())
    except Exception as e:
        st.warning(f"OpenAI parsing issue: {e}")
        return {}

# ---------- Core: apply filters & aggregate at APP LEVEL ----------
def apply_filters_and_aggregate(
    df: pd.DataFrame,
    country: Optional[str] = None,
    os_filter: Optional[List[str]] = None,
    final_format: Optional[List[str]] = None,  # Banner/FSI/Native/Rewarded Video/API Video
    rewarded: Optional[bool] = None,
    premium: bool = False,
    green: bool = False,
    local_apps: bool = False,
    categories_include: Optional[List[str]] = None,  # Gaming / Non Gaming
    policy_flags: Optional[List[str]] = None,        # ["Gambling/Card Games:Allowed", ...]
    min_requests: int = 0, min_rendered: int = 0, min_burn: float = 0.0
) -> pd.DataFrame:

    d = df.copy()

    if country and "Country Name" in d:
        d = d[d["Country Name"].astype(str).str.contains(country, case=False, na=False)]

    if os_filter and "Operating System Name" in d:
        d = d[d["Operating System Name"].astype(str).isin(os_filter)]

    if final_format and "Final Format" in d:
        d = d[d["Final Format"].astype(str).isin(final_format)]

    if rewarded is not None and "Is Rewarded Slot" in d:
        mask = d["Is Rewarded Slot"].str.contains("true|1|yes", case=False, na=False)
        d = d[mask] if rewarded else d[~mask]

    if green and "Certified as Green Media" in d:
        d = d[d["Certified as Green Media"].str.contains("true|1|yes|green", case=False, na=False)]

    if local_apps and {"Publisher Origin Country Name","Country Name"} <= set(d.columns):
        d = d[d["Publisher Origin Country Name"].str.lower() == d["Country Name"].str.lower()]

    if categories_include:
        col = "Vertical" if "Vertical" in d else ("Primary Category" if "Primary Category" in d else None)
        if col:
            d = d[d[col].astype(str).str.contains("|".join(categories_include), case=False, na=False)]

    if policy_flags:
        for item in policy_flags:
            if ":" in item:
                col, need = [t.strip() for t in item.split(":",1)]
                if col in d:
                    d = d[d[col].astype(str).str.contains(need, case=False, na=False)]

    if premium:
        if "Integration Method" in d: d = d[d["Integration Method"].str.contains("sdk", case=False, na=False)]
        if "Content Rating Id" in d: d = d[~d["Content Rating Id"].str.fullmatch(r"MA", case=False)]
        if "Jounce Media" in d: d = d[d["Jounce Media"].str.contains("clean", case=False, na=False)]
        if "Inmobi App Name" in d:
            def ok(s):
                if not isinstance(s,str) or not s: return False
                bad = sum(1 for ch in s if not ch.isalnum() and ch not in " ._-&()")
                return bad / max(1,len(s)) <= 0.3
            d = d[d["Inmobi App Name"].apply(ok)]

    # ---- Aggregate at APP level ----
    group_cols = [c for c in [
        "Publisher Account GUID","Publisher Account Name","Publisher Account Type",
        "Inmobi App Inc ID","Inmobi App Name","Operating System Name",
        "Environment","URL"
    ] if c in d.columns]
    if not group_cols:
        return pd.DataFrame()

    agg = {c: ("mean" if c=="eCPM" else "sum") for c in NUMERIC_COLS if c in d.columns}
    g = d.groupby(group_cols, dropna=False).agg(agg).reset_index()

    if "Valid Ad Request" in g and min_requests: g = g[g["Valid Ad Request"] >= min_requests]
    if "Ad Impressions Rendered" in g and min_rendered: g = g[g["Ad Impressions Rendered"] >= min_rendered]
    if "Total Burn" in g and min_burn: g = g[g["Total Burn"] >= min_burn]

    sort_cols = [c for c in ["Total Burn","Valid Ad Request","Ad Impressions Rendered"] if c in g.columns]
    if sort_cols: g = g.sort_values(sort_cols, ascending=[False]*len(sort_cols))

    return g

# ---------- UI ----------
st.set_page_config(page_title="Avails Bot MVP — App-level", layout="wide")
st.title("Avails Bot MVP — App‑level Aggregation")

user_input = st.text_input("Ask (e.g., 'Rewarded video in India, premium only, Android, >100k req')")

if user_input:
    region = detect_region(user_input)
    df = load_data(region)
    st.caption(f"Region detected: **{region}**")

    filters = extract_filters(user_input)
    if filters:
        st.write("**Parsed filters**:", filters)

        # map filters to our function
        os_filter = [filters["os"]] if isinstance(filters.get("os"), str) else filters.get("os")
        final_format = [filters["final_format"]] if isinstance(filters.get("final_format"), str) else filters.get("final_format")

        res = apply_filters_and_aggregate(
            df,
            country = filters.get("country"),
            os_filter = os_filter,
            final_format = final_format,
            rewarded = filters.get("rewarded"),
            premium = bool(filters.get("premium", False)),
            green = bool(filters.get("green", False)),
            local_apps = bool(filters.get("local_apps", False)),
            categories_include = filters.get("categories"),
            policy_flags = filters.get("policy_flags"),
            min_requests = int(filters.get("min_requests", 0) or 0),
            min_rendered = int(filters.get("min_rendered", 0) or 0),
            min_burn = float(filters.get("min_burn", 0) or 0.0),
        )

        if res.empty:
            st.warning("No results. Loosen filters or thresholds.")
        else:
            st.success(f"Apps matched: {len(res)}")
            cols = [c for c in [
                "Publisher Account GUID","Publisher Account Name","Publisher Account Type",
                "Inmobi App Inc ID","Inmobi App Name","Operating System Name","Environment","URL",
                "Valid Ad Request","Ad Impressions Rendered","Total Burn","eCPM"
            ] if c in res.columns]
            st.dataframe(res[cols].head(200), use_container_width=True)
            st.download_button("Download CSV", res.to_csv(index=False).encode("utf-8"),
                               "avails_app_level.csv","text/csv")
    else:
        st.warning("Couldn’t understand the query. Try rephrasing.")
