import streamlit as st
import pandas as pd
import openai
import os
import json

# ---------- Load OpenAI Key from Streamlit Secrets ----------
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ---------- Region Detection ----------
def detect_region(prompt):
    prompt = prompt.lower()
    na_terms = ["us", "usa", "canada", "north america", "na"]
    apac_terms = ["india", "indonesia", "philippines", "vietnam", "apac", "thailand", "sg", "singapore"]
    if any(term in prompt for term in na_terms):
        return "NA"
    elif any(term in prompt for term in apac_terms):
        return "APAC"
    else:
        return "NA"

# ---------- Load Data & Derive Final Format ----------
@st.cache_data
def load_data(region):
    if region == "NA":
        df = pd.read_excel("na_avails.xlsx", engine="openpyxl")
    else:
        df = pd.read_excel("apac_avails.xlsx", engine="openpyxl")

    def derive_format(row):
        placement = str(row.get("Placement Type", "")).lower()
        rewarded = str(row.get("Is Rewarded Slot", "")).lower() == "true"
        if placement == "banner":
            return "Banner"
        elif placement == "interstitial":
            return "Rewarded Video" if rewarded else "FSI"
        elif placement == "native":
            return "Native"
        elif placement == "video":
            return "API Video"
        return "Unknown"

    df["Final Format"] = df.apply(derive_format, axis=1)
    return df

# ---------- Extract Filters via OpenAI ----------
def extract_filters(prompt):
    system_message = """You are a filter extractor for a supply availability bot.
Given a user query, return a JSON object with filters like:
{
  "country": "India",
  "final_format": "Rewarded Video",
  "allows_crypto_ads": true,
  "os": "Android",
  "is_premium_inventory": true
}
Valid keys: country, final_format, os, publisher_type, is_premium_inventory,
allows_crypto_ads, allows_gambling_ads, allows_cannabis_ads, content_rating, cas_min.
ALWAYS return a valid JSON object. Never include explanation or code block formatting."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        content = response.choices[0].message.content.strip()
        return json.loads(content)

    except Exception as e:
        st.error(f"OpenAI Error: {e}")
        return {}

# ---------- Apply Filters to DF ----------
def apply_filters(df, filters):
    for k, v in filters.items():
        try:
            if k == "country":
                df = df[df["Country Name"].str.lower() == v.lower()]
            elif k == "final_format":
                df = df[df["Final Format"].str.lower() == v.lower()]
            elif k == "os":
                df = df[df["OS"].str.lower() == v.lower()]
            elif k == "publisher_type":
                df = df[df["Publisher Account Type"].str.lower() == v.lower()]
            elif k == "is_premium_inventory":
                df = df[df["is_premium_inventory"] == v]
            elif k == "allows_crypto_ads":
                df = df[df["Crypto.com"] == "Yes"] if v else df[df["Crypto.com"] == "No"]
            elif k == "allows_gambling_ads":
                df = df[df["Gambling/Card Games"] == "Yes"] if v else df[df["Gambling/Card Games"] == "No"]
            elif k == "allows_cannabis_ads":
                df = df[df["Medicinal Drugs/Alternative Medicines"] == "Yes"] if v else df[df["Medicinal Drugs/Alternative Medicines"] == "No"]
            elif k == "content_rating":
                df = df[df["Content Rating Id"].str.upper() == v.upper()]
            elif k == "cas_min":
                df = df[df["CAS Forwards"] >= int(v)]
        except Exception as e:
            st.warning(f"Error applying filter '{k}': {e}")
    return df

# ---------- Streamlit App ----------
st.set_page_config(page_title="Avails Bot", layout="wide")
st.title("Avails Bot MVP — Final Format Logic Integrated")

user_input = st.text_input("Ask a question (e.g. 'Banner apps in USA that allow gambling')")

if user_input:
    region = detect_region(user_input)
    df = load_data(region)

    st.info(f"Region detected: `{region}`")

    filters = extract_filters(user_input)

    if filters:
        st.success(f"Filters applied: `{filters}`")
        filtered_df = apply_filters(df, filters)

        if not filtered_df.empty:
            st.write(f"{len(filtered_df)} results found.")
            st.dataframe(filtered_df.head(20))
            st.download_button("Download CSV", filtered_df.to_csv(index=False), "filtered_avails.csv")
        else:
            st.warning("No matching results found. Try simplifying your query.")
    else:
        st.warning("Couldn’t understand the query. Try rephrasing.")
        st.markdown("Or use manual filters below")

        st.subheader("Manual Filters")
        country = st.selectbox("Country", sorted(df["Country Name"].dropna().unique()))
        os_choice = st.selectbox("OS", ["iOS", "Android"])
        format_choice = st.selectbox("Final Format", sorted(df["Final Format"].dropna().unique()))
        gambling = st.selectbox("Allows Gambling Ads?", ["Yes", "No"])
        crypto = st.selectbox("Allows Crypto Ads?", ["Yes", "No"])

        manual_df = df[
            (df["Country Name"] == country) &
            (df["OS"] == os_choice) &
            (df["Final Format"] == format_choice) &
            (df["Gambling/Card Games"] == gambling) &
            (df["Crypto.com"] == crypto)
        ]

        if not manual_df.empty:
            st.success(f"{len(manual_df)} manual results found.")
            st.dataframe(manual_df.head(20))
            st.download_button("Download CSV", manual_df.to_csv(index=False), "filtered_manual_avails.csv")
        else:
            st.warning("No results found with manual filters.")
