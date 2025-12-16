# ================== NFS PREDICTOR (DUAL CSV: VIEWS + CTR) ==================

import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# ------------------ PAGE ------------------
st.set_page_config(page_title="NFS Predictor", layout="centered")
st.title("🔮 NFS View & CTR Predictor")

status = st.empty()

# ------------------ HELPERS ------------------

def clean_text(x):
    return "" if pd.isna(x) else str(x)

def parse_ctr(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip().replace(" ", "").replace("%", "").replace(",", ".")
    try:
        return float(x)
    except:
        return np.nan

def percentile(arr, x):
    return (arr < x).mean()

def band(p):
    s = int(round(p * 100))
    if s >= 70: return f"{s} — Strong"
    if s >= 50: return f"{s} — Average"
    return f"{s} — Weak"

# ------------------ UPLOADS ------------------

vidiq_file = st.file_uploader("Upload vidIQ CSV (views data)", type=["csv"])
ctr_file   = st.file_uploader("Upload CTR CSV (title + CTR)", type=["csv"])

if not vidiq_file:
    st.stop()

status.info("📥 Reading vidIQ CSV…")
dfv = pd.read_csv(vidiq_file)

# ------------------ VIEW DATA PREP ------------------

def find_col(df, keys):
    for c in df.columns:
        for k in keys:
            if k in c.lower():
                return c
    return None

title_col = find_col(dfv, ["title"])
desc_col  = find_col(dfv, ["desc"])
kw_col    = find_col(dfv, ["tag", "keyword"])
views_col = find_col(dfv, ["view"])

if not title_col or not views_col:
    st.error("❌ vidIQ CSV must contain Title and Views columns")
    st.stop()

dfv["title"] = dfv[title_col].apply(clean_text)
dfv["description"] = dfv[desc_col].apply(clean_text) if desc_col else ""
dfv["keywords"] = dfv[kw_col].apply(clean_text) if kw_col else ""
dfv["views"] = pd.to_numeric(
    dfv[views_col].astype(str).str.replace(r"[^\d]", "", regex=True),
    errors="coerce"
)

dfv = dfv.dropna(subset=["views"])

# ------------------ TRAIN VIEW MODELS ------------------

status.info("⏳ Training view-based ensemble models…")

X = dfv[["title", "description", "keywords"]]
y = np.log1p(dfv["views"])

Xt, _, yt, _ = train_test_split(X, y, test_size=0.2, random_state=42)

def train_ensemble(text_series):
    pipes, preds = [], []
    for seed in [1, 11, 42]:
        pipe = Pipeline([
            ("vec", TfidfVectorizer(stop_words="english")),
            ("m", GradientBoostingRegressor(random_state=seed))
        ])
        pipe.fit(text_series, yt)
        pipes.append(pipe)
        preds.append(np.expm1(pipe.predict(text_series)))
    return pipes, np.mean(np.vstack(preds), axis=0)

title_pipes, title_train = train_ensemble(Xt["title"])
desc_pipes,  desc_train  = train_ensemble(Xt["description"])
kw_pipes,    kw_train    = train_ensemble(Xt["keywords"])

title_arr = np.sort(title_train)
desc_arr  = np.sort(desc_train)
kw_arr    = np.sort(kw_train)

status.success("✅ View models ready")

# ------------------ CTR DATA PREP ------------------

ctr_ready = False

if ctr_file:
    status.info("📥 Reading CTR CSV…")
    dfc = pd.read_csv(ctr_file)

    ctr_title_col = find_col(dfc, ["title"])
    ctr_col = find_col(dfc, ["ctr", "click"])

    if ctr_title_col and ctr_col:
        dfc["title"] = dfc[ctr_title_col].apply(clean_text)
        dfc["ctr"] = dfc[ctr_col].apply(parse_ctr)
        dfc = dfc.dropna(subset=["title", "ctr"])

        st.write("Usable CTR rows:", len(dfc))

        if len(dfc) >= 10:
            status.info("⏳ Training CTR model…")
            ctr_pipes, ctr_preds = [], []

            for seed in [1, 11, 42]:
                pipe = Pipeline([
                    ("vec", TfidfVectorizer(stop_words="english")),
                    ("m", GradientBoostingRegressor(random_state=seed))
                ])
                pipe.fit(dfc["title"], dfc["ctr"])
                ctr_pipes.append(pipe)
                ctr_preds.append(pipe.predict(dfc["title"]))

            ctr_arr = np.sort(np.mean(np.vstack(ctr_preds), axis=0))
            ctr_ready = True
            status.success("✅ CTR model ready")
        else:
            status.warning("⚠️ CTR CSV needs at least 10 rows")
    else:
        status.warning("⚠️ CTR CSV must contain Title and CTR columns")

status.success("🎯 App ready for prediction")

# ------------------ INPUT ------------------

st.markdown("---")
title_in = st.text_input("Title")
desc_in = st.text_area("Description")
kw_in = st.text_input("Keywords")

if not title_in:
    st.stop()

# ------------------ VIEW SCORES ------------------

def ensemble_predict(pipes, text):
    return np.mean([np.expm1(p.predict([text])[0]) for p in pipes])

title_raw = ensemble_predict(title_pipes, title_in)
desc_raw  = ensemble_predict(desc_pipes, desc_in)
kw_raw    = ensemble_predict(kw_pipes, kw_in)

pt = percentile(title_arr, title_raw)
pd = percentile(desc_arr, desc_raw)
pk = percentile(kw_arr, kw_raw)

st.markdown("## 📊 View-Based Scores")
c1, c2, c3 = st.columns(3)
c1.metric("Title Score", band(pt))
c2.metric("Description Score", band(pd))
c3.metric("Keywords Score", band(pk))

# ------------------ CTR SECTION ------------------

st.markdown("---")
st.header("🎯 CTR Prediction")

if ctr_ready:
    ctr_val = np.mean([p.predict([title_in])[0] for p in ctr_pipes])
    ctr_p = percentile(ctr_arr, ctr_val)

    st.metric("Predicted CTR", f"{ctr_val:.2f}%")
    st.metric("CTR Score", band(ctr_p))

    st.subheader("Diagnosis")
    if ctr_p >= 0.6 and pt < 0.5:
        st.warning("High CTR but low Title Score → Clickable, not scalable")
    elif ctr_p < 0.4 and pt >= 0.6:
        st.warning("Scalable title but weak hook")
    elif ctr_p >= 0.6 and pt >= 0.6:
        st.success("Breakout candidate")
    else:
        st.info("Needs rewrite")
else:
    st.info("CTR model not available")

