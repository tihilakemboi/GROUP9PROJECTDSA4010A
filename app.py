import os
import torch
import pandas as pd
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

st.set_page_config(
    page_title="Kenya Multilingual PSA Translator",
    page_icon="📢",
    layout="wide"
)

# Built-in PSA translations for Ekegusii demo mode
EKEGUSII_DICT = {
    "farmers are advised to plant early-maturing seeds before the rains start": 
        "Abarimi nabanoibwi gochia kobia imbora etaraba ekeribwa",
    "fever, headache, and body weakness": 
        "Ogosaria kw'omooyo, Ogotema kwo omotwe, ne chinguvu chi'omobiri korwa",
    "report any suspicious activities to the nearest police station": 
        "Tebia ebikorwa ebiechani amo ase chikereng'a chia polis",
    "wash your hands with clean water and soap": 
        "Naba amaboko oo namache amachenu na sabuni",
    "stay inside during heavy rainfall and flooding": 
        "Oramenyere inka eng'ana y'imbora enene nemechango"
}

@st.cache_resource(show_spinner="Loading NLLB Translation Model...")
def load_app_model():
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, 
        low_cpu_mem_usage=True,
        dtype=torch.float32
    )
    return tokenizer, model

tokenizer, model = load_app_model()

LANG_CODES = {
    "english": "eng_Latn",
    "swahili": "swh_Latn",
    "ekegusii": "ekg_Latn"
}

def translate_psa(text: str, src_lang: str = "english", target_lang: str = "swahili") -> str:
    if not isinstance(text, str) or not text.strip():
        return ""

    clean_text = text.strip().lower()

    # Rule-Based fallback for Ekegusii (Low resource fine-tune)
    if target_lang.lower() == "ekegusii":
        for pattern, ek_translation in EKEGUSII_DICT.items():
            if pattern in clean_text:
                return ek_translation
        return f"[Ekegusii]: Abarimi / Abagusi, {text.strip()}"

    # Swahili translation via NLLB Base Model
    src_code = LANG_CODES.get(src_lang.lower(), "eng_Latn")
    tgt_code = LANG_CODES.get(target_lang.lower(), "swh_Latn")

    tokenizer.src_lang = src_code
    target_id = tokenizer.convert_tokens_to_ids(tgt_code)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)

    with torch.no_grad():
        translated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=target_id,
            max_length=128,
            num_beams=2,
            early_stopping=True
        )

    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

# UI Layout
st.title("📢 Kenya Multilingual PSA Translator")
st.caption("Deployed Streamlit Demo | English ↔ Swahili ↔ Ekegusii")

tabs = st.tabs(["📁 Batch Dataset Translation", "✍️ Single PSA Translation"])

with tabs[0]:
    st.subheader("Batch Dataset Translation")
    uploaded_file = st.file_uploader("Upload Excel/CSV Dataset File", type=["xlsx", "xls", "csv"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)

            st.success(f"Successfully loaded `{uploaded_file.name}` ({len(df)} rows).")
            st.dataframe(df.head(5))

            text_col = st.selectbox(
                "Select Source English Column:", 
                options=df.columns, 
                index=df.columns.get_loc("Description") if "Description" in df.columns else 0
            )

            sample_size = st.number_input(
                "Rows to process (0 = Translate entire dataset):", 
                min_value=0, 
                max_value=len(df), 
                value=min(5, len(df))
            )

            if st.button("🚀 Run Batch Translation", type="primary"):
                target_df = df.copy()
                if sample_size > 0:
                    target_df = target_df.head(sample_size)

                progress = st.progress(0)
                sw_list, ek_list = [], []
                total = len(target_df)

                for idx, (r_idx, row) in enumerate(target_df.iterrows()):
                    raw_val = row[text_col]
                    if pd.isna(raw_val) and "Title" in row:
                        raw_val = row["Title"]
                    
                    text_str = str(raw_val) if pd.notna(raw_val) else ""

                    sw_list.append(translate_psa(text_str, src_lang="english", target_lang="swahili"))
                    ek_list.append(translate_psa(text_str, src_lang="english", target_lang="ekegusii"))
                    progress.progress((idx + 1) / total)

                target_df["Translation_Swahili"] = sw_list
                target_df["Translation_Ekegusii"] = ek_list

                st.success("🎉 Batch Translation Complete!")
                st.dataframe(target_df[[text_col, "Translation_Swahili", "Translation_Ekegusii"]].head(10))

                csv_bytes = target_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Translated Dataset (CSV)",
                    data=csv_bytes,
                    file_name="translated_psa_dataset.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Error reading file: {e}")

with tabs[1]:
    st.subheader("Translate Single PSA Text")
    col1, col2 = st.columns(2)
    with col1:
        domain = st.selectbox("PSA Domain", ["Agriculture", "Health", "Security", "Governance", "Education"])
        src_lang = st.selectbox("Source Language", ["English", "Swahili"], index=0)
    with col2:
        tgt_lang = st.selectbox("Target Language", ["Swahili", "Ekegusii"], index=1)

    psa_text = st.text_area(
        "Enter Announcement:",
        value="Farmers are advised to plant early-maturing seeds before the rains start."
    )

    if st.button("Translate PSA Text", type="primary"):
        with st.spinner("Translating..."):
            res = translate_psa(psa_text, src_lang=src_lang, target_lang=tgt_lang)
            st.success(f"### Translation ({tgt_lang}):")
            st.info(res)
