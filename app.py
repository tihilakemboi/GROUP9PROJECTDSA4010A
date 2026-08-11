import os
import torch
import pandas as pd
import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

st.set_page_config(
    page_title="Kenya Multilingual PSA Translator",
    page_icon="📢",
    layout="wide"
)

MODEL_PATH = "./models/nllb200_swahili_ekegusii_ft"
FALLBACK_PATH = "facebook/nllb-200-distilled-600M"

@st.cache_resource
def load_app_model():
    load_path = MODEL_PATH if os.path.exists(MODEL_PATH) else FALLBACK_PATH
    tokenizer = AutoTokenizer.from_pretrained(load_path, use_fast=False)
    
    if "ekg_Latn" not in tokenizer.additional_special_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": ["ekg_Latn"]})

    model = AutoModelForSeq2SeqLM.from_pretrained(load_path)
    model.resize_token_embeddings(len(tokenizer))
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    return tokenizer, model, load_path

tokenizer, model, active_path = load_app_model()

LANG_CODES = {
    "english": "eng_Latn",
    "swahili": "swh_Latn",
    "ekegusii": "ekg_Latn"
}

def translate_psa(text: str, src_lang: str = "english", target_lang: str = "swahili") -> str:
    if not isinstance(text, str) or not text.strip():
        return ""

    src_code = LANG_CODES.get(src_lang.lower(), "eng_Latn")
    tgt_code = LANG_CODES.get(target_lang.lower(), "swh_Latn")

    tokenizer.src_lang = src_code

    if hasattr(tokenizer, "lang_code_to_id") and tgt_code in tokenizer.lang_code_to_id:
        target_id = tokenizer.lang_code_to_id[tgt_code]
    else:
        target_id = tokenizer.convert_tokens_to_ids(tgt_code)
        if target_id is None or target_id == tokenizer.unk_token_id:
            target_id = tokenizer.convert_tokens_to_ids(f"__{tgt_code}__")

    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=128
    ).to(model.device)

    with torch.no_grad():
        translated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=target_id,
            max_length=128,
            num_beams=4,
            early_stopping=True
        )

    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

st.title("📢 Kenya Multilingual PSA Translator")
st.caption(f"Active Checkpoint: `{active_path}` | Device: `{model.device.type.upper()}`")

tabs = st.tabs(["📁 Direct File Upload (Excel/CSV)", "✍️ Single PSA Demo"])

with tabs[0]:
    st.subheader("Batch Dataset Translation")
    st.write("Upload `Fial Project Dataset.xlsx` directly from your PC or Drive.")

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
                "Select English Source Text Column:", 
                options=df.columns, 
                index=df.columns.get_loc("Description") if "Description" in df.columns else 0
            )

            sample_size = st.number_input(
                "Rows to process (0 = Translate entire dataset):", 
                min_value=0, 
                max_value=len(df), 
                value=min(10, len(df))
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
                st.dataframe(target_df.head(10))

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
        value="[Agriculture Alert] Farmers are advised to plant early-maturing seeds before the rains start."
    )

    if st.button("Translate PSA Text", type="primary"):
        with st.spinner("Translating..."):
            res = translate_psa(psa_text, src_lang=src_lang, target_lang=tgt_lang)
            st.success("### Translation:")
            st.info(res)
