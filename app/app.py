import streamlit as st
import requests

st.set_page_config(page_title="Classificador de Sentimentos", layout="centered")

st.title("Classificador de Sentimentos")
st.write("Digite um texto para análise de sentimento.")

texto = st.text_area("Digite um texto:", height=150)

API_URL = "http://api:8000/predict"

if st.button("Analisar"):
    if texto.strip():
        try:
            response = requests.post(API_URL, json={"text": texto})
            if response.status_code == 200:
                resultado = response.json()["prediction"]
                st.success(f"Sentimento previsto: {resultado}")
            else:
                st.error(f"Erro na API: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Erro ao conectar com a API: {e}")
    else:
        st.warning("Por favor, insira um texto para análise.")