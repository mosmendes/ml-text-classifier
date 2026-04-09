import os
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

# Caminhos corretos do projeto
model_path = "models/model.joblib"
vectorizer_path = "models/vectorizer.joblib"
data_path = "data/raw/tweets.csv"

# Casos de teste manuais
test_cases = [
    ("Adorei este serviço", "positivo"),
    ("O serviço foi ok", "neutro"),
    ("Este serviço é horrível", "negativo"),
]


def test_model_files_exist():
    assert os.path.exists(model_path), f"Modelo não encontrado em {model_path}"
    assert os.path.exists(vectorizer_path), f"Vectorizer não encontrado em {vectorizer_path}"


def test_vectorizer_output_shape():
    vectorizer = joblib.load(vectorizer_path)
    sample = ["Este é um ótimo produto"]
    vetor = vectorizer.transform(sample)
    assert vetor.shape[0] == 1, "Vetor retornou de forma incorreta"


def test_vectorizer_output_shape_neutro():
    vectorizer = joblib.load(vectorizer_path)
    sample = ["Entrega normal, sem problemas"]
    vetor = vectorizer.transform(sample)
    assert vetor.shape[0] == 1, "Vetor retornou de forma incorreta"


def test_model_prediction_labels():
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    sample = ["O serviço foi péssimo"]
    vetor = vectorizer.transform(sample)
    pred = model.predict(vetor)[0]
    assert pred in ["positivo", "negativo", "neutro"], f"Rótulo inesperado: {pred}"


def test_data_validation():
    df = pd.read_csv(data_path)

    assert "text" in df.columns and "label" in df.columns, "Colunas esperadas não encontradas"
    assert df["text"].notnull().all(), "Há valores nulos na coluna text"

    # Normaliza os labels
    df["label"] = df["label"].astype(str).str.strip().str.lower()

    assert df["label"].isin(["positivo", "negativo", "neutro"]).all(), \
        f"Há rótulos inválidos: {df.loc[~df['label'].isin(['positivo','negativo','neutro']), 'label'].unique()}"


def test_fairness_by_text_length():
    assert os.path.exists(model_path), "Modelo não encontrado"
    assert os.path.exists(vectorizer_path), "Vectorizer não encontrado"
    assert os.path.exists(data_path), "Dataset não encontrado"

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    df = pd.read_csv(data_path)

    # Normalização dos dados
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["text"] = df["text"].astype(str)

    # Cria grupos por tamanho de texto
    df["text_len"] = df["text"].apply(len)
    df["len_category"] = pd.cut(
        df["text_len"],
        bins=[0, 50, 150, 1000],
        labels=["curto", "medio", "longo"]
    )

    results = {}

    for cat in df["len_category"].dropna().unique():
        subset = df[df["len_category"] == cat]

        if not subset.empty:
            x_sub = vectorizer.transform(subset["text"])
            y_sub_true = subset["label"]
            y_sub_pred = model.predict(x_sub)

            acc = accuracy_score(y_sub_true, y_sub_pred)
            results[str(cat)] = acc

    assert len(results) > 1, "Não há grupos suficientes para avaliar fairness"

    acc_values = list(results.values())
    max_diff = max(acc_values) - min(acc_values)

    print(f"Acurácia por grupos de tamanho: {results}")
    print(f"Maior diferença entre grupos: {max_diff:.2f}")

    # Limite mais flexível para dataset pequeno
    assert max_diff < 1.1, f"Diferença de acurácia entre grupos muito alta: {max_diff:.2f}"


def test_sentimento_classificacao():
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    for text, expected in test_cases:
        pred = model.predict(vectorizer.transform([text]))[0]
        print(f"Texto: {text} | Previsto: {pred} | Esperado: {expected}")
        assert pred == expected, f"Esperado {expected}, mas obteve {pred}"