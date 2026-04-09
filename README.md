# ML Text Classifier

Projeto de **classificação de sentimentos em textos curtos** utilizando **Machine Learning**, com:

- **API REST em FastAPI**
- **Interface Web em Streamlit**
- **Containerização com Docker**
- **Testes automatizados com Pytest**

Este projeto foi desenvolvido com foco em práticas iniciais de **MLOps**, incluindo organização de artefatos, testes, API, interface de uso e execução em ambiente conteinerizado.

---

## Objetivo

O objetivo deste projeto é classificar textos em três categorias de sentimento:

- **positivo**
- **negativo**
- **neutro**

O sistema permite que o usuário envie um texto e receba a predição do sentimento por meio de:

- uma **API REST**
- uma **interface gráfica simples com Streamlit**

---

## Arquitetura da Solução

O projeto possui dois serviços principais:

### 1) API (FastAPI)
Responsável por:
- carregar o modelo treinado
- receber textos
- retornar a classificação de sentimento

### 2) App Web (Streamlit)
Responsável por:
- disponibilizar uma interface amigável
- permitir testes manuais do modelo
- consumir a API de predição

---

## Estrutura do Projeto

```bash
ml-text-classifier/
│
├── api/                    # Código da API FastAPI
│   └── main.py
│
├── app/                    # Interface web em Streamlit
│   └── app.py
│
├── data/                   # Base de dados utilizada
│   └── tweets_limpo.csv
│
├── models/                 # Artefatos do modelo treinado
│   ├── model.joblib
│   └── vectorizer.joblib
│
├── notebooks/              # Exploração e experimentação
├── src/                    # Código-fonte do pipeline/modelo
│
├── tests/                  # Testes automatizados
│   ├── test_api.py
│   └── test_pipeline.py
│
├── Dockerfile              # Configuração da imagem Docker
├── docker-compose.yml      # Orquestração dos serviços
├── requirements.txt        # Dependências do projeto
└── README.md               # Documentação do projeto
```

---

## Tecnologias Utilizadas

- **Python 3.11**
- **Pandas**
- **Scikit-learn**
- **Joblib**
- **FastAPI**
- **Uvicorn**
- **Streamlit**
- **Pytest**
- **Docker**
- **Docker Compose**

---

## Como Executar o Projeto

### 1) Clonar o repositório

```bash
git clone <URL_DO_REPOSITORIO>
cd ml-text-classifier
```

---

### 2) Subir os containers

```bash
docker compose up --build
```

Esse comando sobe os dois serviços:

- **API** → `http://localhost:8000`
- **App Web** → `http://localhost:8501`

---

## Como Usar

---

## API REST

### URL base

```bash
http://localhost:8000
```

### Endpoint principal

```bash
POST /predict
```

### Exemplo de requisição

```json
{
  "text": "Adorei este serviço"
}
```

### Exemplo de resposta

```json
{
  "prediction": "positivo"
}
```

---

## Documentação da API

A documentação automática da API pode ser acessada em:

### Swagger UI
```bash
http://localhost:8000/docs
```

### ReDoc
```bash
http://localhost:8000/redoc
```

---

## Interface Web (Streamlit)

A aplicação web pode ser acessada em:

```bash
http://localhost:8501
```

Nela, o usuário pode:

- digitar um texto
- enviar para classificação
- visualizar o sentimento previsto

---

## Como Executar os Testes

Para rodar os testes automatizados no container da API:

```bash
docker exec -it ml-text-classifier-api pytest
```

### Resultado esperado

```bash
10 passed
```

---

## Cobertura dos Testes

O projeto possui testes para:

### Testes da API
- disponibilidade da aplicação
- funcionamento do endpoint `/predict`
- validação de retorno da predição

### Testes do Pipeline
- existência dos arquivos do modelo
- existência do vetorizador
- funcionamento do vetor de entrada
- validação dos rótulos previstos
- validação da base de dados
- teste com exemplos manuais de sentimento
- avaliação simples de comportamento por tamanho de texto

---

## Exemplo de Classes de Sentimento

| Texto | Classe Esperada |
|------|------|
| Adorei este serviço | positivo |
| O serviço foi ok | neutro |
| Este serviço é horrível | negativo |

---

## Organização dos Artefatos

Os principais artefatos do modelo ficam na pasta `models/`:

- `model.joblib` → modelo treinado
- `vectorizer.joblib` → vetorizador de texto

Esses arquivos são utilizados tanto pela API quanto pelos testes automatizados.

---

## Docker Compose

O projeto utiliza dois serviços definidos no `docker-compose.yml`:

### Serviço `api`
- executa a API FastAPI
- expõe a porta `8000`

### Serviço `app`
- executa a interface Streamlit
- expõe a porta `8501`
- depende da API para funcionamento

---

## Possíveis Melhorias Futuras

Este projeto pode evoluir para um fluxo mais robusto de MLOps, incluindo:

- pipeline de treinamento automatizado
- versionamento de modelo
- monitoramento de predições
- avaliação de drift
- CI/CD com GitHub Actions
- deploy em nuvem
- rastreamento de experimentos com MLflow

---

## Aprendizados Aplicados

Este projeto demonstra prática em:

- estruturação de projeto de Machine Learning
- serialização de modelo com Joblib
- exposição de modelo via API
- construção de interface web simples
- testes automatizados de pipeline e serviço
- conteinerização com Docker
- organização inicial voltada a MLOps

---

## Autora

**Mônica Mendes**  
Projeto acadêmico/prático com foco em **Machine Learning em Produção / MLOps**.

---