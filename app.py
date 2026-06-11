import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

st.set_page_config(page_title="Fraud Detection", page_icon="🛡️", layout="centered")

# --- загрузка модели и примеров ---
@st.cache_resource
def load_model():
    data = joblib.load("fraud_model.joblib")
    return data["model"], data["columns"]

@st.cache_data
def load_examples():
    with open("examples.json") as f:
        return json.load(f)

model, columns = load_model()
examples = load_examples()

# --- препроцессинг (тот же, что при обучении: log Amount, drop Time/Amount) ---
def preprocess(row: dict) -> pd.DataFrame:
    df = pd.DataFrame([row])
    df["Amount_log"] = np.log1p(df["Amount"])
    df = df.drop(["Time", "Amount"], axis=1)
    return df[columns]  # порядок колонок как при обучении

# --- интерфейс ---
st.title("🛡️ Credit Card Fraud Detection")
st.markdown(
    "Модель **LightGBM** для детекции мошеннических транзакций. "
    "Данные сильно несбалансированы (**1 мошенник на 578 честных**), "
    "поэтому качество измеряется через **PR-AUC = 0.876**, а не accuracy "
    "(которая у бесполезной модели тут была бы 99.8%)."
)

st.divider()

# --- выбор примера ---
kind_map = {
    "Мошенническая": "fraud",
    "Честная": "honest",
    "Сомнительная": "borderline",
}

col1, col2 = st.columns(2)
with col1:
    kind = st.radio("Тип транзакции для теста:", list(kind_map.keys()))

# динамический список примеров — столько, сколько реально есть в этой категории
n_examples = len(examples[kind_map[kind]])
with col2:
    idx = st.selectbox("Пример №", list(range(n_examples)))

sample = examples[kind_map[kind]][idx]

if kind == "Сомнительная":
    st.info(
        "Это пример, в котором модель **не уверена** (вероятность между 0.2 и 0.8). "
        "Подвигай порог ниже — и решение переключится с «Пропустить» на «Блокировать». "
        "Именно здесь виден смысл порога."
    )

# --- порог ---
st.subheader("Порог классификации")
st.markdown(
    "Порог определяет, при какой вероятности блокировать транзакцию. "
    "Он зависит от **бизнес-стоимости ошибок**: пропустить мошенника дорого "
    "(теряем сумму транзакции), ложно заблокировать честного — дёшево "
    "(звонок в поддержку). Поэтому оптимальный порог обычно **ниже 0.5**."
)
threshold = st.slider("Порог", 0.0, 1.0, 0.5, 0.01)

# --- предсказание ---
X = preprocess(sample)
proba = float(model.predict_proba(X)[0, 1])
pred = int(proba >= threshold)

st.divider()
st.subheader("Результат")

c1, c2 = st.columns(2)
c1.metric("Вероятность мошенничества", f"{proba:.1%}")
c2.metric("Решение", "🚫 БЛОКИРОВАТЬ" if pred else "✅ Пропустить")

st.caption(f"Сумма транзакции: ${sample['Amount']:.2f}")

if pred:
    st.error("Транзакция помечена как подозрительная и отправлена на проверку.")
else:
    st.success("Транзакция выглядит легитимной.")

st.divider()
st.caption(
    "Модель: LightGBM с подбором гиперпараметров через Optuna. "
    "Метрика PR-AUC на отложенном тесте: 0.876. "
    "Датасет: Credit Card Fraud Detection (ULB), 284 807 транзакций."
)