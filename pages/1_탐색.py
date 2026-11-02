```python
import streamlit as st
import pandas as pd
import plotly.express as px

# ==========================================
# 페이지 설정
# ==========================================
st.set_page_config(
    page_title="탐색 | 뇌졸중 예측 실습실",
    page_icon="📊",
    layout="wide"
)

# ==========================================
# 데이터 불러오기
# ==========================================
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"


@st.cache_data
def load_data():
    data = pd.read_csv(DATA_URL, encoding="utf-8")
    return data


try:
    df = load_data()
except Exception as e:
    st.error("데이터를 불러오는 중 오류가 발생했습니다.")
    st.exception(e)
    st.stop()


# ==========================================
# 제목
# ==========================================
st.title("📊 뇌졸중 예측 실습실 - 데이터 탐색")
st.write("뇌졸중 데이터의 분포와 여러 변수 사이의 관계를 탐색합니다.")

st.divider()


# ==========================================
# 1. 나이와 평균 혈당 분포
# ==========================================
st.header("1. 나이와 평균 혈당의 분포")

col1, col2 = st.columns(2)

with col1:
    age_fig = px.histogram(
        df,
        x="age",
        nbins=30,
        title="나이 분포",
        labels={
            "age": "나이",
            "count": "사람 수"
        }
    )

    age_fig.update_layout(
        xaxis_title="나이",
        yaxis_title="사람 수"
    )

    st.plotly_chart(age_fig, use_container_width=True)


with col2:
    glucose_fig = px.histogram(
        df,
        x="avg_glucose_level",
        nbins=30,
        title="평균 혈당 분포",
        labels={
            "avg_glucose_level": "평균 혈당",
            "count": "사람 수"
        }
    )

    glucose_fig.update_layout(
        xaxis_title="평균 혈당",
        yaxis_title="사람 수"
    )

    st.plotly_chart(glucose_fig, use_container_width=True)


st.divider()


# ==========================================
# 2. 뇌졸중 유무에 따른 나이와 혈당 비교
# ==========================================
st.header("2. 뇌졸중 유무에 따른 비교")

df["stroke_label"] = df["stroke"].map({
    0: "뇌졸중 없음",
    1: "뇌졸중 있음"
})

col3, col4 = st.columns(2)

with col3:
    age_box = px.box(
        df,
        x="stroke_label",
        y="age",
        color="stroke_label",
        category_orders={
            "stroke_label": [
                "뇌졸중 없음",
                "뇌졸
