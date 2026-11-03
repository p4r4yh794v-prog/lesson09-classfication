import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="탐색 | 뇌졸중 예측 실습실",
    page_icon="📊",
    layout="wide"
)

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"


@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


st.title("📊 데이터 탐색")
st.caption("나이, 평균 혈당, 질병 여부, 흡연 상태에 따른 데이터를 살펴봅니다.")

try:
    df = load_data()
except Exception as error:
    st.error("데이터를 불러오지 못했습니다. 인터넷 연결이나 데이터 주소를 확인해 주세요.")
    st.code(str(error))
    st.stop()

df["stroke_label"] = df["stroke"].map({
    0: "뇌졸중 없음",
    1: "뇌졸중 있음"
})


# 1. 나이와 평균 혈당 분포
st.header("1. 나이와 평균 혈당의 분포")
st.write("각 변수의 값이 어떤 구간에 얼마나 분포하는지 히스토그램으로 확인합니다.")

col_age, col_glucose = st.columns(2)

with col_age:
    fig_age = px.histogram(
        df,
        x="age",
        nbins=30,
        title="나이 분포",
        labels={"age": "나이", "count": "사람 수"}
    )
    fig_age.update_layout(
        xaxis_title="나이",
        yaxis_title="사람 수",
        bargap=0.08
    )
    st.plotly_chart(fig_age, use_container_width=True)

with col_glucose:
    fig_glucose = px.histogram(
        df,
        x="avg_glucose_level",
        nbins=30,
        title="평균 혈당 분포",
        labels={"avg_glucose_level": "평균 혈당", "count": "사람 수"}
    )
    fig_glucose.update_layout(
        xaxis_title="평균 혈당",
        yaxis_title="사람 수",
        bargap=0.08
    )
    st.plotly_chart(fig_glucose, use_container_width=True)

st.divider()


# 2. 뇌졸중 유무에 따른 나이와 평균 혈당 비교
st.header("2. 뇌졸중 유무에 따른 비교")
st.write("두 그룹의 나이와 평균 혈당 분포를 상자그림으로 비교합니다.")

col_age_box, col_glucose_box = st.columns(2)

with col_age_box:
    fig_age_box = px.box(
        df,
        x="stroke_label",
        y="age",
        color="stroke_label",
        category_orders={
            "stroke_label": ["뇌졸중 없음", "뇌졸중 있음"]
        },
        title="뇌졸중 유무에 따른 나이",
        labels={"stroke_label": "뇌졸중 여부", "age": "나이"}
    )
    fig_age_box.update_layout(
        xaxis_title="뇌졸중 여부",
        yaxis_title="나이",
        showlegend=False
    )
    st.plotly_chart(fig_age_box, use_container_width=True)

with col_glucose_box:
    fig_glucose_box = px.box(
        df,
        x="stroke_label",
        y="avg_glucose_level",
        color="stroke_label",
        category_orders={
            "stroke_label": ["뇌졸중 없음", "뇌졸중 있음"]
        },
        title="뇌졸중 유무에 따른 평균 혈당",
        labels={
            "stroke_label": "뇌졸중 여부",
            "avg_glucose_level": "평균 혈당"
        }
    )
    fig_glucose_box.update_layout(
        xaxis_title="뇌졸중 여부",
        yaxis_title="평균 혈당",
        showlegend=False
    )
    st.plotly_chart(fig_glucose_box, use_container_width=True)


mean_table = df.groupby("stroke_label")[
    ["age", "avg_glucose_level"]
].mean()

mean_table = mean_table.reindex([
    "뇌졸중 없음",
    "뇌졸중 있음"
])

mean_table = mean_table.rename(columns={
    "age": "평균 나이",
    "avg_glucose_level": "평균 혈당"
}).round(2)

mean_table.index.name = "뇌졸중 여부"

st.subheader("뇌졸중 유무에 따른 평균값")
st.dataframe(mean_table, use_container_width=True)

st.divider()


# 3. 고혈압과 심장병 유무에 따른 뇌졸중 비율
st.header("3. 질병 유무에 따른 뇌졸중 비율")
st.write("각 그룹에서 stroke가 1인 사람의 비율을 계산합니다.")


def calculate_stroke_rate(column_name, label_map):
    result = df.groupby(column_name)["stroke"].agg(
        ["count", "sum", "mean"]
    )
    result = result.reset_index()
    result["구분"] = result[column_name].map(label_map)
    result["뇌졸중 비율(%)"] = result["mean"] * 100
    return result


hypertension_data = calculate_stroke_rate(
    "hypertension",
    {0: "고혈압 없음", 1: "고혈압 있음"}
)

heart_data = calculate_stroke_rate(
    "heart_disease",
    {0: "심장병 없음", 1: "심장병 있음"}
)

col_hypertension, col_heart = st.columns(2)

with col_hypertension:
    fig_hypertension = px.bar(
        hypertension_data,
        x="구분",
        y="뇌졸중 비율(%)",
        text="뇌졸중 비율(%)",
        title="고혈압 유무에 따른 뇌졸중 비율",
        labels={"구분": "고혈압 여부"}
    )
    fig_hypertension.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )
    fig_hypertension.update_layout(
        xaxis_title="",
        yaxis_title="뇌졸중 비율(%)"
    )
    st.plotly_chart(fig_hypertension, use_container_width=True)

with col_heart:
    fig_heart = px.bar(
        heart_data,
        x="구분",
        y="뇌졸중 비율(%)",
        text="뇌졸중 비율(%)",
        title="심장병 유무에 따른 뇌졸중 비율",
        labels={"구분": "심장병 여부"}
    )
    fig_heart.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )
    fig_heart.update_layout(
        xaxis_title="",
        yaxis_title="뇌졸중 비율(%)"
    )
    st.plotly_chart(fig_heart, use_container_width=True)

st.divider()


# 4. BMI가 비어 있는 사람들의 뇌졸중 비율
st.header("4. BMI 빈 값과 뇌졸중 비율")
st.write("BMI가 비어 있는 사람들과 전체 사람들의 뇌졸중 비율을 비교합니다.")

missing_bmi = df[df["bmi"].isna()]
missing_count = len(missing_bmi)

if missing_count > 0:
    missing_stroke_rate = missing_bmi["stroke"].mean() * 100
else:
    missing_stroke_rate = 0

overall_stroke_rate = df["stroke"].mean() * 100

bmi_comparison = pd.DataFrame({
    "구분": ["BMI 빈 값인 사람", "전체 사람"],
    "사람 수": [missing_count, len(df)],
    "뇌졸중인 사람 수": [
        int(missing_bmi["stroke"].sum()),
        int(df["stroke"].sum())
    ],
    "뇌졸중 비율(%)": [
        round(missing_stroke_rate, 2),
        round(overall_stroke_rate, 2)
    ]
})

st.dataframe(
    bmi_comparison,
    hide_index=True,
    use_container_width=True
)

st.divider()


# 5. 흡연 상태별 사람 수
st.header("5. 흡연 상태별 사람 수")
st.write("흡연 상태에 해당하는 사람 수를 셉니다. 빈 값도 별도로 표시합니다.")

smoking_counts = (
    df["smoking_status"]
    .fillna("빈 값")
    .value_counts()
    .rename_axis("흡연 상태")
    .reset_index(name="사람 수")
)

st.dataframe(
    smoking_counts,
    hide_index=True,
    use_container_width=True
)


st.caption(
    "주의: 이 페이지는 데이터 탐색 실습용입니다. "
    "그룹별 차이만으로 특정 요인이 뇌졸중의 원인이라고 단정할 수 없습니다."
)
