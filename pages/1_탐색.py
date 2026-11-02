    })
    .round(2)
)
mean_table.index.name = "뇌졸중 여부"

st.subheader("뇌졸중 유무에 따른 평균값")
st.dataframe(mean_table, use_container_width=True)

st.divider()

# --------------------------------------------------
# 3. 고혈압 및 심장병 유무에 따른 뇌졸중 비율
# --------------------------------------------------
st.header("3. 질병 유무에 따른 뇌졸중 비율")
st.write("각 그룹에서 stroke가 1인 사람의 비율을 계산해 비교합니다.")

def make_rate_table(column, labels):
    grouped = df.groupby(column, dropna=False)["stroke"].agg(
        전체_사람_수="count",
        뇌졸중_사람_수="sum",
        뇌졸중_비율="mean"
    ).reset_index()

    grouped["구분"] = grouped[column].map(labels)
    grouped["뇌졸중_비율(%)"] = grouped["뇌졸중_비율"] * 100
    return grouped


hypertension_table = make_rate_table(
    "hypertension",
    {0: "고혈압 없음", 1: "고혈압 있음"}
)
heart_table = make_rate_table(
    "heart_disease",
    {0: "심장병 없음", 1: "심장병 있음"}
)

rate_col1, rate_col2 = st.columns(2)

with rate_col1:
    hypertension_fig = px.bar(
        hypertension_table,
        x="구분",
        y="뇌졸중_비율(%)",
        text="뇌졸중_비율(%)",
        title="고혈압 유무에 따른 뇌졸중 비율",
        labels={"구분": "고혈압 여부", "뇌졸중_비율(%)": "뇌졸중 비율(%)"},
    )
    hypertension_fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    hypertension_fig.update_layout(yaxis_title="뇌졸중 비율(%)", xaxis_title="")
    st.plotly_chart(hypertension_fig, use_container_width=True)

with rate_col2:
    heart_fig = px.bar(
        heart_table,
        x="구분",
        y="뇌졸중_비율(%)",
        text="뇌졸중_비율(%)",
        title="심장병 유무에 따른 뇌졸중 비율",
        labels={"구분": "심장병 여부", "뇌졸중_비율(%)": "뇌졸중 비율(%)"},
    )
    heart_fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    heart_fig.update_layout(yaxis_title="뇌졸중 비율(%)", xaxis_title="")
    st.plotly_chart(heart_fig, use_container_width=True)

st.divider()

# --------------------------------------------------
# 4. BMI 결측값이 있는 사람들의 뇌졸중 비율
# --------------------------------------------------
st.header("4. BMI 빈 값과 뇌졸중 비율")
st.write("BMI 값이 비어 있는 사람들만 따로 구분해, 전체 데이터의 뇌졸중 비율과 비교합니다.")

missing_bmi = df[df["bmi"].isna()]
missing_bmi_count = len(missing_bmi)
missing_bmi_stroke_rate = (
    missing_bmi["stroke"].mean() * 100 if missing_bmi_count > 0 else 0
)
overall_stroke_rate = df["stroke"].mean() * 100 if len(df) > 0 else 0

bmi_compare = pd.DataFrame({
    "구분": ["BMI 빈 값인 사람", "전체 사람"],
    "사람 수": [missing_bmi_count, len(df)],
    "뇌졸중인 사람 수": [
        int(missing_bmi["stroke"].sum()),
        int(df["stroke"].sum())
    ],
    "뇌졸중 비율(%)": [
        round(missing_bmi_stroke_rate, 2),
        round(overall_stroke_rate, 2)
    ]
})
st.dataframe(bmi_compare, hide_index=True, use_container_width=True)

st.divider()

# --------------------------------------------------
# 5. 흡연 상태별 사람 수
# --------------------------------------------------
st.header("5. 흡연 상태별 사람 수")
st.write("흡연 상태의 각 값에 해당하는 사람 수를 셉니다. 빈 값이 있다면 별도로 표시합니다.")

smoking_counts = (
    df["smoking_status"]
    .fillna("빈 값")
    .value_counts(dropna=False)
    .rename_axis("흡연 상태")
    .reset_index(name="사람 수")
)

st.dataframe(smoking_counts, hide_index=True, use_container_width=True)

st.caption(
    "주의: 이 페이지는 데이터의 분포와 그룹별 비율을 탐색하는 실습용입니다. "
    "그래프에서 나타나는 차이만으로 특정 요인이 뇌졸중의 원인이라고 단정할 수는 없습니다."
)
