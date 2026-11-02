st.header("3. 질병 유무에 따른 뇌졸중 비율")
st.write("각 그룹에서 stroke가 1인 사람의 비율을 계산합니다.")


def calculate_stroke_rate(column_name, label_map):
    result = df.groupby(column_name)["stroke"].agg(["count", "sum", "mean"])
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
