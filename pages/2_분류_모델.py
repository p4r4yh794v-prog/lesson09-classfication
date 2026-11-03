import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# --------------------------------------------------
# 1. 페이지 설정 및 데이터 불러오기
# --------------------------------------------------

st.set_page_config(
    page_title="분류 모델 | 뇌졸중 예측 실습실",
    page_icon="🤖",
    layout="wide"
)

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"

FEATURE_NAMES = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병"
}

FEATURES = list(FEATURE_NAMES.keys())


@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


st.title("🤖 분류 모델")
st.caption("머신러닝 모델을 이용해 뇌졸중 여부를 예측합니다.")

try:
    df = load_data()
except Exception as error:
    st.error("데이터를 불러오지 못했습니다. 인터넷 연결이나 데이터 주소를 확인해 주세요.")
    st.code(str(error))
    st.stop()


# --------------------------------------------------
# 2. 사용할 속성 선택
# --------------------------------------------------

st.header("1. 모델에 사용할 속성 선택")

selected_features = st.multiselect(
    "예측에 사용할 속성을 선택하세요.",
    options=FEATURES,
    default=[
        "age",
        "avg_glucose_level",
        "hypertension",
        "heart_disease"
    ],
    format_func=lambda x: FEATURE_NAMES[x]
)

if len(selected_features) < 2:
    st.warning("모델을 만들려면 속성을 두 개 이상 선택해야 합니다.")
    st.stop()


# --------------------------------------------------
# 3. 훈련 데이터와 테스트 데이터 분리
# --------------------------------------------------

st.header("2. 훈련 데이터와 테스트 데이터 분리")

# 원본 데이터의 순서를 유지한다.
df = df.reset_index(drop=True)
df["row_number"] = np.arange(len(df))

# 10명씩 묶었을 때 앞의 3명은 테스트 데이터,
# 나머지 7명은 훈련 데이터로 사용한다.
test_mask = (df["row_number"] % 10) < 3

train_df = df.loc[~test_mask].copy()
test_df = df.loc[test_mask].copy()

X_train = train_df[selected_features].copy()
X_test = test_df[selected_features].copy()

# stroke: 1은 뇌졸중(양성), 0은 뇌졸중 아님(음성)
y_train = train_df["stroke"].astype(int)
y_test = test_df["stroke"].astype(int)

st.write(
    f"전체 데이터: **{len(df):,}명**  |  "
    f"훈련 데이터: **{len(train_df):,}명**  |  "
    f"테스트 데이터: **{len(test_df):,}명**"
)

if len(test_df) != 1533:
    st.warning(
        f"현재 데이터의 테스트 인원은 {len(test_df):,}명입니다. "
        "데이터가 5,110명일 때 테스트 인원은 1,533명입니다."
    )


# --------------------------------------------------
# 4. 훈련 데이터만 이용한 결측값 처리와 표준화
# --------------------------------------------------

st.header("3. 데이터 전처리")

# BMI를 선택한 경우에만 훈련 데이터의 중앙값으로 결측값을 채운다.
if "bmi" in selected_features:
    bmi_median = X_train["bmi"].median()

    X_train.loc[:, "bmi"] = X_train["bmi"].fillna(bmi_median)
    X_test.loc[:, "bmi"] = X_test["bmi"].fillna(bmi_median)

    st.write(
        f"체질량지수의 빈 값은 훈련 데이터의 중앙값 "
        f"**{bmi_median:.2f}**으로 채웠습니다."
    )

# 선택한 속성 중 다른 결측값이 있는지 확인한다.
if X_train.isna().any().any() or X_test.isna().any().any():
    st.error("선택한 속성에 처리되지 않은 결측값이 있습니다.")
    st.stop()

# 표준화 기준은 훈련 데이터에서만 학습한다.
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

st.write("표준화 기준은 훈련 데이터에서만 계산했습니다.")


# --------------------------------------------------
# 5. 모델 학습
# --------------------------------------------------

st.header("4. 분류 모델 학습")

# 로지스틱 회귀
logistic_model = LogisticRegression(
    max_iter=2000,
    random_state=42
)

logistic_model.fit(X_train_scaled, y_train)

# 의사결정트리
tree_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=42
)

tree_model.fit(X_train_scaled, y_train)


# --------------------------------------------------
# 6. 훈련 데이터와 테스트 데이터 정확도 계산
# --------------------------------------------------

# 로지스틱 회귀 정확도
logistic_train_pred = logistic_model.predict(X_train_scaled)
logistic_test_pred = logistic_model.predict(X_test_scaled)

logistic_train_accuracy = accuracy_score(
    y_train, logistic_train_pred
)

logistic_test_accuracy = accuracy_score(
    y_test, logistic_test_pred
)

# 의사결정트리 정확도
tree_train_pred = tree_model.predict(X_train_scaled)
tree_test_pred = tree_model.predict(X_test_scaled)

tree_train_accuracy = accuracy_score(
    y_train, tree_train_pred
)

tree_test_accuracy = accuracy_score(
    y_test, tree_test_pred
)

# 아무 속성도 보지 않고 훈련 데이터에서 많은 쪽으로만 예측하는 모델
majority_class = y_train.mode()[0]

baseline_train_pred = np.full(
    len(y_train), majority_class
)

baseline_test_pred = np.full(
    len(y_test), majority_class
)

baseline_train_accuracy = accuracy_score(
    y_train, baseline_train_pred
)

baseline_test_accuracy = accuracy_score(
    y_test, baseline_test_pred
)


# --------------------------------------------------
# 7. 정확도 카드
# --------------------------------------------------

st.header("5. 모델별 정확도 비교")

def accuracy_card(
    title,
    test_accuracy,
    train_accuracy
):
    with st.container(border=True):
        st.subheader(title)

        st.metric(
            label="테스트 데이터 정확도",
            value=f"{test_accuracy * 100:.2f}%"
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "훈련 데이터",
                f"{train_accuracy * 100:.2f}%"
            )

        with col2:
            st.metric(
                "테스트 데이터",
                f"{test_accuracy * 100:.2f}%"
            )


col1, col2, col3 = st.columns(3)

with col1:
    accuracy_card(
        "로지스틱 회귀(확률로 답하는 모델)",
        logistic_test_accuracy,
        logistic_train_accuracy
    )

with col2:
    accuracy_card(
        "의사결정트리(질문으로 답하는 모델)",
        tree_test_accuracy,
        tree_train_accuracy
    )

with col3:
    accuracy_card(
        "기준 모델(다수결)",
        baseline_test_accuracy,
        baseline_train_accuracy
    )

st.caption(
    "기준 모델은 입력 속성을 전혀 사용하지 않고 "
    "훈련 데이터에서 더 많은 쪽으로만 예측합니다."
)


# --------------------------------------------------
# 8. 산점도 축 선택
# --------------------------------------------------

st.divider()
st.header("6. 예측 결과 시각화")

st.write("선택한 속성 중 두 개를 골라 가로축과 세로축으로 사용합니다.")

col_x, col_y = st.columns(2)

with col_x:
    x_feature = st.selectbox(
        "가로축",
        options=selected_features,
        index=0,
        format_func=lambda x: FEATURE_NAMES[x]
    )

with col_y:
    y_options = [
        feature for feature in selected_features
        if feature != x_feature
    ]

    y_feature = st.selectbox(
        "세로축",
        options=y_options,
        index=0,
        format_func=lambda x: FEATURE_NAMES[x]
    )


# --------------------------------------------------
# 9. 나머지 속성은 테스트 데이터 중앙값으로 고정
# --------------------------------------------------

fixed_values = {}

for feature in selected_features:
    if feature not in [x_feature, y_feature]:
        fixed_values[feature] = X_test[feature].median()

if fixed_values:
    fixed_text = ", ".join(
        f"{FEATURE_NAMES[feature]}: {value:.2f}"
        for feature, value in fixed_values.items()
    )

    st.info(
        "두 축에 사용하지 않는 속성은 테스트 데이터의 중앙값으로 고정했습니다.\n\n"
        + fixed_text
    )
else:
    st.info("선택한 두 속성 모두 그림의 축으로 사용합니다.")


# --------------------------------------------------
# 10. 예측 경계 계산
# --------------------------------------------------

# 테스트 데이터의 범위를 기준으로 격자 생성
x_min = X_test[x_feature].min()
x_max = X_test[x_feature].max()

y_min = X_test[y_feature].min()
y_max = X_test[y_feature].max()

# 값이 모두 같을 때를 대비해 범위를 조금 넓힌다.
x_margin = (x_max - x_min) * 0.05
y_margin = (y_max - y_min) * 0.05

if x_margin == 0:
    x_margin = 1

if y_margin == 0:
    y_margin = 1

x_range = np.linspace(
    x_min - x_margin,
    x_max + x_margin,
    150
)

y_range = np.linspace(
    y_min - y_margin,
    y_max + y_margin,
    150
)

xx, yy = np.meshgrid(x_range, y_range)

# 모든 선택 속성의 값을 준비한다.
grid_data = {}

for feature in selected_features:
    if feature == x_feature:
        grid_data[feature] = xx.ravel()

    elif feature == y_feature:
        grid_data[feature] = yy.ravel()

    else:
        grid_data[feature] = np.full(
            xx.size,
            fixed_values[feature]
        )

grid_df = pd.DataFrame(grid_data)[selected_features]

# 격자 데이터도 훈련 데이터에서 구한 표준화 기준을 사용한다.
grid_scaled = scaler.transform(grid_df)

# 로지스틱 회귀의 양성 확률
grid_probability = logistic_model.predict_proba(
    grid_scaled
)[:, 1].reshape(xx.shape)

# 의사결정트리의 예측 결과
grid_tree_prediction = tree_model.predict(
    grid_scaled
).reshape(xx.shape)


# --------------------------------------------------
# 11. Plotly 산점도와 모델의 경계
# --------------------------------------------------

plot_df = test_df.copy()

plot_df["실제 결과"] = plot_df["stroke"].map({
    0: "뇌졸중 아님(음성)",
    1: "뇌졸중(양성)"
})

# 의사결정트리의 예측 영역을 먼저 그린다.
fig = go.Figure()

fig.add_trace(
    go.Heatmap(
        x=x_range,
        y=y_range,
        z=grid_tree_prediction,
        colorscale=[
            [0, "rgba(80, 150, 220, 0.12)"],
            [1, "rgba(230, 90, 90, 0.12)"]
        ],
        showscale=False,
        hoverinfo="skip",
        name="의사결정트리 영역"
    )
)

# 로지스틱 회귀의 0.5 확률 경계
fig.add_trace(
    go.Contour(
        x=x_range,
        y=y_range,
        z=grid_probability,
        contours=dict(
            start=0.5,
            end=0.5,
            size=0.1,
            coloring="none"
        ),
        line=dict(
            color="black",
            width=3
        ),
        showscale=False,
        hoverinfo="skip",
        name="로지스틱 회귀 경계"
    )
)

# 실제 테스트 데이터 점
scatter_fig = px.scatter(
    plot_df,
    x=x_feature,
    y=y_feature,
    color="실제 결과",
    color_discrete_map={
        "뇌졸중 아님(음성)": "#3182CE",
        "뇌졸중(양성)": "#E53E3E"
    },
    category_orders={
        "실제 결과": [
            "뇌졸중 아님(음성)",
            "뇌졸중(양성)"
        ]
    },
    hover_data=["age", "avg_glucose_level", "bmi"]
)

for trace in scatter_fig.data:
    fig.add_trace(trace)

fig.update_layout(
    title="테스트 데이터와 분류 모델의 예측 영역",
    xaxis_title=FEATURE_NAMES[x_feature],
    yaxis_title=FEATURE_NAMES[y_feature],
    legend_title="실제 뇌졸중 여부",
    height=650
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# 0.5 경계가 화면 범위 안에 존재하는지 확인
prob_min = float(np.min(grid_probability))
prob_max = float(np.max(grid_probability))

if prob_min <= 0.5 <= prob_max:
    st.success(
        "검은 선은 로지스틱 회귀 모델이 "
        "뇌졸중 확률 0.5를 기준으로 나눈 경계입니다."
    )
else:
    st.warning(
        "로지스틱 회귀의 0.5 경계가 현재 그림의 범위 안에 나타나지 않습니다. "
        "따라서 경계선이 그림 밖에 있습니다."
    )

st.caption(
    "옅은 색 영역은 의사결정트리의 예측 영역입니다. "
    "점의 색은 모델의 예측이 아닌 실제 뇌졸중 여부를 나타냅니다."
)


# --------------------------------------------------
# 12. 의사결정트리 가지 그림
# --------------------------------------------------

st.divider()
st.header("7. 의사결정트리의 질문 과정")

st.write(
    "의사결정트리는 속성에 관한 질문을 최대 세 번까지 하며 "
    "답에 따라 뇌졸중 여부를 분류합니다."
)

tree_structure = tree_model.tree_

# 각 마디에 도달한 훈련 데이터의 수와 양성 수 계산
train_path = tree_model.decision_path(X_train_scaled)

node_sample_counts = np.asarray(
    train_path.sum(axis=0)
).ravel().astype(int)

node_positive_counts = np.asarray(
    train_path.T @ y_train.to_numpy()
).ravel().astype(int)

used_features = set()
leaf_count = 0
negative_leaf_count = 0

dot_lines = [
    "digraph DecisionTree {",
    'graph [rankdir=TB, bgcolor="white", nodesep="0.35", ranksep="0.55"];',
    'node [shape=box, style="rounded,filled", fontname="Malgun Gothic", fontsize=10, margin="0.16"];',
    'edge [fontname="Malgun Gothic", fontsize=10];'
]


def make_tree_node(node_id):
    global leaf_count, negative_leaf_count

    sample_count = int(node_sample_counts[node_id])
    positive_count = int(node_positive_counts[node_id])

    positive_rate = (
        positive_count / sample_count * 100
        if sample_count > 0 else 0
    )

    is_leaf = (
        tree_structure.children_left[node_id]
        == tree_structure.children_right[node_id]
    )

    if is_leaf:
        leaf_count += 1

        prediction = int(tree_model.classes_[
            np.argmax(tree_structure.value[node_id][0])
        ])

        if prediction == 1:
            answer = "뇌졸중 있음 (양성)"
            fill_color = "#F8B4B4"
        else:
            answer = "뇌졸중 아님 (음성)"
            fill_color = "#A9D2F5"
            negative_leaf_count += 1

        label = (
            f"{answer}\\n"
            f"훈련용: {sample_count}명\\n"
            f"뇌졸중: {positive_count}명\\n"
            f"비율: {positive_rate:.1f}%"
        )

        dot_lines.append(
            f'{node_id} [label="{label}", fillcolor="{fill_color}"];'
        )

        return

    feature_index = tree_structure.feature[node_id]
    feature_name = selected_features[feature_index]

    used_features.add(feature_name)

    # 표준화된 기준값을 원래 단위로 바꾼다.
    threshold_scaled = tree_structure.threshold[node_id]

    threshold_original = (
        threshold_scaled * scaler.scale_[feature_index]
        + scaler.mean_[feature_index]
    )

    question = (
        f"{FEATURE_NAMES[feature_name]} ≤ "
        f"{threshold_original:.2f} ?"
    )

    label = (
        f"{question}\\n"
        f"훈련용: {sample_count}명\\n"
        f"뇌졸중: {positive_count}명\\n"
        f"비율: {positive_rate:.1f}%"
    )

    dot_lines.append(
        f'{node_id} [label="{label}", fillcolor="#FFF2CC"];'
    )

    left_child = tree_structure.children_left[node_id]
    right_child = tree_structure.children_right[node_id]

    make_tree_node(left_child)
    make_tree_node(right_child)

    dot_lines.append(
        f'{node_id} -> {left_child} [label="예"];'
    )

    dot_lines.append(
        f'{node_id} -> {right_child} [label="아니요"];'
    )


make_tree_node(0)

dot_lines.append("}")

dot_string = "\n".join(dot_lines)

st.graphviz_chart(
    dot_string,
    use_container_width=True
)


# --------------------------------------------------
# 13. 의사결정트리 요약
# --------------------------------------------------

st.subheader("의사결정트리 요약")

st.write(
    f"① 최종적으로 답을 내리는 마디는 **{leaf_count}칸**이며, "
    f"그중 **{negative_leaf_count}칸**은 뇌졸중이 아니라고 답합니다."
)

if used_features:
    used_feature_names = [
        FEATURE_NAMES[feature]
        for feature in selected_features
        if feature in used_features
    ]

    st.write(
        "② 이 나무가 실제로 질문한 속성: "
        + ", ".join(used_feature_names)
    )
else:
    st.write("② 이 나무는 별도의 속성 질문을 사용하지 않았습니다.")


# --------------------------------------------------
# 14. 마무리 안내
# --------------------------------------------------

st.divider()

st.caption(
    "주의: 이 페이지는 머신러닝 분류 원리를 이해하기 위한 교육용 실습입니다. "
    "정확도가 높더라도 실제 환자의 뇌졸중 여부를 진단하는 데 사용할 수 없습니다."
)
