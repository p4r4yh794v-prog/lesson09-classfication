import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# --------------------------------------------------
# 1. 기본 설정
# --------------------------------------------------

st.set_page_config(
    page_title="성능 높이기 | 뇌졸중 예측 실습실",
    page_icon="📈",
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


st.title("📈 분류 모델 성능 높이기")
st.caption(
    "뇌졸중을 겪은 사람이 적다는 점을 고려해 "
    "가중치를 맞추기 전과 맞춘 뒤의 성능을 비교합니다."
)


# --------------------------------------------------
# 2. 데이터 불러오기
# --------------------------------------------------

try:
    df = load_data()
except Exception as error:
    st.error("데이터를 불러오지 못했습니다. 인터넷 연결이나 데이터 주소를 확인해 주세요.")
    st.code(str(error))
    st.stop()


# --------------------------------------------------
# 3. 입력 속성 선택
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
# 4. 훈련 데이터와 테스트 데이터 분리
# --------------------------------------------------

st.header("2. 훈련 데이터와 테스트 데이터 분리")

# 채점표 페이지와 같은 방식으로 분리한다.
# stratify를 사용해 훈련용과 테스트용의 양성 비율을 비슷하게 유지한다.
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["stroke"]
)

X_train = train_df[selected_features].copy()
X_test = test_df[selected_features].copy()

# 1은 뇌졸중(양성), 0은 뇌졸중 아님(음성)
y_train = train_df["stroke"].astype(int)
y_test = test_df["stroke"].astype(int)

st.write(
    f"전체 데이터: **{len(df):,}명**  |  "
    f"훈련 데이터: **{len(train_df):,}명**  |  "
    f"테스트 데이터: **{len(test_df):,}명**"
)

train_positive = int(y_train.sum())
test_positive = int(y_test.sum())

col1, col2 = st.columns(2)

with col1:
    st.metric("훈련 데이터의 뇌졸중 환자", f"{train_positive:,}명")

with col2:
    st.metric("테스트 데이터의 뇌졸중 환자", f"{test_positive:,}명")


# --------------------------------------------------
# 5. 결측값 처리
# --------------------------------------------------

st.header("3. 데이터 전처리")

if "bmi" in selected_features:
    bmi_median = X_train["bmi"].median()

    X_train.loc[:, "bmi"] = X_train["bmi"].fillna(bmi_median)
    X_test.loc[:, "bmi"] = X_test["bmi"].fillna(bmi_median)

    st.info(
        f"체질량지수의 빈 값은 훈련 데이터의 중앙값 "
        f"**{bmi_median:.2f}**으로 채웠습니다."
    )

if X_train.isna().any().any() or X_test.isna().any().any():
    st.error("선택한 속성에 처리되지 않은 결측값이 있습니다.")
    st.stop()


# --------------------------------------------------
# 6. 표준화
# --------------------------------------------------

# 로지스틱 회귀에 사용할 표준화 데이터
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --------------------------------------------------
# 7. 모델 만들기
# --------------------------------------------------

st.header("4. 가중치를 맞추기 전과 후의 모델 만들기")

# 로지스틱 회귀
logistic_before = LogisticRegression(
    max_iter=2000,
    random_state=42,
    class_weight=None
)

logistic_after = LogisticRegression(
    max_iter=2000,
    random_state=42,
    class_weight="balanced"
)

logistic_before.fit(X_train_scaled, y_train)
logistic_after.fit(X_train_scaled, y_train)


# 의사결정트리
tree_before = DecisionTreeClassifier(
    random_state=42,
    class_weight=None
)

tree_after = DecisionTreeClassifier(
    random_state=42,
    class_weight="balanced"
)

tree_before.fit(X_train, y_train)
tree_after.fit(X_train, y_train)


# --------------------------------------------------
# 8. 평가 함수
# --------------------------------------------------

def evaluate_model(model, X, y, scaled=False):
    if scaled:
        prediction = model.predict(X)
    else:
        prediction = model.predict(X)

    tn, fp, fn, tp = confusion_matrix(
        y,
        prediction,
        labels=[0, 1]
    ).ravel()

    result = {
        "정확도": accuracy_score(y, prediction),
        "정밀도": precision_score(
            y, prediction, zero_division=0
        ),
        "재현율": recall_score(
            y, prediction, zero_division=0
        ),
        "F1 점수": f1_score(
            y, prediction, zero_division=0
        ),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp)
    }

    return result, prediction


# --------------------------------------------------
# 9. 네 지표와 혼동행렬 계산
# --------------------------------------------------

logistic_before_result, logistic_before_pred = evaluate_model(
    logistic_before, X_test_scaled, y_test, scaled=True
)

logistic_after_result, logistic_after_pred = evaluate_model(
    logistic_after, X_test_scaled, y_test, scaled=True
)

tree_before_result, tree_before_pred = evaluate_model(
    tree_before, X_test, y_test
)

tree_after_result, tree_after_pred = evaluate_model(
    tree_after, X_test, y_test
)


# --------------------------------------------------
# 10. 비교 표 만들기
# --------------------------------------------------

def make_comparison_table(before, after):
    rows = [
        "정확도",
        "정밀도",
        "재현율",
        "F1 점수",
        "실제 음성 / 예측 음성 (TN)",
        "실제 음성 / 예측 양성 (FP)",
        "실제 양성 / 예측 음성 (FN)",
        "실제 양성 / 예측 양성 (TP)"
    ]

    before_values = [
        f"{before['정확도'] * 100:.2f}%",
        f"{before['정밀도'] * 100:.2f}%",
        f"{before['재현율'] * 100:.2f}%",
        f"{before['F1 점수'] * 100:.2f}%",
        f"{before['TN']:,}명",
        f"{before['FP']:,}명",
        f"{before['FN']:,}명",
        f"{before['TP']:,}명"
    ]

    after_values = [
        f"{after['정확도'] * 100:.2f}%",
        f"{after['정밀도'] * 100:.2f}%",
        f"{after['재현율'] * 100:.2f}%",
        f"{after['F1 점수'] * 100:.2f}%",
        f"{after['TN']:,}명",
        f"{after['FP']:,}명",
        f"{after['FN']:,}명",
        f"{after['TP']:,}명"
    ]

    return pd.DataFrame({
        "평가 항목": rows,
        "가중치 맞추기 전": before_values,
        "가중치 맞춘 뒤": after_values
    })


def show_error_change(before, after):
    st.write(
        f"찾아낸 환자 수: **{before['TP']:,}명 → {after['TP']:,}명**  |  "
        f"놓친 환자 수: **{before['FN']:,}명 → {after['FN']:,}명**  |  "
        f"헛짚은 사람 수: **{before['FP']:,}명 → {after['FP']:,}명**"
    )


# --------------------------------------------------
# 11. 로지스틱 회귀 결과
# --------------------------------------------------

st.divider()
st.header("5. 로지스틱 회귀(확률로 답하는 모델)")

st.write(
    "가중치를 맞추기 전과 맞춘 뒤의 테스트 데이터 평가 결과입니다."
)

logistic_table = make_comparison_table(
    logistic_before_result,
    logistic_after_result
)

st.dataframe(
    logistic_table,
    hide_index=True,
    use_container_width=True
)

show_error_change(
    logistic_before_result,
    logistic_after_result
)


# --------------------------------------------------
# 12. 의사결정트리 결과
# --------------------------------------------------

st.divider()
st.header("6. 의사결정트리(질문으로 답하는 모델)")

st.write(
    "가중치를 맞추기 전과 맞춘 뒤의 테스트 데이터 평가 결과입니다."
)

tree_table = make_comparison_table(
    tree_before_result,
    tree_after_result
)

st.dataframe(
    tree_table,
    hide_index=True,
    use_container_width=True
)

show_error_change(
    tree_before_result,
    tree_after_result
)


# --------------------------------------------------
# 13. 시각화에 사용할 축 선택
# --------------------------------------------------

st.divider()
st.header("7. 가중치 적용 전후의 예측 경계 비교")

st.write(
    "테스트 데이터의 실제 뇌졸중 여부를 점으로 표시하고, "
    "가중치를 맞추기 전과 맞춘 뒤의 0.5 경계를 비교합니다."
)

axis_features = selected_features.copy()

col_x, col_y = st.columns(2)

with col_x:
    x_feature = st.selectbox(
        "가로축",
        options=axis_features,
        index=0,
        format_func=lambda x: FEATURE_NAMES[x]
    )

with col_y:
    y_options = [
        feature for feature in axis_features
        if feature != x_feature
    ]

    y_feature = st.selectbox(
        "세로축",
        options=y_options,
        index=0,
        format_func=lambda x: FEATURE_NAMES[x]
    )


# --------------------------------------------------
# 14. 테스트 데이터의 실제 결과 표시용 데이터
# --------------------------------------------------

plot_df = X_test.copy()

plot_df["실제 결과"] = y_test.map({
    0: "뇌졸중 아님(음성)",
    1: "뇌졸중(양성)"
})

plot_df["실제 결과"] = plot_df["실제 결과"].astype(str)


# --------------------------------------------------
# 15. 2차원 산점도와 경계선
# --------------------------------------------------

def make_2d_grid(x_name, y_name):
    x_min = float(X_test[x_name].min())
    x_max = float(X_test[x_name].max())
    y_min = float(X_test[y_name].min())
    y_max = float(X_test[y_name].max())

    x_margin = (x_max - x_min) * 0.05
    y_margin = (y_max - y_min) * 0.05

    if x_margin == 0:
        x_margin = 1

    if y_margin == 0:
        y_margin = 1

    x_range = np.linspace(
        x_min - x_margin,
        x_max + x_margin,
        120
    )

    y_range = np.linspace(
        y_min - y_margin,
        y_max + y_margin,
        120
    )

    xx, yy = np.meshgrid(x_range, y_range)

    grid_data = {}

    for feature in selected_features:
        if feature == x_name:
            grid_data[feature] = xx.ravel()

        elif feature == y_name:
            grid_data[feature] = yy.ravel()

        else:
            grid_data[feature] = np.full(
                xx.size,
                X_test[feature].median()
            )

    grid_df = pd.DataFrame(grid_data)[selected_features]

    return x_range, y_range, xx, yy, grid_df


def add_boundary_contour(
    fig,
    x_range,
    y_range,
    probability,
    name,
    line_color,
    dash_style
):
    probability_min = float(np.min(probability))
    probability_max = float(np.max(probability))

    if probability_min <= 0.5 <= probability_max:
        fig.add_trace(
            go.Contour(
                x=x_range,
                y=y_range,
                z=probability,
                contours=dict(
                    start=0.5,
                    end=0.5,
                    size=0.1,
                    coloring="none"
                ),
                line=dict(
                    color=line_color,
                    width=3,
                    dash=dash_style
                ),
                showscale=False,
                hoverinfo="skip",
                name=name
            )
        )

        return True

    return False


def draw_2d_model(
    model_before,
    model_after,
    model_name,
    scaled=False
):
    x_range, y_range, xx, yy, grid_df = make_2d_grid(
        x_feature,
        y_feature
    )

    if scaled:
        grid_input = scaler.transform(grid_df)
    else:
        grid_input = grid_df

    probability_before = model_before.predict_proba(
        grid_input
    )[:, 1].reshape(xx.shape)

    probability_after = model_after.predict_proba(
        grid_input
    )[:, 1].reshape(xx.shape)

    fig = go.Figure()

    # 실제 테스트 데이터
    color_map = {
        "뇌졸중 아님(음성)": "#3182CE",
        "뇌졸중(양성)": "#E53E3E"
    }

    for label in [
        "뇌졸중 아님(음성)",
        "뇌졸중(양성)"
    ]:
        group = plot_df[plot_df["실제 결과"] == label]

        fig.add_trace(
            go.Scatter(
                x=group[x_feature],
                y=group[y_feature],
                mode="markers",
                name=label,
                marker=dict(
                    color=color_map[label],
                    size=7,
                    opacity=0.65
                ),
                hovertemplate=(
                    f"{FEATURE_NAMES[x_feature]}: %{{x}}<br>"
                    f"{FEATURE_NAMES[y_feature]}: %{{y}}"
                    "<extra></extra>"
                )
            )
        )

    before_inside = add_boundary_contour(
        fig,
        x_range,
        y_range,
        probability_before,
        "가중치 맞추기 전 0.5 경계",
        "#16A34A",
        "dash"
    )

    after_inside = add_boundary_contour(
        fig,
        x_range,
        y_range,
        probability_after,
        "가중치 맞춘 뒤 0.5 경계",
        "#9333EA",
        "solid"
    )

    fig.update_layout(
        title=model_name,
        xaxis_title=FEATURE_NAMES[x_feature],
        yaxis_title=FEATURE_NAMES[y_feature],
        height=600,
        legend_title="실제 결과",
        hovermode="closest"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    if not before_inside:
        st.warning(
            f"{model_name}: 가중치 맞추기 전 0.5 경계가 그림 밖에 있습니다."
        )

    if not after_inside:
        st.warning(
            f"{model_name}: 가중치 맞춘 뒤 0.5 경계가 그림 밖에 있습니다."
        )


# --------------------------------------------------
# 16. 3차원 산점도와 경계면
# --------------------------------------------------

def make_3d_grid(x_name, y_name, z_name):
    ranges = {}

    for feature in [x_name, y_name, z_name]:
        minimum = float(X_test[feature].min())
        maximum = float(X_test[feature].max())

        margin = (maximum - minimum) * 0.05

        if margin == 0:
            margin = 1

        ranges[feature] = np.linspace(
            minimum - margin,
            maximum + margin,
            22
        )

    xx, yy, zz = np.meshgrid(
        ranges[x_name],
        ranges[y_name],
        ranges[z_name],
        indexing="ij"
    )

    grid_data = {}

    for feature in selected_features:
        if feature == x_name:
            grid_data[feature] = xx.ravel()

        elif feature == y_name:
            grid_data[feature] = yy.ravel()

        elif feature == z_name:
            grid_data[feature] = zz.ravel()

        else:
            grid_data[feature] = np.full(
                xx.size,
                X_test[feature].median()
            )

    grid_df = pd.DataFrame(grid_data)[selected_features]

    return ranges, xx, yy, zz, grid_df


def add_3d_boundary(
    fig,
    xx,
    yy,
    zz,
    probability,
    name,
    color,
    opacity
):
    minimum = float(np.min(probability))
    maximum = float(np.max(probability))

    if minimum <= 0.5 <= maximum:
        fig.add_trace(
            go.Isosurface(
                x=xx.ravel(),
                y=yy.ravel(),
                z=zz.ravel(),
                value=probability.ravel(),
                isomin=0.499,
                isomax=0.501,
                surface_count=1,
                colorscale=[
                    [0, color],
                    [1, color]
                ],
                opacity=opacity,
                showscale=False,
                caps=dict(
                    x_show=False,
                    y_show=False,
                    z_show=False
                ),
                name=name,
                hoverinfo="skip"
            )
        )

        return True

    return False


def draw_3d_model(
    model_before,
    model_after,
    model_name,
    x_name,
    y_name,
    z_name,
    scaled=False
):
    ranges, xx, yy, zz, grid_df = make_3d_grid(
        x_name,
        y_name,
        z_name
    )

    if scaled:
        grid_input = scaler.transform(grid_df)
    else:
        grid_input = grid_df

    probability_before = model_before.predict_proba(
        grid_input
    )[:, 1]

    probability_after = model_after.predict_proba(
        grid_input
    )[:, 1]

    fig = go.Figure()

    color_map = {
        "뇌졸중 아님(음성)": "#3182CE",
        "뇌졸중(양성)": "#E53E3E"
    }

    for label in [
        "뇌졸중 아님(음성)",
        "뇌졸중(양성)"
    ]:
        group = plot_df[plot_df["실제 결과"] == label]

        fig.add_trace(
            go.Scatter3d(
                x=group[x_name],
                y=group[y_name],
                z=group[z_name],
                mode="markers",
                name=label,
                marker=dict(
                    color=color_map[label],
                    size=3,
                    opacity=0.65
                )
            )
        )

    before_inside = add_3d_boundary(
        fig,
        xx,
        yy,
        zz,
        probability_before,
        "가중치 맞추기 전 0.5 평면",
        "#16A34A",
        0.35
    )

    after_inside = add_3d_boundary(
        fig,
        xx,
        yy,
        zz,
        probability_after,
        "가중치 맞춘 뒤 0.5 평면",
        "#9333EA",
        0.45
    )

    fig.update_layout(
        title=model_name,
        height=700,
        scene=dict(
            xaxis_title=FEATURE_NAMES[x_name],
            yaxis_title=FEATURE_NAMES[y_name],
            zaxis_title=FEATURE_NAMES[z_name]
        ),
        legend_title="실제 결과"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    if not before_inside:
        st.warning(
            f"{model_name}: 가중치 맞추기 전 0.5 평면이 그림 밖에 있습니다."
        )

    if not after_inside:
        st.warning(
            f"{model_name}: 가중치 맞춘 뒤 0.5 평면이 그림 밖에 있습니다."
        )


# --------------------------------------------------
# 17. 선택한 속성 수에 따라 그래프 표시
# --------------------------------------------------

if len(selected_features) == 2:

    st.subheader("2차원 산점도")

    st.write(
        "두 축 이외의 선택 속성은 테스트 데이터의 중앙값으로 고정해 "
        "예측 경계를 계산합니다."
    )

    fixed_features = [
        feature for feature in selected_features
        if feature not in [x_feature, y_feature]
    ]

    if fixed_features:
        st.write(
            "고정한 속성: "
            + ", ".join(
                f"{FEATURE_NAMES[f]} = {X_test[f].median():.2f}"
                for f in fixed_features
            )
        )

    col1, col2 = st.columns(2)

    with col1:
        draw_2d_model(
            logistic_before,
            logistic_after,
            "로지스틱 회귀(확률로 답하는 모델)",
            scaled=True
        )

    with col2:
        draw_2d_model(
            tree_before,
            tree_after,
            "의사결정트리(질문으로 답하는 모델)",
            scaled=False
        )

else:

    st.subheader("3차원 산점도")

    st.write(
        "세 개의 속성을 축으로 사용합니다. "
        "나머지 속성은 테스트 데이터의 중앙값으로 고정합니다. "
        "그래프를 마우스로 움직여 여러 방향에서 확인할 수 있습니다."
    )

    axis1, axis2, axis3 = st.columns(3)

    with axis1:
        x3_feature = st.selectbox(
            "첫 번째 축",
            options=selected_features,
            index=0,
            format_func=lambda x: FEATURE_NAMES[x]
        )

    with axis2:
        y3_options = [
            f for f in selected_features
            if f != x3_feature
        ]

        y3_feature = st.selectbox(
            "두 번째 축",
            options=y3_options,
            index=0,
            format_func=lambda x: FEATURE_NAMES[x]
        )

    with axis3:
        z3_options = [
            f for f in selected_features
            if f not in [x3_feature, y3_feature]
        ]

        z3_feature = st.selectbox(
            "세 번째 축",
            options=z3_options,
            index=0,
            format_func=lambda x: FEATURE_NAMES[x]
        )

    fixed_features = [
        feature for feature in selected_features
        if feature not in [
            x3_feature,
            y3_feature,
            z3_feature
        ]
    ]

    if fixed_features:
        st.write(
            "고정한 속성: "
            + ", ".join(
                f"{FEATURE_NAMES[f]} = {X_test[f].median():.2f}"
                for f in fixed_features
            )
        )

    col1, col2 = st.columns(2)

    with col1:
        draw_3d_model(
            logistic_before,
            logistic_after,
            "로지스틱 회귀(확률로 답하는 모델)",
            x3_feature,
            y3_feature,
            z3_feature,
            scaled=True
        )

    with col2:
        draw_3d_model(
            tree_before,
            tree_after,
            "의사결정트리(질문으로 답하는 모델)",
            x3_feature,
            y3_feature,
            z3_feature,
            scaled=False
        )


# --------------------------------------------------
# 18. 마무리 안내
# --------------------------------------------------

st.divider()

st.caption(
    "주의: 이 페이지는 머신러닝의 분류 성능과 가중치 조정 원리를 이해하기 위한 교육용 실습입니다. "
    "실제 환자의 뇌졸중 여부를 진단하는 용도로 사용할 수 없습니다."
)
