
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix


# --------------------------------------------------
# 기본 설정
# --------------------------------------------------

st.set_page_config(
    page_title="채점표 | 뇌졸중 예측 실습실",
    page_icon="📝",
    layout="wide"
)

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"


# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


st.title("📝 모델 채점표")
st.caption("테스트 데이터로 두 분류 모델의 예측 결과를 비교합니다.")

try:
    df = load_data()
except Exception as error:
    st.error("데이터를 불러오지 못했습니다.")
    st.code(str(error))
    st.stop()


# --------------------------------------------------
# 데이터 전처리 및 분할
# --------------------------------------------------

# 번호와 정답 열 분리
X = df.drop(columns=["id", "stroke"])
y = df["stroke"]

# 숫자형 열과 문자형 열 구분
numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

# 숫자형 데이터 전처리
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# 문자형 데이터 전처리
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# 전처리 과정 통합
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# 훈련용 데이터와 테스트용 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 두 모델 학습
# --------------------------------------------------

logistic_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=2000))
])

tree_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(random_state=42))
])

logistic_model.fit(X_train, y_train)
tree_model.fit(X_train, y_train)

# 테스트 데이터 예측
logistic_pred = logistic_model.predict(X_test)
tree_pred = tree_model.predict(X_test)


# --------------------------------------------------
# 훈련용에서 많은 쪽으로만 답하는 기준 모델
# --------------------------------------------------

majority_class = int(y_train.mode().iloc[0])

baseline_pred = [majority_class] * len(y_test)


# --------------------------------------------------
# 혼동행렬 및 평가지표 계산 함수
# --------------------------------------------------

def calculate_metrics(y_true, y_pred):

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    total = tn + fp + fn + tp

    accuracy = (tp + tn) / total if total else 0

    # 실제 뇌졸중인 사람 수를 분모로 사용
    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    # 뇌졸중이라고 예측한 사람 수를 분모로 사용
    if tp + fp == 0:
        precision = None
        f1 = None
    else:
        precision = tp / (tp + fp)

        # F1 = 2 × 정밀도 × 재현율 / (정밀도 + 재현율)
        denominator = precision + recall

        if denominator == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / denominator

    return {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "정확도": accuracy,
        "재현율": recall,
        "정밀도": precision,
        "F1": f1
    }


# --------------------------------------------------
# 평가 결과 계산
# --------------------------------------------------

results = {
    "로지스틱 회귀": calculate_metrics(y_test, logistic_pred),
    "의사결정트리": calculate_metrics(y_test, tree_pred),
    "훈련용에서 많은 쪽으로만 답하는 모델": calculate_metrics(
        y_test, baseline_pred
    )
}


# --------------------------------------------------
# 테스트 데이터 정보
# --------------------------------------------------

test_people = len(y_test)
test_stroke = int(y_test.sum())

st.subheader("1. 테스트 데이터")

st.write(
    f"테스트용 사람 수: **{test_people:,}명** "
    f"| 그중 실제 뇌졸중인 사람 수: **{test_stroke:,}명**"
)

st.caption(
    "테스트 데이터는 모델을 학습할 때 사용하지 않고, "
    "학습이 끝난 모델의 예측 능력을 평가하는 데 사용합니다."
)

st.divider()


# --------------------------------------------------
# 혼동행렬 표시 함수
# --------------------------------------------------

def show_confusion_matrix(title, result):

    tn = result["TN"]
    fp = result["FP"]
    fn = result["FN"]
    tp = result["TP"]

    matrix = pd.DataFrame(
        [
            [
                f"TN: {tn}명",
                f"FP: {fp}명",
                f"합계: {tn + fp}명"
            ],
            [
                f"FN: {fn}명",
                f"TP: {tp}명",
                f"합계: {fn + tp}명"
            ],
            [
                f"합계: {tn + fn}명",
                f"합계: {fp + tp}명",
                f"전체: {tn + fp + fn + tp}명"
            ]
        ],
        index=["실제 뇌졸중 없음", "실제 뇌졸중 있음", "열 합계"],
        columns=["예측 뇌졸중 없음", "예측 뇌졸중 있음", "행 합계"]
    )

    st.subheader(title)
    st.dataframe(
        matrix,
        use_container_width=True
    )


# --------------------------------------------------
# 두 모델의 혼동행렬
# --------------------------------------------------

st.header("2. 두 모델의 혼동행렬")

st.write(
    "행은 실제 결과, 열은 모델의 예측 결과입니다."
)

col1, col2 = st.columns(2)

with col1:
    show_confusion_matrix(
        "로지스틱 회귀 (확률로 답하는 모델)",
        results["로지스틱 회귀"]
    )

with col2:
    show_confusion_matrix(
        "의사결정트리 (질문으로 답하는 모델)",
        results["의사결정트리"]
    )

st.caption(
    "TP: 뇌졸중인 사람을 뇌졸중이라고 예측 | "
    "TN: 아닌 사람을 아니라고 예측 | "
    "FP: 아닌 사람을 뇌졸중이라고 예측 | "
    "FN: 뇌졸중인 사람을 아니라고 예측"
)

st.divider()


# --------------------------------------------------
# 평가지표 비교
# --------------------------------------------------

st.header("3. 모델별 평가지표 비교")

def format_score(value):

    if value is None:
        return "계산할 수 없음"

    return f"{value:.4f}"


metric_table = pd.DataFrame([
    {
        "모델": "로지스틱 회귀 (확률로 답하는 모델)",
        "정확도": format_score(results["로지스틱 회귀"]["정확도"]),
        "재현율": format_score(results["로지스틱 회귀"]["재현율"]),
        "정밀도": format_score(results["로지스틱 회귀"]["정밀도"]),
        "F1": format_score(results["로지스틱 회귀"]["F1"])
    },
    {
        "모델": "의사결정트리 (질문으로 답하는 모델)",
        "정확도": format_score(results["의사결정트리"]["정확도"]),
        "재현율": format_score(results["의사결정트리"]["재현율"]),
        "정밀도": format_score(results["의사결정트리"]["정밀도"]),
        "F1": format_score(results["의사결정트리"]["F1"])
    },
    {
        "모델": "훈련용에서 많은 쪽으로만 답하는 모델",
        "정확도": format_score(results["훈련용에서 많은 쪽으로만 답하는 모델"]["정확도"]),
        "재현율": format_score(results["훈련용에서 많은 쪽으로만 답하는 모델"]["재현율"]),
        "정밀도": format_score(results["훈련용에서 많은 쪽으로만 답하는 모델"]["정밀도"]),
        "F1": format_score(results["훈련용에서 많은 쪽으로만 답하는 모델"]["F1"])
    }
])

st.dataframe(
    metric_table,
    hide_index=True,
    use_container_width=True
)

st.write(
    """
    - **정확도**: 전체 사람 중 예측을 맞힌 비율
    - **재현율**: 실제 뇌졸중인 사람 중 찾아낸 비율
    - **정밀도**: 뇌졸중이라고 예측한 사람 중 실제 뇌졸중인 비율
    - **F1**: 정밀도와 재현율의 조화평균
    """
)

st.divider()


# --------------------------------------------------
# 놓친 환자와 헛짚은 사람 찾기
# --------------------------------------------------

st.header("4. 놓친 환자와 헛짚은 사람")

st.write(
    "모델이 잘못 예측한 사람들을 확인합니다. "
    "나이가 많은 순서로 표시하며 각 목록은 최대 20명까지 보여 줍니다."
)

model_options = {
    "로지스틱 회귀 (확률로 답하는 모델)": logistic_pred,
    "의사결정트리 (질문으로 답하는 모델)": tree_pred
}


# 테스트 데이터의 원래 정보 보관
test_info = df.loc[X_test.index].copy()

# 실제 정답과 예측 결과
test_info["실제"] = y_test
test_info["로지스틱 회귀"] = logistic_pred
test_info["의사결정트리"] = tree_pred

# 두 목록을 나란히 표시
left_col, right_col = st.columns(2)


def make_error_list(data, model_key, error_type):

    if error_type == "FN":
        selected = data[
            (data["stroke"] == 1) &
            (data[model_key] == 0)
        ].copy()

    else:
        selected = data[
            (data["stroke"] == 0) &
            (data[model_key] == 1)
        ].copy()

    # 나이가 많은 순서로 정렬
    selected = selected.sort_values(
        by="age",
        ascending=False
    )

    # 화면에 보여 줄 열
    display = selected[
        [
            "id",
            "age",
            "avg_glucose_level",
            "hypertension",
            "heart_disease"
        ]
    ].copy()

    display = display.rename(columns={
        "id": "번호",
        "age": "나이",
        "avg_glucose_level": "평균 혈당",
        "hypertension": "고혈압 여부",
        "heart_disease": "심장병 여부"
    })

    # 숫자형 여부를 보기 쉽게 변경
    display["고혈압 여부"] = display["고혈압 여부"].map({
        0: "없음",
        1: "있음"
    })

    display["심장병 여부"] = display["심장병 여부"].map({
        0: "없음",
        1: "있음"
    })

    return display, len(selected)


with left_col:

    st.subheader("① 놓친 환자 (FN)")

    selected_fn_model = st.selectbox(
        "FN 목록을 확인할 모델",
        list(model_options.keys()),
        key="fn_model"
    )

    fn_key = (
        "로지스틱 회귀"
        if selected_fn_model.startswith("로지스틱")
        else "의사결정트리"
    )

    fn_data, fn_count = make_error_list(
        test_info,
        fn_key,
        "FN"
    )

    st.write(f"전체 놓친 환자: **{fn_count:,}명**")

    st.dataframe(
        fn_data.head(20),
        hide_index=True,
        use_container_width=True
    )

with right_col:

    st.subheader("② 헛짚은 사람 (FP)")

    selected_fp_model = st.selectbox(
        "FP 목록을 확인할 모델",
        list(model_options.keys()),
        key="fp_model"
    )

    fp_key = (
        "로지스틱 회귀"
        if selected_fp_model.startswith("로지스틱")
        else "의사결정트리"
    )

    fp_data, fp_count = make_error_list(
        test_info,
        fp_key,
        "FP"
    )

    st.write(f"전체 헛짚은 사람: **{fp_count:,}명**")

    st.dataframe(
        fp_data.head(20),
        hide_index=True,
        use_container_width=True
    )


# --------------------------------------------------
# 혼동행렬과 목록의 사람 수 일치 확인
# --------------------------------------------------

st.divider()

st.subheader("목록과 혼동행렬 비교")

st.write(
    f"""
    - 로지스틱 회귀: 놓친 환자 {results["로지스틱 회귀"]["FN"]}명(FN),
      헛짚은 사람 {results["로지스틱 회귀"]["FP"]}명(FP)
    - 의사결정트리: 놓친 환자 {results["의사결정트리"]["FN"]}명(FN),
      헛짚은 사람 {results["의사결정트리"]["FP"]}명(FP)
    """
)

st.caption(
    "목록의 전체 인원수는 혼동행렬의 FN, FP 칸에 표시된 사람 수와 같습니다. "
    "화면에는 각 목록의 앞 20명만 표시됩니다."
)

st.divider()

st.caption(
    "주의: 이 페이지의 결과는 학습용 데이터에 기반한 분류 실습 결과이며, "
    "실제 환자의 뇌졸중 여부를 진단하는 용도로 사용할 수 없습니다."
)
