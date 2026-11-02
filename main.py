import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# --------------------------------------------------
# 기본 설정
# --------------------------------------------------
st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide"
)

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
BASE_DIR = Path(__file__).resolve().parent
SAVE_DIR = BASE_DIR / "내_기록"
SAVE_FILE = SAVE_DIR / "stroke_notes.json"

st.title("🧠 뇌졸중 예측 실습실")
st.caption("데이터를 살펴보고, 각 열의 의미를 직접 정리해 보는 공간입니다.")

# --------------------------------------------------
# 기록 파일 불러오기
# --------------------------------------------------
def load_notes():
    if SAVE_FILE.exists():
        try:
            with open(SAVE_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
            return saved
        except (json.JSONDecodeError, OSError):
            st.warning("저장된 기록 파일을 읽지 못해 빈 기록으로 시작합니다.")
    return {}

notes = load_notes()

# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")

try:
    df = load_data()
except Exception as e:
    st.error("데이터를 불러오지 못했습니다. 인터넷 연결이나 데이터 주소를 확인해 주세요.")
    st.code(str(e))
    st.stop()

# 열 설명 표 구성
saved_meanings = notes.get("meanings", {})
info = pd.DataFrame({
    "열 이름": df.columns,
    "우리말 뜻": [saved_meanings.get(col, "") for col in df.columns],
    "값의 종류": [
        f"숫자형 ({df[col].nunique(dropna=True):,}종)" if
        pd.api.types.is_numeric_dtype(df[col])
        else f"문자형 ({df[col].nunique(dropna=True):,}종)"
        for col in df.columns
    ],
    "빈 값 개수": [int(df[col].isna().sum()) for col in df.columns]
})

# --------------------------------------------------
# 첫 화면: 데이터 소개
# --------------------------------------------------
st.header("1. 데이터 한눈에 보기")
st.write(
    "이 데이터는 여러 사람의 기본 정보와 건강 관련 특성, 그리고 뇌졸중 발생 여부를 담고 있습니다. "
    "각 행은 한 사람의 자료이며, `stroke` 열은 뇌졸중 발생 여부를 나타내는 값으로 분석의 목표 열로 활용할 수 있습니다."
)

total_people = len(df)
column_count = len(df.columns)
stroke_count = int((df["stroke"] == 1).sum())
stroke_ratio = stroke_count / total_people * 100 if total_people else 0

metric_cols = st.columns(4)
metric_cols[0].metric("전체 사람 수", f"{total_people:,}명")
metric_cols[1].metric("열 개수", f"{column_count:,}개")
metric_cols[2].metric("stroke가 1인 사람 수", f"{stroke_count:,}명")
metric_cols[3].metric("stroke = 1 비율", f"{stroke_ratio:.2f}%")

st.divider()
st.header("2. 열 이름과 데이터 정보")
st.write("교재를 참고해 **우리말 뜻** 칸을 직접 채워 보세요. 값의 종류와 빈 값 개수는 데이터에서 자동으로 계산합니다.")

with st.form("notes_form"):
    edited_info = st.data_editor(
        info,
        hide_index=True,
        use_container_width=True,
        num_rows="fixed",
        disabled=["열 이름", "값의 종류", "빈 값 개수"],
        column_config={
            "열 이름": st.column_config.TextColumn("열 이름", width="medium"),
            "우리말 뜻": st.column_config.TextColumn("우리말 뜻", width="large"),
            "값의 종류": st.column_config.TextColumn("값의 종류", width="medium"),
            "빈 값 개수": st.column_config.NumberColumn("빈 값 개수", format="%d")
        }
    )

    st.subheader("데이터 출처 기록")
    source_text = st.text_area(
        "교재에 적힌 데이터 출처를 입력하세요.",
        value=notes.get("source", ""),
        placeholder="예: 교재에 제시된 데이터 출처",
        height=100
    )
    save_clicked = st.form_submit_button("💾 우리말 뜻과 출처 저장", type="primary")

if save_clicked:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    updated_notes = {
        "meanings": {
            str(row["열 이름"]): str(row["우리말 뜻"] or "")
            for _, row in edited_info.iterrows()
        },
        "source": source_text
    }
    with open(SAVE_FILE, "w", encoding="utf-8") as f:
        json.dump(updated_notes, f, ensure_ascii=False, indent=2)
    st.success(f"기록을 저장했습니다. 저장 위치: {SAVE_FILE.relative_to(BASE_DIR)}")
    st.rerun()

st.divider()
st.header("3. 데이터 처음 다섯 줄")
st.write("원본 데이터의 앞부분을 수정하지 않고 그대로 표시합니다.")
st.dataframe(df.head(5), use_container_width=True)

st.divider()
st.header("4. 데이터 출처")
if notes.get("source", "").strip():
    st.write(notes["source"])
else:
    st.info("아직 출처가 입력되지 않았습니다. 위의 출처 기록 칸에 교재의 내용을 입력하고 저장해 주세요.")

st.caption("참고: 이 앱은 데이터 탐색 실습용입니다. 데이터의 통계적 관계가 개인의 실제 뇌졸중 발생 여부를 진단하거나 확정하지는 않습니다.")
