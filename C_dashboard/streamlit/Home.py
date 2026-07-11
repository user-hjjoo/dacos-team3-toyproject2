from pathlib import Path

import pandas as pd
import streamlit as st

# ------------------------------------------------------------
# 경로 설정 (절대경로 대신 pathlib 기반 상대경로)
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

st.set_page_config(
    page_title="리뷰 신뢰도 분석 대시보드",
    page_icon="⭐",
    layout="wide",
)


@st.cache_data
def load_products() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "A_amazon_final_processed.csv")


df = load_products()

# ------------------------------------------------------------
# 헤더 & 핵심 질문
# ------------------------------------------------------------
st.title("⭐ 리뷰 신뢰도 분석 대시보드")
st.caption("리뷰 수와 평점, 무엇을 더 믿어야 할까?")

st.info(
    "**프로젝트 핵심 질문**\n\n"
    "리뷰 수가 적지만 평점이 높은 상품 vs 리뷰 수가 많지만 평점이 조금 낮은 상품, "
    "**무엇을 선택할 것인가?**"
)

st.markdown("---")

# ------------------------------------------------------------
# 실제 데이터 기준 요약 카드
# ------------------------------------------------------------
st.subheader("📊 전체 데이터 요약")

col1, col2, col3, col4 = st.columns(4)
col1.metric("전체 상품 수", f"{df['product_id'].nunique():,} 개")
col2.metric("카테고리 수", f"{df['main_category'].nunique()} 개")
col3.metric("평균 평점", f"{df['rating'].mean():.2f} ⭐")
col4.metric("평균 리뷰 수", f"{df['rating_count'].mean():,.0f} 개")

st.markdown("---")

# ------------------------------------------------------------
# 분석 흐름 소개
# ------------------------------------------------------------
st.subheader("🧭 분석 흐름")

flow_cols = st.columns(5)
flow_steps = [
    ("1️⃣", "EDA", "리뷰 수 · 평점 관계 탐색"),
    ("2️⃣", "베이지안 보정 평점", "카테고리별 m값 적용"),
    ("3️⃣", "회귀 모델 비교", "Random Forest 등 3개 모델"),
    ("4️⃣", "SHAP 해석", "예측에 영향을 준 요인 분석"),
    ("5️⃣", "구매 가이드", "실제 입력 → 보정 평점 계산"),
]
for col, (num, title, desc) in zip(flow_cols, flow_steps):
    with col:
        st.markdown(f"### {num}")
        st.markdown(f"**{title}**")
        st.caption(desc)

st.markdown("---")

# ------------------------------------------------------------
# 핵심 메시지
# ------------------------------------------------------------
st.subheader("💡 핵심 메시지")
st.success(
    "평점은 같거나 높더라도 리뷰 수가 적으면 신뢰하기 어려울 수 있습니다. "
    "따라서 **평점과 리뷰 수를 함께 고려한 보정 기준**이 필요합니다."
)

st.markdown("---")

# ------------------------------------------------------------
# 대표 상품 미리보기 (전체 표 대신 일부만)
# ------------------------------------------------------------
st.subheader("🛍️ 대표 상품 미리보기")

preview_cols = ["product_name", "main_category", "rating", "rating_count", "bayesian_rating"]
preview_df = (
    df[preview_cols]
    .rename(columns={
        "product_name": "상품명",
        "main_category": "카테고리",
        "rating": "평점",
        "rating_count": "리뷰 수",
        "bayesian_rating": "보정 평점",
    })
    .sample(8, random_state=42)
    .reset_index(drop=True)
)
st.dataframe(preview_df, use_container_width=True)

st.caption("※ 위 표는 전체 1,350개 상품 중 일부(8개)만 보여주는 미리보기입니다. 전체 데이터는 [EDA 탐색] 페이지에서 확인할 수 있습니다.")
