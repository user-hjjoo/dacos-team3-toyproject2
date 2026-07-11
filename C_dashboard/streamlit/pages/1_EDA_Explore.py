from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ------------------------------------------------------------
# 경로 설정
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

st.set_page_config(page_title="EDA 탐색", page_icon="🔎", layout="wide")


@st.cache_data
def load_products() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "A_amazon_final_processed.csv")


@st.cache_data
def load_m_value() -> pd.DataFrame:
    m_df = pd.read_csv(DATA_DIR / "A_m_value_q1.csv")
    m_df.columns = [c.strip() for c in m_df.columns]
    return m_df


df = load_products()
m_df = load_m_value()

st.title("🔎 EDA 탐색 페이지")
st.caption("왜 단순 평점이 아니라 리뷰 수를 함께 고려해야 할까요? 데이터로 직접 확인합니다.")

# ============================================================
# 2-1. 리뷰 수 vs 평점 산점도 (Funnel Plot)
# ============================================================
st.markdown("## 2-1. 리뷰 수 vs 평점 산점도")

categories = ["전체"] + sorted(df["main_category"].unique().tolist())
selected_category = st.selectbox("카테고리 필터", categories, key="scatter_category")

filtered = df if selected_category == "전체" else df[df["main_category"] == selected_category]

fig1 = px.scatter(
    filtered,
    x="rating_count",
    y="rating",
    color="main_category",
    log_x=True,
    hover_data={
        "product_name": True,
        "main_category": True,
        "rating": True,
        "rating_count": True,
        "bayesian_rating": True,
    },
    labels={"rating_count": "리뷰 수 (log scale)", "rating": "평점", "main_category": "카테고리"},
    title="리뷰 수(log) vs 평점",
    opacity=0.6,
)
fig1.update_layout(height=520)
st.plotly_chart(fig1, use_container_width=True)

st.info(
    "💡 **인사이트**: 리뷰 수가 적은 구간에서는 평점이 2.0점부터 5.0점까지 넓게 퍼져 있어, "
    "리뷰 수가 적을수록 평점 변동성이 크다는 점을 확인할 수 있습니다. "
    "따라서 단순 평점만으로 상품을 비교하기보다, 리뷰 수를 함께 고려한 보정 기준이 필요합니다."
)

st.markdown("---")

# ============================================================
# 2-2. 리뷰 수 상위 20% vs 하위 20% 평점 비교
# ============================================================
st.markdown("## 2-2. 리뷰 수 상위 20% vs 하위 20% 평점 비교")

q20 = df["rating_count"].quantile(0.2)
q80 = df["rating_count"].quantile(0.8)

bottom_group = df[df["rating_count"] <= q20].copy()
top_group = df[df["rating_count"] >= q80].copy()

bottom_group["그룹"] = "하위 20% (리뷰 수 적음)"
top_group["그룹"] = "상위 20% (리뷰 수 많음)"
compare_df = pd.concat([bottom_group, top_group], ignore_index=True)

bottom_std = bottom_group["rating"].std()
top_std = top_group["rating"].std()

col_std1, col_std2 = st.columns(2)
col_std1.metric("하위 20% 그룹 평점 표준편차", f"{bottom_std:.3f}", help=f"상품 수 {len(bottom_group)}개")
col_std2.metric("상위 20% 그룹 평점 표준편차", f"{top_std:.3f}", help=f"상품 수 {len(top_group)}개")

fig2 = px.box(
    compare_df,
    x="그룹",
    y="rating",
    color="그룹",
    points="all",
    labels={"rating": "평점"},
    title="리뷰 수 상·하위 20% 그룹의 평점 분포",
)
fig2.update_layout(height=480, showlegend=False)
st.plotly_chart(fig2, use_container_width=True)

st.info(
    f"💡 **인사이트**: 리뷰 수 하위 20% 그룹의 평점 표준편차는 **{bottom_std:.3f}**, "
    f"상위 20% 그룹은 **{top_std:.3f}**로 나타났습니다. "
    "리뷰 수가 적은 그룹일수록 평점 분산이 약 2배 이상 커, "
    "리뷰 수는 평점 신뢰도를 판단하는 중요한 기준이 됩니다."
)

st.markdown("---")

# ============================================================
# 2-3. 카테고리별 m값 비교
# ============================================================
st.markdown("## 2-3. 카테고리별 리뷰 수(m값) 비교")

m_sorted = m_df.sort_values("m_value", ascending=True)

fig3 = go.Figure(
    go.Bar(
        x=m_sorted["m_value"],
        y=m_sorted["main_category"],
        orientation="h",
        marker_color="#4C78A8",
    )
)
fig3.update_layout(
    title="카테고리별 m값 (Q1 기준)",
    xaxis_title="m값",
    yaxis_title="카테고리",
    height=500,
)
st.plotly_chart(fig3, use_container_width=True)

st.info(
    "💡 **인사이트**: 카테고리별 m값은 HomeTheater,TV&Video 248.2개부터 "
    "ExternalDevices&DataStorage 19,747.5개까지 큰 차이를 보입니다. "
    "카테고리마다 일반적인 리뷰 수 수준이 크게 다르므로, 모든 상품에 동일한 기준을 적용하지 않고 "
    "카테고리별 m값을 적용하였습니다."
)

st.markdown("#### 카테고리 상세 정보")
selected_detail = st.selectbox("카테고리 선택", sorted(df["main_category"].unique().tolist()), key="detail_category")

cat_products = df[df["main_category"] == selected_detail]
cat_m_value = m_df.loc[m_df["main_category"] == selected_detail, "m_value"].values[0]

d1, d2, d3, d4 = st.columns(4)
d1.metric("상품 수", f"{len(cat_products):,} 개")
d2.metric("평균 리뷰 수", f"{cat_products['rating_count'].mean():,.0f} 개")
d3.metric("Q1 기반 m값", f"{cat_m_value:,.1f}")
d4.metric("평균 평점", f"{cat_products['rating'].mean():.2f} ⭐")

st.markdown("---")

# ============================================================
# EDA 페이지 최종 결론
# ============================================================
st.subheader("✅ EDA 페이지 최종 결론")
st.success(
    "- 평점만으로 상품을 비교하기 어렵습니다.\n"
    "- 리뷰 수와 카테고리 특성을 함께 고려한 **베이지안 보정 평점**이 필요합니다."
)
