from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ------------------------------------------------------------
# 경로 설정
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

st.set_page_config(page_title="구매 가이드", page_icon="🧮", layout="wide")


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

CATEGORY_LIST = sorted(df["main_category"].unique().tolist())
M_MAP = dict(zip(m_df["main_category"], m_df["m_value"]))
CAT_MEAN_RATING = df.groupby("main_category")["rating"].mean().to_dict()


def calc_bayesian_rating(rating: float, review_count: float, category: str) -> tuple[float, float, float]:
    """카테고리별 m값과 평균 평점(C)을 이용한 베이지안 보정 평점 계산."""
    m = M_MAP[category]
    c = CAT_MEAN_RATING[category]
    v = review_count
    corrected = (v / (v + m)) * rating + (m / (v + m)) * c
    return corrected, m, c


def judge_message(decline: float) -> tuple[str, str]:
    """보정 하락폭에 따른 판단 메시지 반환 (레벨, 메시지)."""
    if decline < 0.1:
        return "success", "리뷰가 충분해 현재 평점을 비교적 신뢰할 수 있습니다."
    elif decline < 0.3:
        return "warning", "평점은 양호하지만 리뷰 수를 함께 확인하세요."
    else:
        return "error", "높은 평점에 비해 리뷰 수가 부족해 신중한 검토가 필요합니다."


st.title("🧮 구매 가이드 페이지")
st.caption("카테고리 · 평점 · 리뷰 수를 입력하면, 리뷰 수를 고려한 '보정 평점'을 계산해드립니다.")

st.markdown(
    """
    ### 계산 방식 (베이지안 평균 보정)
    > 보정 평점 = (v / (v + m)) × R + (m / (v + m)) × C
    > - v: 입력한 리뷰 수, R: 입력한 평점
    > - m: **카테고리별** 최소 신뢰 리뷰 수 기준값 (Q1 기반)
    > - C: **카테고리별** 평균 평점
    """
)

st.markdown("---")

# ============================================================
# 4-1. 사용자 입력 (단일 상품)
# ============================================================
st.markdown("## 4-1. 상품 정보 입력")

col_a, col_b, col_c = st.columns(3)
with col_a:
    category = st.selectbox("카테고리", CATEGORY_LIST, key="single_category")
with col_b:
    rating = st.number_input("원래 평점 (1.0~5.0)", min_value=1.0, max_value=5.0, value=4.9, step=0.1, key="single_rating")
with col_c:
    review_count = st.number_input("리뷰 수", min_value=0, value=12, step=1, key="single_review_count")

with st.expander("➕ 선택 입력 (참고용 · 보정 평점 계산에는 반영되지 않습니다)"):
    e1, e2, e3 = st.columns(3)
    with e1:
        product_name = st.text_input("상품명 (선택)", key="single_name")
    with e2:
        actual_price = st.number_input("실제 가격 (선택)", min_value=0.0, value=0.0, step=100.0, key="single_price")
    with e3:
        discount_rate = st.number_input("할인율 % (선택)", min_value=0, max_value=100, value=0, key="single_discount")

st.markdown("---")

# ============================================================
# 4-2. 계산 결과
# ============================================================
st.markdown("## 4-2. 계산 결과")

# 선택 입력값은 계산식에는 쓰이지 않지만, 참고 정보로만 별도 표시
if product_name or actual_price > 0 or discount_rate > 0:
    info_bits = []
    if product_name:
        info_bits.append(f"**{product_name}**")
    if actual_price > 0:
        info_bits.append(f"실제 가격 {actual_price:,.0f}원")
    if discount_rate > 0:
        info_bits.append(f"할인율 {discount_rate}%")
    st.caption("📦 참고 정보 (보정 평점 계산에는 반영되지 않음): " + " · ".join(info_bits))

corrected, m_val, c_val = calc_bayesian_rating(rating, review_count, category)
decline = rating - corrected

r1, r2, r3, r4 = st.columns(4)
r1.metric("입력 평점", f"{rating:.2f} ⭐")
r2.metric("카테고리별 m값", f"{m_val:,.1f}")
r3.metric("카테고리 평균 평점 (C)", f"{c_val:.2f} ⭐")
r4.metric("보정 평점", f"{corrected:.2f} ⭐", delta=f"{-decline:+.2f}", delta_color="inverse")

fig_gauge = go.Figure()
fig_gauge.add_trace(go.Bar(
    y=["원래 평점", "보정 평점"],
    x=[rating, corrected],
    orientation="h",
    marker_color=["#4C78A8", "#F58518"],
    text=[f"{rating:.2f}", f"{corrected:.2f}"],
    textposition="outside",
))
fig_gauge.update_layout(
    title="원래 평점 vs 보정 평점",
    xaxis=dict(range=[0, 5.5], title="평점"),
    height=300,
    showlegend=False,
)
st.plotly_chart(fig_gauge, use_container_width=True)

st.markdown("---")

# ============================================================
# 4-3. 판단 메시지
# ============================================================
st.markdown("## 4-3. 판단 메시지")

level, message = judge_message(decline)
if level == "success":
    st.success(f"✅ {message}")
elif level == "warning":
    st.warning(f"⚠️ {message}")
else:
    st.error(f"🚨 {message}")

st.caption("※ 위 기준은 절대적인 구매 기준이 아니라, 본 프로젝트에서 제안하는 참고용 기준입니다.")

st.markdown("---")

# ============================================================
# 4-4. 두 상품 비교 기능
# ============================================================
st.markdown("## 4-4. 두 상품 비교")
st.caption("두 상품의 카테고리 · 평점 · 리뷰 수를 각각 입력하면, 보정 평점을 기준으로 비교해드립니다.")

col_x, col_y = st.columns(2)

with col_x:
    st.markdown("#### 🅰️ 상품 A")
    cat_a = st.selectbox("카테고리 (A)", CATEGORY_LIST, key="cat_a")
    rating_a = st.number_input("평점 (A)", min_value=1.0, max_value=5.0, value=4.9, step=0.1, key="rating_a")
    count_a = st.number_input("리뷰 수 (A)", min_value=0, value=12, step=1, key="count_a")

with col_y:
    st.markdown("#### 🅱️ 상품 B")
    cat_b = st.selectbox("카테고리 (B)", CATEGORY_LIST, key="cat_b")
    rating_b = st.number_input("평점 (B)", min_value=1.0, max_value=5.0, value=4.4, step=0.1, key="rating_b")
    count_b = st.number_input("리뷰 수 (B)", min_value=0, value=3200, step=1, key="count_b")

if st.button("🔍 두 상품 비교하기", type="primary", use_container_width=True):
    corrected_a, m_a, c_a = calc_bayesian_rating(rating_a, count_a, cat_a)
    corrected_b, m_b, c_b = calc_bayesian_rating(rating_b, count_b, cat_b)
    decline_a = rating_a - corrected_a
    decline_b = rating_b - corrected_b
    _, msg_a = judge_message(decline_a)
    _, msg_b = judge_message(decline_b)

    compare_table = pd.DataFrame({
        "항목": ["카테고리", "원래 평점", "리뷰 수", "보정 평점", "하락폭", "판단"],
        "상품 A": [cat_a, f"{rating_a:.2f}", f"{count_a:,}", f"{corrected_a:.2f}", f"{decline_a:+.2f}", msg_a],
        "상품 B": [cat_b, f"{rating_b:.2f}", f"{count_b:,}", f"{corrected_b:.2f}", f"{decline_b:+.2f}", msg_b],
    })
    st.dataframe(compare_table.set_index("항목"), use_container_width=True)

    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Bar(name="원래 평점", x=["상품 A", "상품 B"], y=[rating_a, rating_b], marker_color="#4C78A8"))
    fig_cmp.add_trace(go.Bar(name="보정 평점", x=["상품 A", "상품 B"], y=[corrected_a, corrected_b], marker_color="#F58518"))
    fig_cmp.update_layout(barmode="group", yaxis=dict(range=[0, 5.5], title="평점"), height=380, title="상품 A vs 상품 B 비교")
    st.plotly_chart(fig_cmp, use_container_width=True)

    if corrected_a > corrected_b:
        winner, winner_val, loser_val = "상품 A", corrected_a, corrected_b
    elif corrected_b > corrected_a:
        winner, winner_val, loser_val = "상품 B", corrected_b, corrected_a
    else:
        winner = None

    if winner:
        st.info(
            f"💡 보정 평점과 리뷰 신뢰도를 기준으로 볼 때, **{winner}**(보정 평점 {winner_val:.2f})이(가) "
            f"상대적으로 더 안정적인 선택으로 판단됩니다. (비교 대상 보정 평점 {loser_val:.2f})"
        )
    else:
        st.info("💡 두 상품의 보정 평점이 동일합니다. 다른 조건(가격, 할인율 등)을 함께 고려해보세요.")

    st.caption("※ 이 비교는 절대적인 구매 기준이 아니라, 리뷰 수를 고려한 참고용 신뢰도 비교입니다.")
