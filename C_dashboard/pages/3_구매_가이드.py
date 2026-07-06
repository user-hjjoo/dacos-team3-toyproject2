import streamlit as st

st.set_page_config(page_title="구매 가이드", page_icon="🧮", layout="wide")

st.title("🧮 구매 가이드 페이지")
st.caption("리뷰 수 / 평점을 입력하면 보정 평점을 계산합니다.")

review_count = st.number_input("리뷰 수", min_value=0, value=12)
rating = st.number_input("평점", min_value=1.0, max_value=5.0, value=4.9, step=0.1)

m, c = 50, 4.0  # 더미 기준값
corrected = (review_count / (review_count + m)) * rating + (m / (review_count + m)) * c

st.metric("보정 평점", f"{corrected:.2f} ⭐")

if review_count < 20 and (rating - corrected) > 0.3:
    st.warning("⚠️ 신중하게 검토하세요.")
else:
    st.success("✅ 비교적 신뢰할 수 있는 평점입니다.")
