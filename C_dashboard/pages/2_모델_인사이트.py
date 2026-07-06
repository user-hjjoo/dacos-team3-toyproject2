import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="모델 인사이트", page_icon="🧠", layout="wide")

st.title("🧠 모델 인사이트 페이지")
st.caption("리뷰 신뢰도에 영향을 미치는 요인 (더미 그래프)")

# 더미 SHAP 데이터
shap_df = pd.DataFrame({
    "feature": ["리뷰 수", "평점 표준편차", "최근 리뷰 비중", "리뷰 길이", "평점"],
    "importance": [0.32, 0.24, 0.18, 0.11, 0.08],
})

fig = px.bar(
    shap_df.sort_values("importance"),
    x="importance", y="feature", orientation="h",
    title="SHAP 요약 그래프 (더미)",
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("🏆 Top 5 요인")
cols = st.columns(5)
medal = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
top5 = shap_df.sort_values("importance", ascending=False).reset_index(drop=True)
for i, col in enumerate(cols):
    with col:
        st.metric(f"{medal[i]} {top5.loc[i,'feature']}", f"{top5.loc[i,'importance']:.2f}")
