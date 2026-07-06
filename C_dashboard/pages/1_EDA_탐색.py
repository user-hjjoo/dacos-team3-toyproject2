import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="EDA 탐색", page_icon="🔎", layout="wide")

df = pd.read_csv("C:\dacos-team3-toyproject2\data\processed\dummy_amazon.csv")

st.title("🔎 EDA 탐색 페이지")
st.caption("리뷰 수 vs 평점 Funnel Plot")

categories = ["전체"] + sorted(df["main_category"].unique().tolist())
selected = st.selectbox("카테고리 선택", categories)

filtered = df if selected == "전체" else df[df["main_category"] == selected]

fig = px.scatter(
    filtered,
    x="rating_count",
    y="rating",
    color="main_category",
    hover_name="product_name",
    labels={"rating_count": "리뷰 수", "rating": "평점"},
    title="리뷰 수 vs 평점 (Funnel Plot)",
    opacity=0.7,
)
st.plotly_chart(fig, use_container_width=True)
