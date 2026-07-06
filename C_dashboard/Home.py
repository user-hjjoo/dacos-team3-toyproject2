import streamlit as st
import pandas as pd

st.set_page_config(page_title="리뷰 신뢰도 분석 대시보드", page_icon="⭐", layout="wide")

df = pd.read_csv("C:\dacos-team3-toyproject2\data\processed\dummy_amazon.csv")

st.title("⭐ 리뷰 신뢰도 분석 대시보드")
st.caption("리뷰 수와 평점만으로 상품을 신뢰해도 될까?")

st.info("**핵심 질문**  \n\"리뷰 적은 5점 vs 리뷰 많은 4점, 뭘 사야 할까?\"")

st.markdown("---")

st.subheader("📊 전체 데이터 요약")
col1, col2, col3, col4 = st.columns(4)
col1.metric("전체 상품 수", f"{df['product_id'].nunique():,} 개")
col2.metric("카테고리 수", f"{df['main_category'].nunique()} 개")
col3.metric("평균 평점", f"{df['rating'].mean():.2f} ⭐")
col4.metric("평균 리뷰 수", f"{df['rating_count'].mean():.0f} 개")

st.markdown("")
st.dataframe(df.head(10), use_container_width=True)
