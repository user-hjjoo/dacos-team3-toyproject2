import json
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

st.set_page_config(page_title="모델 인사이트", page_icon="🧠", layout="wide")


@st.cache_data
def load_model_metrics() -> pd.DataFrame:
    with open(DATA_DIR / "B_model_metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)
    return pd.DataFrame(metrics)


@st.cache_data
def load_shap_importance() -> pd.DataFrame:
    shap_df = pd.read_csv(DATA_DIR / "B_shap_importance.csv")
    shap_df.columns = [c.strip() for c in shap_df.columns]
    return shap_df


@st.cache_data
def load_shap_raw() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "B_shap_values_raw.csv")


metrics_df = load_model_metrics()
shap_df = load_shap_importance()
shap_raw = load_shap_raw()

st.title("🧠 모델 인사이트 페이지")
st.caption("모델 성능 비교와 SHAP 해석을 통해, 보정 평점 예측에 영향을 준 주요 요인을 살펴봅니다.")

# ============================================================
# 3-1. 모델 성능 비교
# ============================================================
st.markdown("## 3-1. 모델 성능 비교")

best_model_row = metrics_df.loc[metrics_df["rmse"].idxmin()]

fig1 = px.bar(
    metrics_df.sort_values("rmse"),
    x="model",
    y="rmse",
    color="model",
    text="rmse",
    labels={"model": "모델", "rmse": "RMSE"},
    title="모델별 RMSE 비교 (낮을수록 우수)",
)
fig1.update_traces(texttemplate="%{text:.4f}", textposition="outside")
fig1.update_layout(height=450, showlegend=False)
st.plotly_chart(fig1, use_container_width=True)

st.markdown(f"#### 🏆 Best Model: **{best_model_row['model']}**")
b1, b2, b3 = st.columns(3)
b1.metric("RMSE", f"{best_model_row['rmse']:.4f}")
b2.metric("MAE", f"{best_model_row['mae']:.4f}")
b3.metric("R²", f"{best_model_row['r2']:.4f}")

st.info(
    f"💡 **인사이트**: {best_model_row['model']}가 가장 낮은 RMSE와 MAE, 가장 높은 R²를 기록하여 "
    "최종 모델로 선정되었습니다."
)
st.warning(
    f"⚠️ **한계**: R²는 {best_model_row['r2']:.4f}로 높지 않으므로, 모델 예측값을 절대적인 품질 점수로 보기보다 "
    "주요 영향 요인을 해석하는 데 활용하였습니다."
)

st.markdown("---")

# ============================================================
# 3-2. SHAP 중요도 순위
# ============================================================
st.markdown("## 3-2. SHAP 중요도 순위 (Top 7)")

# 내부 변수명 → 한글 이름 매핑
FEATURE_NAME_MAP = {
    "num__log_rating_count": "리뷰 수",
    "num__discount_percentage": "할인율",
    "num__log_actual_price": "정가",
    "num__log_review_length": "리뷰 길이",
    "num__log_discounted_price": "할인가",
    "cat__main_category_Headphones,Earbuds&Accessories": "카테고리: 이어폰·헤드폰",
    "cat__main_category_Accessories&Peripherals": "카테고리: 액세서리·주변기기",
    "cat__main_category_Kitchen&HomeAppliances": "카테고리: 주방·가전",
    "cat__main_category_Heating,Cooling&AirQuality": "카테고리: 냉난방·공기질",
    "cat__main_category_Mobiles&Accessories": "카테고리: 모바일·액세서리",
    "cat__main_category_Other": "카테고리: 기타",
    "cat__main_category_WearableTechnology": "카테고리: 웨어러블",
    "cat__main_category_HomeTheater,TV&Video": "카테고리: 홈시어터·TV",
    "cat__main_category_OfficePaperProducts": "카테고리: 사무용품",
    "cat__main_category_HomeStorage&Organization": "카테고리: 홈스토리지·정리",
    "cat__main_category_ExternalDevices&DataStorage": "카테고리: 외장저장장치",
    "cat__main_category_HomeAudio": "카테고리: 홈오디오",
    "cat__main_category_NetworkingDevices": "카테고리: 네트워크 장비",
    "cat__main_category_Cameras&Photography": "카테고리: 카메라·사진",
}

shap_df["feature_kr"] = shap_df["feature"].map(FEATURE_NAME_MAP).fillna(shap_df["feature"])
top7 = shap_df.sort_values("importance", ascending=False).head(7).sort_values("importance", ascending=True)

fig2 = go.Figure(
    go.Bar(
        x=top7["importance"],
        y=top7["feature_kr"],
        orientation="h",
        marker_color="#F58518",
    )
)
fig2.update_layout(
    title="SHAP 중요도 Top 7 (|SHAP value| 평균)",
    xaxis_title="평균 |SHAP value|",
    yaxis_title="피처",
    height=450,
)
st.plotly_chart(fig2, use_container_width=True)

top5_names = top7.sort_values("importance", ascending=False).head(5)["feature_kr"].tolist()
st.info(f"💡 **Top 5 요인**: {' / '.join(top5_names)}")

st.warning(
    "⚠️ **주의**: 리뷰 수는 보정 평점 계산에도 직접 사용되므로, 중요도 1위 결과를 새로운 독립적 발견으로 "
    "해석하기보다 보정 평점 구조가 반영된 결과로 보아야 합니다."
)

st.markdown("---")

# ============================================================
# 3-3. SHAP 방향성 해석 (Dependence Plot)
# ============================================================
st.markdown("## 3-3. SHAP 방향성 해석 (Dependence Plot)")
st.caption("핵심 변수 3개(리뷰 수 · 할인율 · 정가)에 대해서만 방향성을 살펴봅니다.")

dep_tab1, dep_tab2, dep_tab3 = st.tabs(["리뷰 수", "할인율", "정가"])

with dep_tab1:
    fig_d1 = px.scatter(
        shap_raw,
        x="num__log_rating_count",
        y="shap_value_num__log_rating_count",
        labels={"num__log_rating_count": "리뷰 수 (log 변환)", "shap_value_num__log_rating_count": "SHAP value"},
        title="리뷰 수 Dependence Plot",
        opacity=0.6,
    )
    fig_d1.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_d1.update_layout(height=420)
    st.plotly_chart(fig_d1, use_container_width=True)
    st.info(
        "💡 리뷰 수가 증가할수록 SHAP 값이 대체로 음수에서 양수 방향으로 이동하는 경향이 나타납니다. "
        "즉, 모델은 리뷰 수가 많은 상품일수록 보정 평점 예측값을 높이는 방향으로 반영하는 경향을 학습했습니다."
    )

with dep_tab2:
    fig_d2 = px.scatter(
        shap_raw,
        x="num__discount_percentage",
        y="shap_value_num__discount_percentage",
        labels={"num__discount_percentage": "할인율 (%)", "shap_value_num__discount_percentage": "SHAP value"},
        title="할인율 Dependence Plot",
        opacity=0.6,
        color_discrete_sequence=["#E45756"],
    )
    fig_d2.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_d2.update_layout(height=420)
    st.plotly_chart(fig_d2, use_container_width=True)
    st.info(
        "💡 할인율이 높아질수록 SHAP 값이 음의 방향으로 이동하는 경향이 비교적 뚜렷하게 나타납니다. "
        "높은 할인율이 보정 평점 예측값을 낮추는 방향으로 작용하는 경향을 보입니다. "
        "다만 이는 데이터에서 확인된 연관성이며, 인과관계로 해석해서는 안 됩니다."
    )

with dep_tab3:
    fig_d3 = px.scatter(
        shap_raw,
        x="num__log_actual_price",
        y="shap_value_num__log_actual_price",
        labels={"num__log_actual_price": "정가 (log 변환)", "shap_value_num__log_actual_price": "SHAP value"},
        title="정가 Dependence Plot",
        opacity=0.6,
        color_discrete_sequence=["#54A24B"],
    )
    fig_d3.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_d3.update_layout(height=420)
    st.plotly_chart(fig_d3, use_container_width=True)
    st.info(
        "💡 가격 변수는 리뷰 수나 할인율에 비해 한 방향으로 뚜렷한 관계가 나타나지 않아, "
        "가격대와 다른 변수의 상호작용에 따라 영향이 달라지는 것으로 볼 수 있습니다."
    )

st.markdown("---")

# ============================================================
# 종합 인사이트
# ============================================================
st.subheader("✅ 모델 인사이트 종합")
st.success(
    "- Random Forest가 가장 좋은 성능을 보였지만 R²가 낮아, 모델 예측 자체보다 SHAP을 활용한 영향 요인 해석에 중점을 두었습니다.\n"
    "- 리뷰 수가 가장 중요한 변수로 나타났으며, 리뷰 수가 많을수록 보정 평점 예측에 긍정적으로 작용하는 경향을 확인하였습니다.\n"
    "- 할인율·가격·리뷰 길이·카테고리도 주요 영향 요인으로 나타났으며, 특히 높은 할인율은 보정 평점 예측에 부정적으로 작용하는 경향을 보였습니다."
)
