import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# =========================================================
# Amazon EDA 초보자용 코드
# =========================================================
# 이 코드는 전처리가 끝난 CSV 파일을 이용해서
# PDF Step2 기준의 EDA 그래프와 인사이트 파일을 만듭니다.
#
# 생성되는 산출물:
# 1. EDA 그래프 이미지 5개
# 2. eda_insights.txt
# 3. main_category_m_values.csv
# =========================================================


# =========================================================
# 1. 파일 경로 설정
# =========================================================

# 본인 컴퓨터에서 전처리 완료 CSV 파일 위치
DATA_PATH = Path(r"data\processed\preprocessed_amazon.csv")

# 결과를 저장할 폴더
OUT_DIR = Path(r"result\eda")
GRAPH_DIR = OUT_DIR / "graphs"

# 폴더가 없으면 새로 만들기
OUT_DIR.mkdir(exist_ok=True)
GRAPH_DIR.mkdir(exist_ok=True)


# =========================================================
# 2. 데이터 불러오기
# =========================================================

df = pd.read_csv(DATA_PATH)

print("데이터 크기:", df.shape)
print("컬럼명:")
print(df.columns)


# =========================================================
# 3. 숫자형 컬럼 확인
# =========================================================
# 그래프를 그리려면 숫자 컬럼이 진짜 숫자형이어야 함

number_columns = [
    "actual_price",
    "discounted_price",
    "discount_percentage",
    "rating",
    "rating_count",
    "review_length"
]

for col in number_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# =========================================================
# 4. 그래프에 필요한 새 컬럼 만들기
# =========================================================

# 리뷰 수는 값 차이가 너무 커서 로그 변환한 컬럼을 만듦
# +1을 하는 이유: 리뷰 수가 0이면 log 계산이 안 되기 때문
df["log_review_count"] = np.log10(df["rating_count"] + 1)

# 리뷰 수를 5개 그룹으로 나누기
# 0-20%, 20-40%, 40-60%, 60-80%, 80-100%
df["review_count_group"] = pd.qcut(
    df["rating_count"].rank(method="first"),
    q=5,
    labels=["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
)


# =========================================================
# 5. main_category 순서 정하기
# =========================================================
# 상품 수가 많은 카테고리부터 그래프에 표시

category_count = df["main_category"].value_counts()
category_order = category_count.index.tolist()


# =========================================================
# 1-1. 퍼널 플롯
# 리뷰 수에 따른 평점 산점도
# =========================================================

plt.figure(figsize=(8, 5))
plt.scatter(df["log_review_count"], df["rating"], alpha=0.5)
plt.title("1-1. Funnel Plot: Review Count vs Rating")
plt.xlabel("log10(rating_count + 1)")
plt.ylabel("rating")
plt.tight_layout()
plt.savefig(GRAPH_DIR / "1-1_funnel_plot.png", dpi=200)
plt.close()

print("1-1 퍼널 플롯 저장 완료")


# =========================================================
# 1-2. 리뷰 수 그룹별 평점 박스플롯
# =========================================================

group_names = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]

box_data = []

for group in group_names:
    group_rating = df[df["review_count_group"] == group]["rating"]
    box_data.append(group_rating)

plt.figure(figsize=(8, 5))
plt.boxplot(box_data, tick_labels=group_names)
plt.title("1-2. Rating by Review Count Group")
plt.xlabel("review_count group")
plt.ylabel("rating")
plt.tight_layout()
plt.savefig(GRAPH_DIR / "1-2_rating_boxplot_by_review_group.png", dpi=200)
plt.close()

print("1-2 박스플롯 저장 완료")


# =========================================================
# 2. main_category별 상품 수 막대그래프
# =========================================================

plt.figure(figsize=(12, 6))
plt.bar(category_count.index, category_count.values)
plt.title("2. Product Count by Main Category")
plt.xlabel("main_category")
plt.ylabel("product_count")
plt.xticks(rotation=80)
plt.tight_layout()
plt.savefig(GRAPH_DIR / "2_product_count_by_main_category.png", dpi=200)
plt.close()

print("2 카테고리별 상품 수 그래프 저장 완료")


# =========================================================
# 3-1. main_category별 리뷰 수 히스토그램
# 14개 그래프를 하나의 큰 figure에 배치
# =========================================================

category_num = len(category_order)

# 4열짜리 그래프판 만들기
col_num = 4
row_num = int(np.ceil(category_num / col_num))

fig, axes = plt.subplots(row_num, col_num, figsize=(16, 12))

# axes를 1차원 리스트처럼 바꾸기
axes = axes.flatten()

for i, category in enumerate(category_order):
    category_data = df[df["main_category"] == category]

    axes[i].hist(category_data["log_review_count"], bins=15)
    axes[i].set_title(category, fontsize=9)
    axes[i].set_xlabel("log10(rating_count + 1)")
    axes[i].set_ylabel("product_count")

# 남는 빈 칸은 숨기기
for j in range(category_num, len(axes)):
    axes[j].axis("off")

plt.suptitle("3-1. Review Count Histogram by Main Category")
plt.tight_layout()
plt.savefig(GRAPH_DIR / "3-1_review_count_histograms.png", dpi=200)
plt.close()

print("3-1 카테고리별 히스토그램 저장 완료")


# =========================================================
# 3-2. main_category별 리뷰 수 박스플롯 + 개별 점
# =========================================================

box_data = []

for category in category_order:
    category_log_review = df[df["main_category"] == category]["log_review_count"]
    box_data.append(category_log_review)

plt.figure(figsize=(13, 6))

# 박스플롯
plt.boxplot(box_data, tick_labels=category_order)

# 박스플롯 위에 개별 데이터 점 찍기
for i, category in enumerate(category_order):
    category_log_review = df[df["main_category"] == category]["log_review_count"]

    # x축 위치
    x_position = i + 1

    # 점이 완전히 겹치지 않도록 x축을 아주 조금 흔들기
    x_jitter = np.random.normal(x_position, 0.04, size=len(category_log_review))

    plt.scatter(x_jitter, category_log_review, alpha=0.35, s=12)

plt.title("3-2. Review Count Boxplot by Main Category")
plt.xlabel("main_category")
plt.ylabel("log10(rating_count + 1)")
plt.xticks(rotation=80)
plt.tight_layout()
plt.savefig(GRAPH_DIR / "3-2_review_count_boxplot_points.png", dpi=200)
plt.close()

print("3-2 박스플롯 + 점 그래프 저장 완료")


# =========================================================
# 3-2 결과 CSV 만들기
# 카테고리별 rating_count 중앙값 = Bayesian 공식에 사용할 m
# =========================================================

m_table = (
    df.groupby("main_category")["rating_count"]
    .median()
    .reset_index()
)

m_table = m_table.rename(columns={"rating_count": "m_value"})

# 보기 좋게 상품 수 많은 순서로 정렬
m_table["main_category"] = pd.Categorical(
    m_table["main_category"],
    categories=category_order,
    ordered=True
)

m_table = m_table.sort_values("main_category")

# 소수점 1자리로 정리
m_table["m_value"] = m_table["m_value"].round(1)

# m_table.to_csv(
#     OUT_DIR / "main_category_m_values.csv",
#     index=False,
#     encoding="utf-8-sig"
# )

print("\n[3-2. main_category별 m_value]")
print(m_table)


# =========================================================
# 인사이트 TXT 만들기
# =========================================================

# 리뷰 수 그룹별 평점 요약
group_summary = (
    df.groupby("review_count_group", observed=False)
    .agg(
        rating_mean=("rating", "mean"),
        rating_median=("rating", "median"),
        rating_std=("rating", "std"),
        rating_min=("rating", "min"),
        rating_max=("rating", "max")
    )
    .reset_index()
)

low_group = group_summary.iloc[0]
high_group = group_summary.iloc[-1]

top_category_name = category_count.index[0]
top_category_count = category_count.iloc[0]

second_category_name = category_count.index[1]
second_category_count = category_count.iloc[1]

other_count = category_count["Other"]

highest_m_row = m_table.sort_values("m_value", ascending=False).iloc[0]
lowest_m_row = m_table.sort_values("m_value", ascending=True).iloc[0]

# insight = []

# insight.append("[EDA 인사이트 정리]")
# insight.append("")

# insight.append("1-1. 퍼널 플롯")
# insight.append(
#     f"- 리뷰 수가 적은 구간에서는 평점이 {low_group['rating_min']:.1f}점부터 {low_group['rating_max']:.1f}점까지 넓게 퍼져 있어, 리뷰 수가 적을수록 평점 변동성이 크다는 점을 확인할 수 있다."
# )
# insight.append(
#     "- 따라서 단순 평점만 보고 상품을 비교하기보다, 리뷰 수를 함께 고려한 보정 평점이 필요하다."
# )
# insight.append("")

# insight.append("1-2. 리뷰 수 그룹별 평점 박스플롯")
# insight.append(
#     f"- 리뷰 수 하위 20% 그룹의 평점 표준편차는 {low_group['rating_std']:.3f}, 상위 20% 그룹의 평점 표준편차는 {high_group['rating_std']:.3f}로 나타났다."
# )
# insight.append(
#     "- 리뷰 수 그룹별 평점 분포 차이를 통해 리뷰 수가 평점 신뢰도 판단에 중요한 기준이 될 수 있음을 확인할 수 있다."
# )
# insight.append("")

# insight.append("2. main_category별 상품 수 막대그래프")
# insight.append(
#     f"- 상품 수가 가장 많은 카테고리는 {top_category_name}({top_category_count}개), 두 번째는 {second_category_name}({second_category_count}개)이다."
# )
# insight.append(
#     f"- 상품 수가 15개 미만인 세부 카테고리들은 Other로 통합되었고, Other에는 총 {other_count}개 상품이 포함되었다."
# )
# insight.append("")

# insight.append("3-1. main_category별 리뷰 수 히스토그램")
# insight.append(
#     "- 카테고리별 리뷰 수 분포는 대부분 한쪽으로 치우쳐 있으며, 일부 상품에 리뷰 수가 집중되는 형태가 나타난다."
# )
# insight.append(
#     "- 카테고리마다 리뷰 수 분포가 다르므로 전체 데이터에 동일한 기준을 적용하기보다는 카테고리별 기준이 필요하다."
# )
# insight.append("")

# insight.append("3-2. main_category별 리뷰 수 박스플롯 + 점")
# insight.append(
#     f"- 카테고리별 m_value는 {lowest_m_row['main_category']} {lowest_m_row['m_value']}개부터 {highest_m_row['main_category']} {highest_m_row['m_value']}개까지 차이가 난다."
# )
# insight.append(
#     "- 따라서 3-2에서 구한 14개의 m_value를 카테고리별 베이지안 보정 평점 공식의 기준 리뷰 수로 사용할 수 있다."
# )

# with open(OUT_DIR / "eda_insights.txt", "w", encoding="utf-8") as f:
#     f.write("\n".join(insight))


# print("\n인사이트 TXT 저장 완료")
# print("저장 폴더:", OUT_DIR)
print("그래프 저장 폴더:", GRAPH_DIR)
# print("m_value CSV:", OUT_DIR / "main_category_m_values.csv")
# print("인사이트 TXT:", OUT_DIR / "eda_insights.txt")
