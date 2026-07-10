import pandas as pd
from pathlib import Path


# =========================================================
# 베이지안 타겟 생성 코드
# =========================================================
# 목표:
# 전처리 완료 데이터에 bayesian_rating 컬럼을 추가한다.
#
# 기준:
# rating_count >= m_value
# → 믿을만한 상품
# → 원래 rating을 bayesian_rating에 할당
#
# rating_count < m_value
# → 의심 상품
# → 베이지안 공식으로 보정 평점 계산
#
# 공식:
# bayesian_rating = (v * R + m * C) / (v + m)
#
# v = 해당 상품의 리뷰 수 rating_count
# R = 해당 상품의 원래 평점 rating
# m = 해당 카테고리의 리뷰 수 Q1, 즉 25% 값
# C = 해당 카테고리의 평균 평점
# =========================================================


# =========================================================
# 1. 파일 경로 설정
# =========================================================

DATA_PATH = Path(r"C:\Users\minky\OneDrive\Desktop\amazon_preprocessed_no_bayesian_other.csv")

OUT_DIR = Path(r"C:\Users\minky\OneDrive\Desktop\bayesian_target_outputs")
OUT_DIR.mkdir(exist_ok=True)


# =========================================================
# 2. 데이터 불러오기
# =========================================================

df = pd.read_csv(DATA_PATH)

print("데이터 크기:", df.shape)


# =========================================================
# 3. 숫자형 컬럼 정리
# =========================================================

number_columns = [
    "actual_price",
    "discounted_price",
    "discount_percentage",
    "rating",
    "rating_count",
    "review_length",
]

for col in number_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# bayesian_rating 계산에 필요한 값이 없는 행 제거
df = df.dropna(subset=["main_category", "rating", "rating_count"]).copy()

print("필수 결측 제거 후 데이터 크기:", df.shape)


# =========================================================
# 4. 카테고리별 m_value 계산
# =========================================================
# m_value = 각 main_category별 rating_count의 Q1, 즉 25% 값

m_table = (
    df.groupby("main_category")["rating_count"]
    .quantile(0.25)
    .reset_index()
)

m_table = m_table.rename(columns={"rating_count": "m_value"})
m_table["m_value"] = m_table["m_value"].round(1)

print("\n[카테고리별 m_value]")
print(m_table)


# =========================================================
# 5. 카테고리별 C 계산
# =========================================================
# C = 각 main_category별 평균 평점

c_table = (
    df.groupby("main_category")["rating"]
    .mean()
    .reset_index()
)

c_table = c_table.rename(columns={"rating": "category_mean_rating"})
c_table["category_mean_rating"] = c_table["category_mean_rating"].round(4)

print("\n[카테고리별 평균 평점 C]")
print(c_table)


# =========================================================
# 6. m_value와 C 합치기
# =========================================================

param_table = pd.merge(
    m_table,
    c_table,
    on="main_category",
    how="left"
)

print("\n[카테고리별 베이지안 파라미터]")
print(param_table)


# =========================================================
# 7. 원본 데이터에 m_value와 C 붙이기
# =========================================================

df = pd.merge(
    df,
    param_table,
    on="main_category",
    how="left"
)


# =========================================================
# 8. if문으로 bayesian_rating 계산하는 함수 만들기
# =========================================================

def make_bayesian_rating(row):
    v = row["rating_count"]
    R = row["rating"]
    m = row["m_value"]
    C = row["category_mean_rating"]

    # 믿을만한 상품이면 원래 rating 그대로 사용
    if v >= m:
        return R

    # 의심 상품이면 베이지안 공식 적용
    else:
        bayesian_score = ((v * R) + (m * C)) / (v + m)
        return bayesian_score


# =========================================================
# 9. bayesian_rating 생성
# =========================================================

df["bayesian_rating"] = df.apply(make_bayesian_rating, axis=1)
df["bayesian_rating"] = df["bayesian_rating"].round(4)


# =========================================================
# 10. 확인용 컬럼 만들기
# =========================================================

df["is_suspicious"] = df["rating_count"] < df["m_value"]

df["rating_gap"] = (df["rating"] - df["bayesian_rating"]).round(4)

df["target_rule"] = "reliable: keep original rating"
df.loc[df["is_suspicious"], "target_rule"] = "suspicious: apply Bayesian formula"


print("\n[bayesian_rating 생성 확인]")
print(
    df[
        [
            "main_category",
            "rating",
            "rating_count",
            "m_value",
            "is_suspicious",
            "category_mean_rating",
            "bayesian_rating",
            "target_rule",
        ]
    ].head(10)
)


# =========================================================
# 11. 최종 데이터셋 저장
# =========================================================
# 팀원2에게 넘길 최종 파일
# 확인용 컬럼은 빼고 bayesian_rating만 추가

final_columns = [
    "product_id",
    "product_name",
    "main_category",
    "actual_price",
    "discounted_price",
    "discount_percentage",
    "rating",
    "rating_count",
    "review_content",
    "review_length",
    "bayesian_rating",
]

final_df = df[final_columns].copy()

FINAL_OUT = OUT_DIR / "amazon_with_bayesian_rating_category_q1_if.csv"

final_df.to_csv(
    FINAL_OUT,
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# 12. 검증용 상세 파일 저장
# =========================================================
# 어떤 상품이 의심 상품인지 확인할 수 있는 파일

detail_columns = [
    "product_id",
    "product_name",
    "main_category",
    "rating",
    "rating_count",
    "m_value",
    "category_mean_rating",
    "is_suspicious",
    "bayesian_rating",
    "rating_gap",
    "target_rule",
]

detail_df = df[detail_columns].copy()

DETAIL_OUT = OUT_DIR / "bayesian_target_detail_check.csv"

detail_df.to_csv(
    DETAIL_OUT,
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# 13. 카테고리별 m, C 파일 저장
# =========================================================

PARAM_OUT = OUT_DIR / "bayesian_category_parameters_q1.csv"

param_table.to_csv(
    PARAM_OUT,
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# 14. 순위 변화 Top 50 저장
# =========================================================
# 원래 rating 기준 순위와 bayesian_rating 기준 순위 비교

df["rating_rank"] = df["rating"].rank(method="first", ascending=False).astype(int)

df["bayesian_rank"] = df["bayesian_rating"].rank(method="first", ascending=False).astype(int)

df["rank_change"] = df["bayesian_rank"] - df["rating_rank"]

df["abs_rank_change"] = df["rank_change"].abs()

top50 = (
    df.sort_values("abs_rank_change", ascending=False)
    .head(50)
    [
        [
            "product_id",
            "product_name",
            "main_category",
            "rating",
            "rating_count",
            "m_value",
            "category_mean_rating",
            "is_suspicious",
            "bayesian_rating",
            "rating_gap",
            "rating_rank",
            "bayesian_rank",
            "rank_change",
            "target_rule",
        ]
    ]
)

TOP50_OUT = OUT_DIR / "bayesian_rank_change_top50.csv"

top50.to_csv(
    TOP50_OUT,
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# 15. 리포트 저장
# =========================================================

total_count = len(df)

suspicious_count = df["is_suspicious"].sum()

reliable_count = total_count - suspicious_count

suspicious_ratio = suspicious_count / total_count * 100

report = []

report.append("[베이지안 타겟 생성 리포트]")
report.append("")
report.append("[적용 기준]")
report.append("rating_count >= m_value: 믿을만한 상품 -> 원래 rating 유지")
report.append("rating_count < m_value: 의심 상품 -> 베이지안 공식 적용")
report.append("")
report.append("[데이터 요약]")
report.append(f"전체 상품 수: {total_count}")
report.append(f"믿을만한 상품 수: {reliable_count}")
report.append(f"의심 상품 수: {suspicious_count}")
report.append(f"의심 상품 비율: {suspicious_ratio:.2f}%")
report.append(f"원래 평점 평균: {df['rating'].mean():.4f}")
report.append(f"bayesian_rating 평균: {df['bayesian_rating'].mean():.4f}")
report.append(f"평균 보정폭 rating - bayesian_rating: {df['rating_gap'].mean():.4f}")
report.append("")
report.append("[카테고리별 m, C]")
report.append(param_table.to_string(index=False))
report.append("")
report.append("[순위 변화 Top 50 중 상위 10개 미리보기]")
report.append(
    top50.head(10)[
        [
            "product_name",
            "main_category",
            "rating",
            "rating_count",
            "m_value",
            "is_suspicious",
            "bayesian_rating",
            "rank_change",
        ]
    ].to_string(index=False)
)

REPORT_OUT = OUT_DIR / "bayesian_target_report_if_top50.txt"

REPORT_OUT.write_text(
    "\n".join(report),
    encoding="utf-8"
)


# =========================================================
# 16. 실행 결과 출력
# =========================================================

print("\n베이지안 타겟 생성 완료!")
print("결과 저장 폴더:", OUT_DIR)
print("최종 데이터셋:", FINAL_OUT)
print("검증용 상세 파일:", DETAIL_OUT)
print("카테고리별 파라미터:", PARAM_OUT)
print("순위 변화 Top 50:", TOP50_OUT)
print("리포트:", REPORT_OUT)