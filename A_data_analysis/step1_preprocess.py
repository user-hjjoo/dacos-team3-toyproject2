import pandas as pd
import numpy as np
from pathlib import Path

# =========================
# 1. 파일 경로 설정
# =========================

RAW_PATH = Path(r"C:\Users\minky\OneDrive\Desktop\archive (1).zip")
DUMMY_PATH = Path(r"C:\Users\minky\OneDrive\Desktop\dummy_amazon_sales (1).csv")

REAL_OUT = Path(r"C:\Users\minky\OneDrive\Desktop\amazon_preprocessed_no_bayesian_other.csv")
DUMMY_OUT = Path(r"C:\Users\minky\OneDrive\Desktop\dummy_amazon_sales_no_bayesian_other.csv")
COUNTS_OUT = Path(r"C:\Users\minky\OneDrive\Desktop\main_category_counts_no_bayesian_other.csv")
REPORT_OUT = Path(r"C:\Users\minky\OneDrive\Desktop\schema_check_no_bayesian_other.txt")


# =========================
# 2. 최종 컬럼 정하기
# =========================
# 실제 데이터와 더미 데이터가 최종적으로 가져야 하는 컬럼

FINAL_COLUMNS = [
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
]


# =========================
# 3. 원본 데이터 불러오기
# =========================

df = pd.read_csv(RAW_PATH, compression="zip")

print("원본 데이터 크기:", df.shape)


# =========================
# 4. 숫자형 컬럼 전처리
# =========================
# 가격: ₹, 쉼표 제거
# 할인율: % 제거
# 리뷰 수: 쉼표 제거

df["actual_price"] = (
    df["actual_price"]
    .astype(str)
    .str.replace("₹", "", regex=False)
    .str.replace(",", "", regex=False)
)

df["discounted_price"] = (
    df["discounted_price"]
    .astype(str)
    .str.replace("₹", "", regex=False)
    .str.replace(",", "", regex=False)
)

df["discount_percentage"] = (
    df["discount_percentage"]
    .astype(str)
    .str.replace("%", "", regex=False)
)

df["rating_count"] = (
    df["rating_count"]
    .astype(str)
    .str.replace(",", "", regex=False)
)

# 실제 숫자형으로 변환
df["actual_price"] = pd.to_numeric(df["actual_price"], errors="coerce")
df["discounted_price"] = pd.to_numeric(df["discounted_price"], errors="coerce")
df["discount_percentage"] = pd.to_numeric(df["discount_percentage"], errors="coerce")
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
df["rating_count"] = pd.to_numeric(df["rating_count"], errors="coerce")

# rating_count 결측치는 리뷰가 없는 것으로 보고 0으로 대체
df["rating_count"] = df["rating_count"].fillna(0)


# =========================
# 5. 이상치 / 결측치 제거
# =========================
# rating이 0~5 범위를 벗어나거나,
# 필수 컬럼이 비어 있으면 제거

df = df[
    (df["rating"].notna()) &
    (df["rating"] >= 0) &
    (df["rating"] <= 5) &
    (df["product_id"].notna()) &
    (df["category"].notna()) &
    (df["actual_price"].notna()) &
    (df["discounted_price"].notna()) &
    (df["discount_percentage"].notna())
].copy()

print("이상치/결측치 제거 후:", df.shape)


# =========================
# 6. 중복 상품 제거
# =========================
# product_id가 같은 상품은 하나만 남김

df = df.drop_duplicates(subset="product_id", keep="first").copy()

print("중복 제거 후:", df.shape)


# =========================
# 7. main_category 만들기
# =========================
# 원본 category 예시:
# Computers&Accessories|Accessories&Peripherals|Cables&Accessories|Cables|USBCables
#
# 여기서 두 번째 항목인 Accessories&Peripherals를 main_category로 사용

df["second_category"] = (
    df["category"]
    .astype(str)
    .str.split("|")
    .str[1]
    .str.strip()
)

# 두 번째 세부항목별 상품 수 세기
category_counts = df["second_category"].value_counts()

# 상품 수가 15개 이상인 카테고리만 유지
major_categories = category_counts[category_counts >= 15].index.tolist()

# 15개 미만이면 Other로 묶기
df["main_category"] = np.where(
    df["second_category"].isin(major_categories),
    df["second_category"],
    "Other"
)


# =========================
# 8. review_length 만들기
# =========================
# review_content의 글자 수를 계산해서 새 컬럼 생성

df["review_content"] = df["review_content"].fillna("")
df["review_length"] = df["review_content"].astype(str).str.len()


# =========================
# 9. 실제 데이터 최종 컬럼만 남기기
# =========================

real_final = df[FINAL_COLUMNS].copy()


# =========================
# 10. 더미 데이터 불러오기
# =========================

dummy = pd.read_csv(DUMMY_PATH)


# =========================
# 11. 더미 데이터 컬럼 맞추기
# =========================
# 더미 데이터에 필요한 컬럼이 없으면 새로 만들어줌

for col in FINAL_COLUMNS:
    if col not in dummy.columns:
        if col in ["actual_price", "discounted_price", "discount_percentage", "rating", "rating_count", "review_length"]:
            dummy[col] = 0
        elif col == "main_category":
            dummy[col] = "Other"
        else:
            dummy[col] = ""

# 컬럼 순서를 실제 데이터와 동일하게 맞춤
dummy_final = dummy[FINAL_COLUMNS].copy()


# =========================
# 12. 더미 데이터 main_category 맞추기
# =========================
# Others, others, other 같은 값은 전부 Other로 통일

dummy_final["main_category"] = dummy_final["main_category"].replace({
    "Others": "Other",
    "others": "Other",
    "other": "Other"
})

# 실제 데이터에 없는 카테고리는 Other로 바꿈
allowed_categories = major_categories + ["Other"]

dummy_final.loc[
    ~dummy_final["main_category"].isin(allowed_categories),
    "main_category"
] = "Other"

# 더미 데이터에도 실제 데이터의 모든 main_category가 최소 1번씩 나오게 함
for i, category_name in enumerate(allowed_categories):
    if i < len(dummy_final):
        dummy_final.loc[dummy_final.index[i], "main_category"] = category_name


# =========================
# 13. main_category unique 출력
# =========================

main_category_unique = real_final["main_category"].unique()

print("\n[main_category unique values]")
print(main_category_unique)

print("\n[main_category unique count]")
print(len(main_category_unique))


# =========================
# 14. 검증
# =========================

schema_match = list(real_final.columns) == list(dummy_final.columns)

real_category_set = set(real_final["main_category"].unique())
dummy_category_set = set(dummy_final["main_category"].unique())

category_match = real_category_set == dummy_category_set

print("\n[검증 결과]")
print("실제/더미 컬럼 일치:", schema_match)
print("실제/더미 main_category 일치:", category_match)
print("최종 컬럼 수:", len(FINAL_COLUMNS))
print("main_category 개수:", real_final["main_category"].nunique())


# =========================
# 15. main_category별 상품 수 저장용 표 만들기
# =========================

main_category_counts = (
    real_final["main_category"]
    .value_counts()
    .rename_axis("main_category")
    .reset_index(name="product_count")
)


# =========================
# 16. 결과 파일 저장
# =========================

real_final.to_csv(REAL_OUT, index=False, encoding="utf-8-sig")
dummy_final.to_csv(DUMMY_OUT, index=False, encoding="utf-8-sig")
main_category_counts.to_csv(COUNTS_OUT, index=False, encoding="utf-8-sig")


# =========================
# 17. 리포트 저장
# =========================

report = []

report.append("[전처리 결과 리포트]")
report.append(f"최종 실제 데이터 행 수: {len(real_final)}")
report.append(f"최종 더미 데이터 행 수: {len(dummy_final)}")
report.append(f"최종 컬럼 수: {len(FINAL_COLUMNS)}")
report.append(f"실제/더미 컬럼 일치: {schema_match}")
report.append(f"실제/더미 main_category 일치: {category_match}")
report.append("")
report.append("[최종 컬럼명]")
report.append(", ".join(FINAL_COLUMNS))
report.append("")
report.append("[main_category unique values]")

for category_name in main_category_unique:
    report.append(f"- {category_name}")

report.append("")
report.append("[main_category별 상품 수]")
report.append(main_category_counts.to_string(index=False))

REPORT_OUT.write_text("\n".join(report), encoding="utf-8")


print("\n저장 완료!")
print(REAL_OUT)
print(DUMMY_OUT)
print(COUNTS_OUT)
print(REPORT_OUT)