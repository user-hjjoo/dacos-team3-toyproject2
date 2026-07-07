import pandas as pd
import numpy as np
import re
from pathlib import Path

RAW_PATH = Path(r"C:\Users\minky\OneDrive\Desktop\archive (1).zip")
DUMMY_PATH = Path(r"C:\Users\minky\OneDrive\Desktop\dummy_amazon_sales (1).csv")

REAL_OUT = Path(r"C:\Users\minky\OneDrive\Desktop\amazon_preprocessed_schema_matched.csv")
REAL_OUT_LEGACY = Path(r"C:\Users\minky\OneDrive\Desktop\amazon_preprocessed.csv")
DUMMY_OUT = Path(r"C:\Users\minky\OneDrive\Desktop/dummy_amazon_sales_schema_matched.csv")
REPORT_OUT = Path(r"C:\Users\minky\OneDrive\Desktop/schema_category_check.txt")
COUNTS_OUT = Path(r"C:\Users\minky\OneDrive\Desktop/main_category_counts.csv")
REMOVED_OUT = Path(r"C:\Users\minky\OneDrive\Desktop/amazon_removed_rows_schema_matched.csv")

#최종 데이터 컬럼명 고정
#실제 데이터와 더미 데이터의 컬럼명/컬럼 순서를 똑같이 맞춤
FINAL_COLUMNS = [
    'product_id',
    'product_name',
    'main_category',
    'actual_price',
    'discounted_price',
    'discount_percentage',
    'rating',
    'rating_count',
    'review_content',
    'bayesian_rating',
    'review_length',
]

#숫자형 변환 함수

#가격 컬럼에서 ₹, 쉼표 제거하고 숫자로 변환
def clean_price(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str)
         .str.replace('₹', '', regex=False)
         .str.replace(',', '', regex=False)
         .str.strip()
         .replace({'nan': np.nan, '': np.nan}),
        errors='coerce'
    )

#할인율 컬럼에서 % 제거하고 숫자로 변환
def clean_percent(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str)
         .str.replace('%', '', regex=False)
         .str.strip()
         .replace({'nan': np.nan, '': np.nan}),
        errors='coerce'
    )

#리뷰 수 컬럼에서 쉼표 제거하고 숫자로 변환
def clean_count(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str)
         .str.replace(',', '', regex=False)
         .str.strip()
         .replace({'nan': np.nan, '': np.nan}),
        errors='coerce'
    )

#category에서 두 번째 세부항목 추출
def get_second_category(cat):
    if pd.isna(cat):
        return np.nan
    parts = str(cat).split('|')
    if len(parts) >= 2:
        return parts[1].strip()
    return np.nan

# 원본 Amazon 데이터 불러오기\
raw = pd.read_csv(RAW_PATH, compression="zip")
df = raw.copy()

#숫자형 컬럼 전처리(가격, 할인율, 평점, 리뷰 수를 머신러닝에 사용 가능하도록 숫자형으로 변환)
for col in ['discounted_price', 'actual_price']:
    df[col] = clean_price(df[col])
df['discount_percentage'] = clean_percent(df['discount_percentage'])
df['rating'] = pd.to_numeric(df['rating'].astype(str).str.strip(), errors='coerce')
df['rating_count'] = clean_count(df['rating_count']).fillna(0)

#이상치, 필수 결측치 제거(rating이 비어 있거나 0~5 범위 벗어나면 제거 / product_id, category, 가격, 할인율처럼 꼭 필요한 값이 없으면 제거)
invalid_rating = df['rating'].isna() | (df['rating'] < 0) | (df['rating'] > 5)
invalid_essential = df['product_id'].isna() | df['category'].isna() | df['actual_price'].isna() | df['discounted_price'].isna() | df['discount_percentage'].isna()
removed_invalid = df[invalid_rating | invalid_essential].copy()
df = df[~(invalid_rating | invalid_essential)].copy()

#중복 상품 제거(같은 product_id가 여러 번 있으면 상품 단위 분석에서 중복되므로 첫 번째만 남김)
removed_duplicates = df[df.duplicated(subset=['product_id'], keep='first')].copy()
df = df.drop_duplicates(subset=['product_id'], keep='first').copy()

#main_category 생성(main_category를 14개로 맞추는 과정)
second_category = df['category'].map(get_second_category)
second_counts = second_category.value_counts(dropna=False)
major_categories = sorted([cat for cat, cnt in second_counts.items() if pd.notna(cat) and cnt >= 15])
# keep in frequency order for reporting, not alphabetical
major_categories_by_count = [cat for cat, cnt in second_counts.items() if pd.notna(cat) and cnt >= 15]
allowed_categories = major_categories_by_count + ['Others']

df['main_category'] = second_category.where(second_category.isin(major_categories_by_count), 'Others')

#review_length 생성(리뷰 내용의 글자 수 계산)
df['review_content'] = df['review_content'].fillna('')
df['review_length'] = df['review_content'].astype(str).str.len().astype(int)

#베이지안 보정 평점 생성(리뷰 수가 적은데 평점이 너무 높은 상품 보정 위한 타겟 컬럼)
# 공식: bayesian_rating = (rating_count * rating + m * C) / (rating_count + m)
# C = 전체 평균 평점,  m = 리뷰 수의 중앙값, rating_count = 해당 상품 리뷰 수, rating = 해당 상품 원래 평점
C = float(df['rating'].mean())
m = float(df['rating_count'].median())
df['bayesian_rating'] = ((df['rating_count'] * df['rating']) + (m * C)) / (df['rating_count'] + m)
df['bayesian_rating'] = df['bayesian_rating'].round(2)

#실제 데이터 최종 컬럼 정리
real_final = df[FINAL_COLUMNS].copy()

for col in ['actual_price', 'discounted_price', 'discount_percentage', 'rating', 'rating_count', 'bayesian_rating']:
    real_final[col] = pd.to_numeric(real_final[col], errors='coerce')
real_final['review_length'] = real_final['review_length'].astype(int)

# 더미 데이터 불러오기
dummy = pd.read_csv(DUMMY_PATH)
#더미 데이터 컬럼 맞추기(실제 데이터와 더미 데이터의 컬럼명/컬럼 개수 같도록)
for col in FINAL_COLUMNS:
    if col not in dummy.columns:
        if col == 'main_category':
            dummy[col] = 'Others'
        elif col in ['actual_price', 'discounted_price', 'discount_percentage', 'rating', 'rating_count', 'bayesian_rating', 'review_length']:
            dummy[col] = 0
        else:
            dummy[col] = ''

dummy_final = dummy[FINAL_COLUMNS].copy()

#더미 데이터 main_category 맞추기
dummy_final['main_category'] = dummy_final['main_category'].replace({'Others': 'Others', 'other': 'Others', 'others': 'Others'})
dummy_final.loc[~dummy_final['main_category'].isin(allowed_categories), 'main_category'] = 'Others'


if len(dummy_final) >= len(allowed_categories):
    for i, cat in enumerate(allowed_categories):
        dummy_final.loc[dummy_final.index[i], 'main_category'] = cat

#더미 데이터 숫자형/파생컬럼 정리(더미 데이터도 실제 데이터와 같은 방식으로 숫자형 변환, review_length, bayseian_rating 생성)
for col in ['actual_price', 'discounted_price', 'discount_percentage', 'rating', 'rating_count']:
    dummy_final[col] = pd.to_numeric(dummy_final[col], errors='coerce').fillna(0)
dummy_final['review_content'] = dummy_final['review_content'].fillna('')
dummy_final['review_length'] = dummy_final['review_content'].astype(str).str.len().astype(int)
dummy_final['bayesian_rating'] = ((dummy_final['rating_count'] * dummy_final['rating']) + (m * C)) / (dummy_final['rating_count'] + m)
dummy_final['bayesian_rating'] = dummy_final['bayesian_rating'].round(2)

#결과 파일 저장(csv로)
real_final.to_csv(REAL_OUT, index=False, encoding='utf-8-sig')
real_final.to_csv(REAL_OUT_LEGACY, index=False, encoding='utf-8-sig')
dummy_final.to_csv(DUMMY_OUT, index=False, encoding='utf-8-sig')

removed_invalid['_removed_reason'] = 'invalid_or_missing_essential_value'
removed_duplicates['_removed_reason'] = 'duplicated_product_id'
removed = pd.concat([removed_invalid, removed_duplicates], ignore_index=True, sort=False)
removed.to_csv(REMOVED_OUT, index=False, encoding='utf-8-sig')

#main_category별 상품 수 저장
counts = (
    df['main_category']
      .value_counts()
      .rename_axis('main_category')
      .reset_index(name='product_count')
)

raw_second_counts = (
    second_category.value_counts(dropna=False)
      .rename_axis('second_category_original')
      .reset_index(name='product_count_before_grouping')
)
counts.to_csv(COUNTS_OUT, index=False, encoding='utf-8-sig')

#실제 데이터와 더미 데이터가 잘 맞는지 검증(컬럼명, 컬럼순서, main_category 항목명)
schema_match = list(real_final.columns) == list(dummy_final.columns)
real_cat_set = set(real_final['main_category'].dropna().unique())
dummy_cat_set = set(dummy_final['main_category'].dropna().unique())
category_match = real_cat_set == dummy_cat_set

#검증 리포트
report = []
report.append('[Schema/category alignment report]')
report.append(f'Raw rows: {len(raw):,}')
report.append(f'Real final rows: {len(real_final):,}')
report.append(f'Dummy rows: {len(dummy_final):,}')
report.append(f'Removed invalid/missing rows: {len(removed_invalid):,}')
report.append(f'Removed duplicate product_id rows: {len(removed_duplicates):,}')
report.append('')
report.append(f'Final column count: {len(FINAL_COLUMNS)}')
report.append('Final columns: ' + ', '.join(FINAL_COLUMNS))
report.append(f'Real vs dummy columns match: {schema_match}')
report.append('')
report.append('Rule for main_category: category split by "|" → use the 2nd segment → categories with <15 products are grouped as Others.')
report.append(f'Bayesian formula used for schema-compatible target: (rating_count*rating + m*C)/(rating_count+m), C={C:.4f}, m={m:.1f}')
report.append('')
report.append(f'Real category count: {len(real_cat_set)}')
report.append(f'Dummy category count: {len(dummy_cat_set)}')
report.append(f'Real vs dummy category labels match: {category_match}')
report.append('')
report.append('[Final main_category product counts in real data]')
report.append(counts.to_string(index=False))
report.append('')
report.append('[Original 2nd-level category counts before grouping]')
report.append(raw_second_counts.to_string(index=False))
REPORT_OUT.write_text('\n'.join(report), encoding='utf-8')

#실행 결과 출력
print('\n'.join(report[:25]))
print('\nSaved:')
for p in [REAL_OUT, REAL_OUT_LEGACY, DUMMY_OUT, COUNTS_OUT, REPORT_OUT, REMOVED_OUT]:
    print('-', p, p.stat().st_size)
