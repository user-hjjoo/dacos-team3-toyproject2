"""
모델링용 최종 데이터셋 생성
- amazon_feature_engineering.csv (Step4 산출물): 모델에 넣을 피처(X)
- amazon_final_processed.csv (팀원1 최종본): 타겟(y) bayesian_rating
- product_id를 키로 병합해서 Step5(model_pipeline)에 바로 넣을 수 있는 파일 생성
"""

import pandas as pd

feature_df = pd.read_csv(r'data\processed\amazon_feature_engineering.csv')   # X (Step4 산출물)
target_df = pd.read_csv(r'data\processed\amazon_final_processed.csv')        # bayesian_rating 포함 (팀원1 최종본)

# ── 타겟 컬럼만 추출 (필요한 것만 가져와서 중복 컬럼 충돌 방지) ──
target_only = target_df[['product_id', 'bayesian_rating']]

# ── product_id 기준으로 병합 ──
# how='inner': 양쪽에 다 있는 상품만 사용 (지금은 두 파일의 product_id가 완전히 일치하므로 안전)
merged = feature_df.merge(target_only, on='product_id', how='inner')

# ── 병합 검증 ──
print(f"feature_engineering 행 개수: {len(feature_df)}")
print(f"final_processed 행 개수: {len(target_df)}")
print(f"병합 후 행 개수: {len(merged)}")

if len(merged) != len(feature_df):
    print("⚠️ 경고: 병합 후 행 개수가 줄었습니다. product_id 불일치가 있는지 확인하세요.")
else:
    print("병합 검증 통과: 행 개수 손실 없음")

if merged['bayesian_rating'].isna().sum() > 0:
    print(f"⚠️ 경고: bayesian_rating에 결측치 {merged['bayesian_rating'].isna().sum()}건 발생")
else:
    print("타겟(bayesian_rating) 결측치 없음")

output_path = 'data\\processed\\amazon_model_ready.csv'
merged.to_csv(output_path, index=False)
print(f"\n최종 모델링용 데이터셋 저장 완료: {output_path}")
print(f"컬럼: {merged.columns.tolist()}")
print(merged.head())