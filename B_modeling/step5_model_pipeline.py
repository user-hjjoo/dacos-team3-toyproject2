"""
Step5. 모델링 파이프라인
- Step4(피처엔지니어링)와 완전히 분리된 버전
- 이 파일은 "이미 최종 피처 + 타겟이 준비된 테이블"을 받아서 학습/평가만 담당
- 피처를 어떻게 만들지(로그 변환, 파생 변수 등)는 이 파일의 관심사가 아님

사용 방법:
1. 지금은 더미 데이터의 원본 컬럼을 "완성된 피처"인 셈 치고 파이프라인 동작만 검증
2. 나중에 Step4에서 실제 최종 피처 테이블(전처리_피처.csv)이 나오면:
   - DATA_PATH만 교체
   - NUMERIC_FEATURES / CATEGORICAL_FEATURES 리스트를 그 테이블의 실제 컬럼명으로 교체
   - 나머지 코드는 그대로 작동해야 함
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

# ── 나중에 Step4 최종 산출물 경로로 교체 ──
DATA_PATH = 'data\\processed\\amazon_model_ready.csv'

# ── 나중에 Step4 최종 피처 테이블의 실제 컬럼명으로 교체 ──
# 지금은 더미 데이터의 원본 컬럼을 그대로 사용 (가공 없이, 파이프라인 검증용)
NUMERIC_FEATURES = [
    'log_discounted_price', 'log_actual_price', 'discount_percentage', 'log_rating_count', 'log_review_length',
]
CATEGORICAL_FEATURES = ['main_category']
TARGET = 'bayesian_rating'


def load_data(path: str) -> pd.DataFrame:
    """
    최종 피처 테이블 로드
    - 이 함수는 데이터가 이미 모델에 넣을 수 있는 형태로 준비되어 있다고 가정
    - 최소한의 결측치 방어만 수행 (타겟/피처 결측 행 제거)
    """
    df = pd.read_csv(path)
    required_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
    df = df.dropna(subset=required_cols)
    return df


def build_pipeline(model):
    """
    전처리(원핫인코딩) + 모델을 하나의 sklearn Pipeline으로 묶기
    - 수치형 피처(NUMERIC_FEATURES): 변환 없이 그대로 사용 (passthrough)
    - 범주형 피처(CATEGORICAL_FEATURES): 원핫인코딩으로 0/1 컬럼들로 변환
    - handle_unknown='ignore': 나중에 학습 때 못 본 새 카테고리 값이 들어와도 에러 없이 처리
    - 어떤 모델(Linear/RF/XGBoost)이 들어오든 동일한 전처리를 재사용할 수 있게 함수화
    """
    preprocessor = ColumnTransformer(transformers=[
        ('num', 'passthrough', NUMERIC_FEATURES),
        ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES),
    ])
    return Pipeline(steps=[('preprocess', preprocessor), ('model', model)])


def evaluate(name, pipeline, X_test, y_test):
    """
    학습된 모델을 테스트셋으로 평가
    - RMSE, MAE: 예측 오차 (낮을수록 좋음)
    - R2: 모델의 설명력, 1에 가까울수록 좋음 (0=평균으로 찍는 것과 동일, 음수=평균보다 못함)
    - 결과를 dict로 반환 (여러 모델 결과를 모아 비교표 만들기 위함)
    """
    preds = pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    return {'model': name, 'RMSE': round(rmse, 4), 'MAE': round(mae, 4), 'R2': round(r2, 4)}


def main():
    """
    전체 실행 흐름
    1. 데이터 로드
    2. 피처(X)/타겟(y) 분리
    3. train/test 분할 (8:2)
    4. Linear -> Random Forest -> XGBoost 순서로 각각 학습
    5. 세 모델 성능을 표로 비교 출력
    6. 학습된 모델들과 데이터 분할 결과를 반환 (다음 단계인 SHAP 해석에서 재사용하기 위함)
    """
    # 1. 데이터 로드
    df = load_data(DATA_PATH).set_index('product_id')

    # 2. 피처(X)/타겟(y) 분리
    # product_id를 완전히 버리지 않고 인덱스로 유지 -> 나중에 SHAP 해석 시
    # "이 행이 어떤 상품이었는지" 되짚을 수 있게 함
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    # 3. train/test 분할 (8:2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4-1. 비교할 모델 3종 정의 (쉬운 모델 -> 복잡한 모델 순서)
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=200, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=200, random_state=42, verbosity=0),
    }

    # 4-2. 모델별로 파이프라인 생성 -> 학습 -> 평가를 반복
    results = []
    trained_pipelines = {}
    for name, model in models.items():
        pipeline = build_pipeline(model)
        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline
        results.append(evaluate(name, pipeline, X_test, y_test))

    # 5. 세 모델 성능을 표로 비교 출력 (RMSE 낮은 순으로 정렬)
    results_df = pd.DataFrame(results).sort_values('RMSE')
    print("=== 모델 성능 비교 ===")
    print(results_df.to_string(index=False))

    # 6. 학습된 모델들과 데이터 분할 결과를 반환 (다음 단계인 SHAP 해석에서 재사용하기 위함)
    #    - 성능이 가장 좋은(RMSE 최소) 모델과 train/test 데이터를 파일로 저장
    #    - step6.py에서는 재학습 없이 joblib.load()로 바로 불러와서 사용
    best_model_name = results_df.iloc[0]['model']
    joblib.dump(trained_pipelines[best_model_name], 'result\\modeling\\best_model.pkl')
    joblib.dump((X_train, X_test, y_train, y_test), 'result\\modeling\\train_test_split.pkl')
    
    return trained_pipelines, results_df, X_train, X_test, y_train, y_test


if __name__ == '__main__':
    main()
