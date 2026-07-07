"""
Step4. 피처 엔지니어링
- 입력: 팀원1의 Step1 전처리 결과 (타겟 bayesian_rating 없는 버전, product_id 포함)
- 출력: 모델에 바로 넣을 수 있는 최종 피처 테이블 (X만 있음, y는 나중에 merge)
- Step3(타겟 생성)와 무관하게 지금 바로 진행 가능한 작업
"""

import pandas as pd
import numpy as np

DATA_PATH = 'data\\processed\\amazon_preprocessed_no_bayesian_other.csv'  # step1 실제 산출물


def load_data(path: str) -> pd.DataFrame:
    """Step1 산출물 로드"""
    df = pd.read_csv(path)
    return df


def check_skewness(df: pd.DataFrame, columns: list, threshold: float = 1.0) -> dict:
    """
    각 컬럼의 왜도(skewness)를 계산하고, 임계값을 넘는지 확인
    - 왜도 절댓값이 threshold(기본 1.0)를 넘으면 "치우친 분포"로 판단해 로그 변환 대상으로 표시
    - 이 기준(1.0)은 통계학에서 흔히 쓰는 경험적 기준(rule of thumb)이지 절대적 법칙은 아님
    - 반환값: {컬럼명: 로그변환필요여부(True/False)}
    """
    decisions = {}
    print("=== 왜도(skewness) 확인 ===")
    for col in columns:
        skew = df[col].skew()
        need_log = abs(skew) > threshold
        decisions[col] = need_log
        flag = "-> 로그 변환 O" if need_log else "-> 로그 변환 X"
        print(f"{col}: skew={skew:.2f} {flag}")
    return decisions


def log_transform_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    왜도 확인 결과를 바탕으로 치우친 컬럼에 자동으로 로그 변환 적용
    - 어떤 컬럼이 변환됐는지는 check_skewness()의 출력으로 확인 가능
    """
    df = df.copy()
    candidate_cols = ['discounted_price', 'actual_price', 'discount_percentage', 'rating_count']
    decisions = check_skewness(df, candidate_cols)

    for col, need_log in decisions.items():
        if need_log:
            df[f'log_{col}'] = np.log1p(df[col])
    return df


def check_missing(df: pd.DataFrame, required_cols: list) -> pd.DataFrame:
    """
    Step1(팀원1)에서 결측치 처리가 이미 끝났다고 가정하고, 여기서는 제거하지 않음
    - Step4의 로그 변환/파생 변수 생성 과정에서 새로 생긴 결측치가 없는지 최종 확인만 함
    - 결측치가 남아있으면 경고만 출력 -> 원인(Step1 누락인지, Step4 로직 문제인지) 파악 후
      Step1으로 돌아가서 고치는 게 원칙 (Step4에서 임의로 제거/대체하지 않음)
    """
    missing_counts = df[required_cols].isna().sum()
    if missing_counts.sum() > 0:
        print("경고: 다음 컬럼에 결측치가 남아있습니다 (Step1 처리 결과 재확인 필요):")
        print(missing_counts[missing_counts > 0])
    else:
        print("결측치 확인 완료: 이상 없음")
    return df


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step5에 넘길 최종 피처 테이블 구성
    - log_transform_features()가 왜도 기준으로 동적으로 변환하므로,
      각 후보 컬럼에 대해 'log_' 버전이 있으면 그걸, 없으면 원본을 사용
    [확인 필요 : product_id를 반드시 포함해서, 나중에 팀원1의 bayesian_rating과
        merge할 수 있게 해야 함 (키 컬럼이 없으면 나중에 합칠 방법이 없음)]
    """
    price_count_cols = ['discounted_price', 'actual_price', 'discount_percentage', 'rating_count']
    resolved_cols = [f'log_{c}' if f'log_{c}' in df.columns else c for c in price_count_cols]

    keep_cols = ['product_id'] + resolved_cols + ['review_length', 'main_category']
    return df[keep_cols]


def main():
    df = load_data(DATA_PATH)
    df = log_transform_features(df)

    price_count_cols = ['discounted_price', 'actual_price', 'discount_percentage', 'rating_count']
    required_cols = [f'log_{c}' if f'log_{c}' in df.columns else c for c in price_count_cols]
    required_cols += ['main_category']
    df = check_missing(df, required_cols)

    feature_table = build_feature_table(df)

    output_path = 'data\\processed\\amazon_preprocessing_final.csv'
    feature_table.to_csv(output_path, index=False)
    print(f"\n최종 피처 테이블 저장 완료: {output_path}")
    print(f"행 개수: {len(feature_table)}, 컬럼: {feature_table.columns.tolist()}")
    print(f"\n{feature_table.head()}")

    return feature_table


if __name__ == '__main__':
    main()
