"""
Step6. 모델 해석 (SHAP)
- Step5(step5_model_pipeline.py)에서 저장한 best_model.pkl, train_test_split.pkl을 불러와
  "어떤 피처가 예측에 얼마나 영향을 미치는지" 분석
- 지금은 더미 데이터로 학습된 모델이라 SHAP 결과 자체는 의미 없음 (코드가 에러 없이 도는지만 확인)
- 실제 데이터로 step5를 재실행해서 pkl이 갱신되면, 이 파일은 수정 없이 그대로 재실행하면 됨
"""

import math
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt

# ── Step5에서 저장한 파일 불러오기 ──
pipeline = joblib.load(r'result\modeling\best_model.pkl')          # 전처리 + 모델이 합쳐진 sklearn Pipeline
X_train, X_test, y_train, y_test = joblib.load(r'result\modeling\train_test_split.pkl')

# ── Pipeline에서 전처리 단계와 모델 단계를 분리 ──
# SHAP은 "전처리가 끝난 숫자 데이터"와 "모델"을 따로 필요로 하기 때문
preprocessor = pipeline.named_steps['preprocess']
model = pipeline.named_steps['model']

# ── 원핫인코딩 등 전처리를 실제로 적용 (사람이 읽는 컬럼 -> 모델이 보는 숫자 배열) ──
X_train_transformed = preprocessor.transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# ── 전처리 후 컬럼 이름 가져오기 (원핫인코딩으로 늘어난 컬럼들 포함) ──
# 예: main_category 1개 컬럼 -> Accessories&Peripherals, Kitchen&HomeAppliances... 14개 컬럼으로 분리됨
feature_names = preprocessor.get_feature_names_out()

# ── 원핫인코딩 결과가 희소 행렬(sparse matrix)로 나올 수 있어 dense 배열로 변환 ──
# (SHAP 일부 시각화 함수가 희소 행렬을 지원하지 않기 때문)
if hasattr(X_train_transformed, 'toarray'):
    X_train_transformed = X_train_transformed.toarray()
if hasattr(X_test_transformed, 'toarray'):
    X_test_transformed = X_test_transformed.toarray()

# ── SHAP Explainer 생성 ──
# shap.Explainer는 모델 종류(선형/트리 등)를 자동으로 감지해서 알맞은 계산 방식을 선택함
explainer = shap.Explainer(model, X_train_transformed, feature_names=feature_names)

# ── 테스트셋에 대해 SHAP value 계산 ──
# SHAP value: 각 피처가 "이 예측값을 평균 예측값에서 얼마나 밀어올렸는지/내렸는지"를 나타내는 숫자
shap_values = explainer(X_test_transformed)

# ── 1. Summary Plot: 전체적으로 어떤 피처가 중요한지 한눈에 보기 ──
plt.figure()
shap.summary_plot(shap_values, X_test_transformed, feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig(r'result\shap\shap_summary_plot.png', dpi=150)
plt.close()
print("shap_summary_plot.png 저장 완료")

# ── 2. 피처 중요도 순위를 텍스트로도 출력 (그림 없이 빠르게 확인용) ──
mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
importance_ranking = sorted(zip(feature_names, mean_abs_shap), key=lambda x: -x[1])

print("\n=== 피처 중요도 순위 (SHAP 절댓값 평균 기준) ===")
for rank, (name, value) in enumerate(importance_ranking, start=1):
    print(f"{rank}. {name}: {value:.4f}")

# ── 3. Dependence Plot: 모든 피처를 한 번에, 하나의 큰 figure에 서브플롯으로 배치 ──
# - 중요도 순으로 정렬된 순서 그대로 왼쪽 위 -> 오른쪽 아래로 배치
# - 피처 개수에 따라 격자(grid) 크기를 자동으로 계산
n_features = len(feature_names)
n_cols = 4
n_rows = math.ceil(n_features / n_cols)
 
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.5, n_rows * 3.8))
axes = np.array(axes).reshape(-1)  # 1차원으로 펼쳐서 인덱스로 접근하기 쉽게 함
 
for i, (name, _) in enumerate(importance_ranking):
    shap.dependence_plot(
        name, shap_values.values, X_test_transformed,
        feature_names=feature_names,
        ax=axes[i], show=False,
    )
    #axes[i].set_title(name, fontsize=9)
 
# 피처 개수가 격자 칸 수보다 적을 때, 남는 빈 칸은 숨김 처리
for j in range(n_features, len(axes)):
    axes[j].axis('off')

for ax in fig.axes:
    ax.tick_params(labelsize=8)          # 눈금 숫자 크기
    ax.xaxis.label.set_size(9)           # x축 라벨 크기
    ax.yaxis.label.set_size(9)           # y축 라벨 크기
 
plt.tight_layout()
plt.savefig(r'result\shap\shap_dependence_grid.png', dpi=150)
plt.close()
print(f"\nshap_dependence_grid.png 저장 완료 (피처 {n_features}개, {n_rows}x{n_cols} 격자)")