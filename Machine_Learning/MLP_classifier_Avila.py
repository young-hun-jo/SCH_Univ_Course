# 0.필요한 라이브러리 import
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.preprocessing import RobustScaler
# 데이터 불균형 해결하기 위해 Loss weight control 계산해주는 메소드
from sklearn.utils.class_weight import compute_class_weight

## 1.문제정의
#해당 데이터는 Bible의 라틴 버전이라고 할 수 있는 'Avila Bible'이라는 Bible의 800개의 이미지로부터 추출된 데이터
#이러한 데이터들을 이용해서 Avila Bible 이미지의 패턴을 분류하는 문제(Classification)
#Labels: A,B,C,D,E,F,G,H,I,W,X,Y (패턴의 종류들)
#Features(F1~F10)은 '상단여백', '하단여백'과 같은 그림을 묘사하는 특징을 숫자화 시킨 특성들

## 2.데이터 로드
# 원본파일이 txt파일로 구분자가 쉼표(,)임
fold_dir = './'
train = pd.read_csv(fold_dir+"avila-tr.txt", sep =',', header=None)
test = pd.read_csv(fold_dir+"avila-ts.txt", sep=',', header=None)
print(f'Train 데이터 Shape : {train.shape}')
print(f'Test 데이터 Shape : {test.shape}')
print(train.head())

# Train, Test 데이터 결합시키기
data = pd.concat([train, test], axis=0)

# Attribute이름을 조건, 반복문으로 변경
for i in range(0, len(data.columns)):
    if i == 10:
        data = data.rename(columns={i:'label'})
    else:
        data = data.rename(columns={i:f'F{i+1}'})
data = data.reset_index(drop=True)
print(data.tail())

## 3.데이터 탐색
# 결측치 확인
print(data.isnull().sum())

# 데이터 행,열 갯수 확인
#20867개의 행과 11개의 열(feature)로 구성
print(data.shape)

# 데이터 타입확인
print(data.dtypes)

# object타입인 label의 기술통계량 확인
print(data.describe(include=object))

#label 개수 : 12개
# label개수를 살펴보고 Class imbalance가 있는지 살펴보기 => imbalance 존재
print(data['label'].value_counts())

# Feature Normalization이 필요한지 아닌지 보기 위해 수치로 이루어진 Feature기술통계량 보기
print(data.describe())
# F2, F3, F7의 값들의 최대값이 매우 큰 걸 보아하니 이상치가 존재하고 이상치에 민감하지 않은 Robustscaler 사용해야 하겠다.

## 4.데이터 전처리
# Label값 LabelEncoding하기, 바꿔서 반환되는 값의 type은 int형이다.
label_encoder = preprocessing.LabelEncoder()
data['label'] = label_encoder.fit_transform(data['label'])
print(data.head())

# 데이터 Label imbalance 해결하기
# Loss weight control 방법을 사용해서 class 개수비율에 따라 weight 계산해주기
# []안에 들어가 있는 값이 label이고 *뒤에있는 숫자는 해당 label 개수
label = [0]*8572 + [1]*10 + [2]*206 + [3]*705 + [4]*2190 + [5]*3923 + [6]*893 + [7]*1039 + [8]*1663 + [9]*89 + [10]*1044 + [11]*533
print(compute_class_weight(class_weight='balanced', classes=np.unique(label), y= label))

#계산된 class별 weight값을 'class_weight'의 dictionary 변수에 넣어주기
class_weight = {0:0.2, 1:173, 2:8.4, 3:2.4, 4:0.7, 5:0.4, 6:1.9, 7:1.6, 8:1, 9:19, 10:1.6, 11:3.2}

# class별 weight값을 feature로 넣어주기 위해 'weight'라는 feature(칼럼) 새로 생성
data['weight'] = ''
for key in class_weight.keys():
    data.loc[data['label'] == key, 'weight'] = class_weight[key]
# weight칼럼 잘 들어갔는지 확인
print(data['weight'].value_counts())

# 모델링할 때 feature 변수 할당 쉽게 해주기 위해 데이터프레임 칼럼명 재배열(label이 마지막 순서로 가도록 해주기)
col_order = [f"F{x}" for x in range(1, 11)] + ['weight','label']
final_data = data[col_order]
print(final_data.columns)

# Features 중에 weight 변수 제외하고 F1~F10 feature들만 Robustscaler 적용해주기
# F1~F10 feature들로만 이루어진 dataframe으로 분해해서 scaler적용해주고 scaler 적용하지 않은 'weight','label'있는 dataframe과 재결합
scaler = preprocessing.RobustScaler()
f_feature = pd.DataFrame(scaler.fit_transform(final_data[[f"F{x}" for x in range(1,11)]]), columns=[f"F{x}" for x in range(1,11)])
final_data = pd.concat([f_feature, final_data[['weight','label']]], axis=1).copy()
print(final_data.head())

## 5.모델링( MLP Classifier 사용)
# 집어넣을 feature 정의
features = []
for i in final_data.columns[:-1]:
    features.append(i)
print(features)

# MLP Classifier 모델링
# 선택이유: label이 11개인 많은편의 다중분류를 해결해야 했음.  Logistic regression을 시도해보았지만 성능이  좋지 않았고 feature개수도 많다 보니 다충퍼셉트론(MLP) Classifier를 사용하기로 결정. 그리고 클래스 불균형을 해소하기 위해 loss weight control을 하기에 MLP가 적합할 것으로 판단함.
kf = KFold(n_splits=10, shuffle=True)
# KFold 10번 수행해 나온 Accuracy담을 빈 리스트 할당
accrs = []
fold_idx = 1

for train_idx, test_idx in kf.split(final_data):
    print(f'Fold 횟수 : {fold_idx}')
    train_d, test_d = final_data.iloc[train_idx], final_data.iloc[test_idx]
    
    train_y = train_d['label']
    train_x = train_d[features]
    
    test_y = test_d['label']
    test_x = test_d[features]
    
    # alpha값으로 L2 normalization 사용해 파라미터 커지는거 제한
    # MLP Classifier 모델 정의
    model = MLPClassifier(hidden_layer_sizes=[512, 256, 64, 16], max_iter=500, alpha=0.0005, random_state=42)
    # 해당 모델에 train 데이터 적합시켜 학습시키기
    model.fit(train_x, train_y)
    # 학습된 모델에 test_x 데이터를 input으로 넣어 예측값 output으로 출력
    pred_y = model.predict(test_x)
    #train 데이터의 accuracy 측정
    train_accr = model.score(train_x, train_y)
    #test 데이터의 accuracy 측정
    test_accr = model.score(test_x, test_y)
    
    print('Train accuracy :', train_accr)
    print('Test accuracy :', test_accr)
    # 1번 fold 끝날때마다 accuracy 담는 list에다가 담기
    accrs.append(test_accr)
    
    fold_idx += 1
    
print()
# 전체 kfold 10번 수행한 후 10번의 accuracy의 평균값 출력
print("전체 평균 Accuracy :", np.average(accrs))
print()
# precision, recall, F1-score값 보기(모두 1에 가까울수록 정확하게 예측한 것임!)
print(classification_report(test_y, pred_y, labels=[x for x in range(0,12)]))





