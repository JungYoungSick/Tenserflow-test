import tensorflow as tf
print("TensorFlow version:", tf.__version__)

# 데이터 세트 로드하기(데이터셋을 불러오고 전처리 과정을 나타냄.)
mnist = tf.keras.datasets.mnist # 훈련용 데이터셋과 테스트용 데이터셋으로 나뉘어져 있다.
(x_train, y_train), (x_test, y_test) = mnist.load_data() # 훈련용 데이터셋과 레이블이다.
x_train, x_test = x_train / 255.0, x_test / 255.0 # 테스트용 데이터셋과 레이블이다.


# 머신러닝 모델 빌드하기.
model= tf.keras.models.Sequential([ # 해당 구문은 여러 레이어를 순서대로 쌓아 올릴 수 있는 Sequential 모델을 생성한다.
  #! 레이어를 쌓는 이유는 무엇일까?
  tf.keras.layers.Flatten(input_shape=(28,28)), # Flatten은 다차원 입력을 평탄화 하여 하나의 긴 벡터로 변환. 입력 데이터 형태를 정의한다.
  tf.keras.layers.Dense(128, activation='relu'), # Dense는 레이어의 완전연결(fully-connected)레이어로 알려져 있으며, 입력과 가중치를 곱하고 바이어스를 더한 후 활성화 함수를 적용한다.
  #? 128개의 뉴런을 가진 완전 연결 레이어다.
  tf.keras.layers.Dropout(0.2),
  # Dropout은 과적합을 방지하기 위한 효과적인 정규화 기법이다.
  tf.keras.layers.Dense(10, activation='softmax') # Dense는 레이어의 완전연결(fully-connected)레이어로 알려져 있으며, 입력과 가중치를 곱하고 바이어스를 더한 후 활성화 함수를 적용한다.
  #? 10개의 뉴런을 가진 완전 연결 레이어다.
])

# Tensorflow의 Keras API를 사용하여 신경망 모델을 컴파일하는 과정.
model.compile(optimizer='adam', 
              # optomizer : 모델 학습 알고리즘을 결정한다. 여기서 'adam'은 최적화 알고리즘을 사용한다는 것을 의미한다.
              loss='sparse_categorical_crossentiropy', # loss : 손실함수(loss function)는 모델이 얼마나 잘 수행되고 있는지를 측정하는 지표이다.(모델의 출력과 실제 레이블 사이의 차이를 계산한다.)
              metrics=['accuracy']) # metrics : 모델의 성능을 평가하기 위한 지표를 나타낸다. 여기서 정확도('accuracy')를 사용하여 모델이 얼마나 정확하게 분류를 수행하는지를 측정한다.

# 모델 예측 수행
# 1. model(x_train[:1]) : 신경망 모델에 x_train의 첫번째 항목(첫번째 이미지)을 입력하여, 예측을 수행하라는 의미이다.
# 2. Numpy를 사용하여 배열로 변환
# 3. 결과 저장 및 출력: 모델에 의해 수행된 예측 결과를 predictions 변수에 저장합니다.
predictions = model(x_train[:1]).numpy()
predictions

# tf.nn은 TensorFlow에서 제공하는 신경망(neural network)관련 함수와 유틸리티를 모아놓은 모듈이다.
tf.nn.softmax(predictions).numpy()
# tf.nn은 신경망을 구성하고 훈련하는데 필요한 다양한 저수준 함수와 연산이 포함되어있고, 활성화 함수, 손실 함수, 합성 곱연산 등이 포함된다.
# 여기서 사용되는 tf.nn.softmax는 softmax함수를 구현한 것이며, 각 요소의 0과 1사이의 값으로 변환하여 모든 요소값의 합이 1이 되도록 한다.

# tf.keras.losses.SparseCategoricalCrossentropy 다중 클래스 분류문제를 위한 손실함수이다.(예측된 확률 분포 사이 차이를 측정하며, 모델의 성능을 개선할 수 있다.)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# 매개변수는 함수에 전달되는 예측값이 로짓 형태인지 나타낸다.
# 실제 레이블과의 크로스 엔트로피를 계산하며, 모델출력이 확률이 아니라 로짓일 경우 이 옵션을 사용해야 한다.
# !로짓이란? : 로지스틱 회귀와 같은 분류 알고리즘에 사용되는 개념으로 확률을 입력으로 받아 자연로그비율을 변환한 값이다.(신경망에서는 강도 또는 신뢰도를 나타내기 위해 로짓을 사용한다.)

# 모델의 예측값과 실제 레이블 사이의 손실을 계산하고 그 결과를 NumPy배열로 변환하는 과정이다.
loss_fn(y_train[:1], predictions).numpy()
# SparseCategoricalCrossentropy는 실제 레이블은 일반적으로 정수 형태로 제공되며, 해당샘플이 속하는 클래스의 인덱스를 나타냅니다.
model.compile(optimizer='adam',
              loss=loss_fn, # 앞서 정의된 손실함수의 인스턴스이다.
              metrics=['accuracy'])


# 모델 훈련 및 평가하기
model.fit(x_train, y_train, epochs=5)# 해당 명령은 모델을 주어진 데이터에 대해 학습시키라는 지시이다. epochs는 해당 데이터를 숫자만큼 반복해서 학습하라는 뜻이다.

model.fit(x_train, y_train, epochs=5)
# !두번 쓰인 이유는 무엇일까? 실수로 그런 것은 아닌 것 같고, 추가 훈련 또는 실험의 목적같은데..

model.evaluate(x_test, y_test, verbose=2) # 
# 훈련된 모델을 평가 데이터셋에 대해 평가하라는 지시이다.
# verbose 설정은 에포크마다 한줄씩 결과를 출력하도록 설정하는 것으로 테스트 샘플에 대한 손실 및 평가지표가 출력된다.

probability_model = tf.keras.Sequential([ # 식별자에 순차적으로 실행하는 Sequential을 쌓아서 생성하는 클래스를 담는다.
  model, # model은 이미 정의되어 있는 신경망 모델을 나타낸다.
  # 첫번째 레이어로 추가함으로써 이전에 훈련된 모델의 구조와 가중치를 가져온다
  tf.keras.layers.Softmax()
  # Softmax 활성화 레이어를 나타낸다.
  # Softmax 함수는 로짓(소프트맥스 함수 적용 전의 모델 출력)배열을 입력으로 받아, 각 클래스에 대한 예측 확률 분포를 출력한다.
])

probability_model(x_test[:5]) # x_test는 테스트 데이터셋을 나타내며, x_test[:5]는 이 데이터셋의 첫 5개 샘플을 선택하는 부분이다.
#  명명된 모델로 테스트 데이터셋의 첫 5개 샘플에 대해 예측을 수행하는 과정을 나타낸다.