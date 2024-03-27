# Todo: TensorFlow and tf.keras
import tensorflow as tf

# Todo: Helper libraries
import numpy as np
import matplotlib.pyplot as plt #pip install matplotlib 설치필요

print(tf.__version__) # todo: Tensorflow version 확인

# Todo: 패션 MNIST 데이터셋 임포트하기
fashion_mnist = tf.keras.datasets.fashion_mnist #? keras 모듈에서 Fashin MNIST 데이터셋을 불러오기 위한 메서드 작성
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data() #? fashion_mnist 객체의 load_data() 메소드를 호출하여 데이터셋을 로드한다.
#? 데이터셋을 로드하여 훈련 이미지, 훈련 레이블, 테스트 이미지, 테스트 레이블로 분할하여 각각 변수에 할당하는 과정임.

class_names = ['T-shirt/top', 'Trouser','Pullover', 'Dress','Coat','Sandal','Shirt','Sneaker','bag','Ankle boot']
#? 의류에 대한 이름을 문자열로 저장.

# Todo: 데이터 탐색
train_images.shape #? 훈련 데이터셋(train_images)의 형태(shape)를 확인하는 부분이다.
# 넘파이 배열(ndarray)의 속성으로, 이미지 데이터의 형태를 확인한다.
len(train_labels) #? 훈련 데이터셋(train_labels)의 길이(langth)를 확인하는 부분이다.
# 훈련 이미지에 대한 라벨의 수를 나타낸다. 훈련 이미지와 훈련 라벨은 일대일 대응이 되어야 한다.
train_labels #? 훈련 라벨(train_labels)을 출력하는 부분이다.
# 각 이미지에 대한 라벨을 보여준다.
test_images.shape #? 테스트 데이터셋(test_images)의 형태를 확인하는 부분이다.
# 테스트 데이터셋은 모델의 성능을 평가하기 위해 사용된다.
len(test_labels) #? 테스트 데이터셋(test_labels)의 길이(length)를 확인하는 부분이다.
# 테스트 이미지에 대한 라벨의 수를 나타낸다. 테스트 이미지와 테스트 라벨도 일대일 대응이 되어야 한다.

# Todo: 데이터 전처리 
plt.figure() #? 새로운 figure(그림)을 생성합니다. 
# Figure는 그림을 담는 하나의 컨테이너 역할을 한다.
plt.imshow(train_images[0]) #? imshow 함수(이미지 시각화)를 사용하여 첫번째 훈련 이미지를 시각화한다.
plt.colorbar() #? 이미지 컬러바를 추가한다. 
# 컬러바는 픽셀값과 해당하는 색상을 보여준다.
plt.grid(False) #? 그리드를 숨긴다. 
# 이미지 주위에 그리드 라인을 표시하지 않도록 설정한다.
plt.show() #? 그림을 화면에 표시한다.

train_images = train_images / 255.0 # 훈련 데이터셋(train_images)의 모든 픽셀 값을 255로 나눠준다.
test_images = test_images / 255.0 # 테스트 데이터셋(test_images)의 모든 픽셀 값을 255로 나눠준다.

plt.figure(figsize=(10,10)) #? figure(그림)을 생성하며, 그 크기를 (10,10)으로 설정한다.
for i in range(25): #? 0부터 24까지의 숫자를 반복한다.
  # 처음 25개의 이미지를 시각화하기 위한 반복문이다.
  plt.subplot(5,5,i+1) #? 5x5 그리드 형태의 서브플롯(subplot)을 생성한다.(i+1은 subplot)의 위치를 결정한다.
  plt.xticks([]) #? x축의 눈금을 제거한다.
  plt.yticks([]) #? x축의 눈금을 제거한다.
  plt.grid(False) #? 그리드를 숨긴다.
  plt.imshow(train_images[i], cmap=plt.cm.binary) #? imshow 함수를 사용하여 현재 순회중인 훈련 이미지(train_images[i]를 시각화 한다.
  # cmap = plt.cm.binary는 이미지를 흑백으로 표시한다.
  plt.xlabel(class_names[train_labels[i]]) #? x축에 해당 이미지의 라벨을 나타내는 부분이다.
  # train_labels[i]는 현재 이미지의 라벨을 나타낸다.
plt.show() #? 모든 서브플롯을 그림으로 보여준다.

# Todo: 모델구성(층 설정)
model = tf.keras.Sequential([ #? Sequential 모델을 생성한다.
  # Sequential 모델은 레이어를 선형으로 쌓은 신경망 구조를 나타낸다.
  tf.keras.layers.Flatten(input_shape=(28,28)), #? 입력 이미지를 28x28 픽셀의 2차원 배열로 받아들이는 입력 레이어이다.
  # Flatten 레이어는 다차원 배열을 1차원으로 평탄화(flatten)한다.
  # 레이어는 첫번째 레이어로 사용되며, 입력 이미지를 28x28의 2D 배열에서 784크기의 1D배열로 변환한다.
  tf.keras.layers.Dense(128,activation='relu'), #? 128개의 뉴런을 가진 filly connected(Dense)레이어를 추가한다.
  # ReLU(Rectified Linear Unit)를 사용한다. 입력이 0보다 작을 때 0으로 만들고, 그렇지 않으면 입력 값을 그대로 출력한다.
  tf.keras.layers.Dense(10) #? 10개의 뉴런을 가진 fully connected(Dense)레이어를 추가한다.
  # 이 레이어는 출력 레이어로 사용되며, 순수한 선형 레이어로 사용될 것을 의미한다.
])

# Todo: 모델 컴파일
model.compile(optimizer='adam', #? compile 메서드를 사용하여 모델을 컴파일하며, 옵티마이저로 adam을 선택한다..
              # 컴파일에 필요한 세가지 요소를 지정한다.
              loss=tf.keras.losses.SparseCategoricalCrossentropy
              #? 손실함수로 Sparse Categorical Crossentropy를 사용한다.
              # SparseCategoricalCrossentropy 함수는 TensorFlow의 Keras API에서 제공되는 손실 함수 중 하나입니다.
              (from_logits=True), # from_logits 매개변수를 True로 설정하면 모델의 출력이 확률 분포가 아니라 로짓으로 주어진다고 가정한다.
              metrics=['accuracy']) #? 모델을 평가할 때 사용할 평가 지표를 지정한다.
              # 여기서 정확도(accuracy)를 사용한다.

# Todo: 모델 훈련
model.fit(train_images, train_labels, epochs=10) #? Fit 메서드는 주어진 데이터를 사용하여 모델을 여러 에포크(epoch)동안 학습시킨다.
# train_images : 훈련 이미지 데이터셋이다.
# Train_labels : 훈련 이미지에 대한 라벨(정답) 데이터셋이다.
# epochs=10 : 학습할 에포크 수를 지정한다.(에포크는 전체 훈련 데이터셋에 대한 단일순회를 말한다)

# Todo: 정확도 평가
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2) #? evaluate 메서드를 사용하여 모델을 테스트 데이터셋으로 평가한다.
# evaluate 메서드의 반환값으로 테스트 데이터셋에 대한 손실과 정확도를 나타낸다.
# test_acc는 테스트 데이터셋에 대한 정확도를 나타낸다.
print('\nTest accuracy :', test_acc) #? 평가 결과인 테스트 정확도를 출력한다.

# Todo: 예측하기
probability_model = tf.keras.Sequential([model,
                                        tf.keras.layers.Softmax()]) #? 기존 학습된 모델인 model에 소프트맥스(Softmax)레이어를 추가하여 확률 모델을 생성한다.
predictions = probability_model.predict(test_images) #? 테스트 이미지 데이터셋에 대한 예측을 수행한다.
# 생성된 확률 모델을 사용하여 테스트 이미지에 대한 각 클래스에 속할 확률을 예측하는 것을 의미한다.
#predict메서드는 입력 데이터에 대한 모델의 출력을 계산한다.

predictions[0] #? 첫번째 테스트 이미지에 대한 예측 결과를 출력한다.
# 각 클래스에 속할 확률을 나타내는 배열이다.

np.argmax(predictions[0]) #? 첫 번째 테스트 이미지에 대한 예측 결과 중 가장 높은 확률을 가지는 클래스의 인덱스를 출력한다.
# argmax 함수는 배열에서 가장 큰 값의 인덱스를 반환한다.

test_labels[0] #? 첫번째 테스트 이미지에 대한 실제 라벨을 출력한다.
# 예측한 결과와 비교하여 모델의 성능을 평가하는 데 사용된다.

# Todo: 그래프를 통한 표현
def plot_image(i, predictions_array, true_label, img):
  #? plot_image 라는 함수를 정의한다.
  true_label, img =true_label[i], img[i]
  #? 인덱스 i에 해당하는 이미지와 해당 이미지에 대한 예측 결과를 시각화 하는데 사용된다.
  # i : 시각화 할 이미지의 인덱스
  # predictions_array : 예측 결과에 대한 배열
  # true_lavbel : 해당 이미지의 실제 라벨
  # img : 시각화할 이미지 데이터
  plt.xticks([]) #? X축의 눈금을 제거한다.
  plt.yticks([]) #? y축의 눈금을 제거한다.
  plt.imshow(img,cmap=plt.cm.binary) #? 선택한 이미지를 시각화한다.
  # img는 이미지 데이터이다.
  # Cmap=plt.cm.binary는 이미지를 흑백으로 표시한다.

  predicted_label = np.argmax(predictions_array) #? 배열에서 가장 높은 확률을 가지는 클래스의 인덱스를 찾는다. 
  # np.argmax()함수는 배열에서 가장 큰 값의 인덱스를 반환한다.
  # 해당 이미지를 예측한 클래스를 나타낸다.
  if predicted_label == true_label: #? 예측한 클래스와 실제 클래스가 일치하는지 확인한다.
    color = 'blue' #? 예측 결과가 일치하면 파란색으로 표시
  else:
    color='red' #? 예측 결과가 일치하지 않는다면 빨간색으로 표시

  plt.xlabel("{}{:2.of}%({})".format(class_names[predicted_label],
  #? x축 레이블에 예측 결과를 표시한다. format()메서드를 사용하여 문자열을 포멧팅한다.
                              100*np.max(predictions_array),
                              class_names[true_label]), #? 예측 확률을 소수점 두자리까지 표시된다.
                              color=color) #? 앞서 결정한 파란색과 빨간색 중 하나를 결정.
  
def plot_value_array(i, predictions_array, true_label):
  #? plot_value_array라는 함수를 정의한다.
  # i : 시각화할 이미지의 인덱스
  # predictions_array : 예측 확률 배열
  # true_label : 해당 이미지의 실제 라벨
  true_label = true_label[i] #? 함수 내에서 사용할 i번째 실제 라벨을 선택한다.
  plt.grid(False)
  plt.xticks(range(10)) #? x축의 눈금을 설정한다.
  plt.yticks([]) #? y축의 눈금을 설정한다.
  thisplot = plt.bar(range(10), predictions_array, color="#777777") #? 막대 그래프를 생성한다.
  # predictions_array는 각 클래스에 대한 예측확률을 나타낸다.
  plt.ylim([0,1]) #? y축의 범위를 설정한다.
  predicted_label = np.argmax(predictions_array) #? 예측 확률 배열에서 가장 높은 확률을 가지는 클래스의 인덱스를 찾습니다.

  thisplot[predicted_label].set_color('red') #? 예측 결과에 해당하는 막대의 색상을 표시
  thisplot[true_label].set_color('blue') #? 예측 결과에 해당하는 막대의 색상을 표시

# Todo: 예측 확인
  #? 해당 코드는 두개의 테스트 이미지에 대한 시각화를 생성하는 과정을 나타낸다.
  i=0
  plt.figure(figsize=(6,3)) #? 새로운 시각화를 생성한다.(가로,세로 길이)
  plt.subplot(1,2,1) #? 그림을 1x2 그리드로 분할하고, 첫번째 위치 서브플릇을 생성한다.
  plot_image(i, predictions[i], test_labels, test_images) #? 함수를 호출하여 이미지에 대한 시각화를 생성한다.
  plt.subplot(1,2,2) #? 그림을 1x2 그리드로 분할하고, 두번째 위치 서브플릇을 생성한다.
  plot_value_array(i, predictions[i], test_labels) #? 함수를 호출하여 선택한 이미지에 대한 예측 결과의 확률 분포를 시각화 한다.
  plt.show() #? 생성한 시각화를 표시한다.

  i=12
  plt.figure(figsize=(6,3))
  plt.subplot(1,2,1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(1,2,2)
  plot_value_array(i,predictions[i], test_labels)
  plt.show()

#Todo: Plot the first X test images, their predicted labels, and the true labels.
#Todo: Color correct predict predictions in blue and incorrect predictions in red.
  num_rows = 5 #? 행의 개수 설정
  num_cols = 3 #? 열의 개수 설정
  num_images = num_rows*num_cols #? 전체 이미지의 개수를 계산
  plt.figure(figsize=(2*2*num_cols, 2*num_rows)) #? 새로운 시각화를 생성
  for i in range(num_images): #? 전체 이미지 개수만큼 반복하는 루프를 실행.
    plt.subplot(num_rows, 2*num_cols,2*i+1) 
    plot_image(i, predictions[i], test_labels, test_images) #? 그림을 rows로 분할
    plt.subplot(num_rows, 2*num_cols, 2*i+2) 
    plot_value_array(i, predictions[i], test_labels) #? 서브플롯은 예측 결과에 대한 확률 분포를 표시
  plt.tight_layout() #? 서브 플릇 간의 간격을 조정하여 레이아웃을 더욱 조밀하게 만든다.
  plt.show() #? 생성한 시각화를 표시한다.

#Todo: 훈련된 모델 사용하기
#Todo: Grab an image from the test dataset.
  img = test_images[1] #? 테스트 데이터셋에서 첫번째 이미지를 선택하여 Img변수에 할당.
  print(img.shape) #? 선택한 이미지 형태(shape)를 출력한다.

#Todo: Add the image to a batch where it's the only member.
  img = (np.expand_dims(img,0)) #? 선택한 이미지를 배치(batch)에 추가한다.
  print(img.shape) #? 함수를 사용하여 이미지의 차원을 확장하여 배치에 추가.

  predictions_single = probability_model.predict(img) #? 모델을 사용하여 단일 이미지에 대한 예측을 수행한다.
  print(predictions_single) #? 단일 이미지에 대한 예측 결과를 출력한다.

  plot_value_array(1, predictions_single[0], test_labels) #? 함수를 사용하여 단일 이미지에 대한 예측 결과의 확률 분포를 시각화한다.
  _ = plt.xticks(range(10), class_names, rotation=45) #? x축의 눈금을 설정한다.
  # 숫자를 클래스로 대체하고 텍스트를 45도로 회전한다.
  plt.show() #? 생성한 시각화를 표시한다.


  np.argmax(predictions_single[0]) #? 단일 이미지에 대한 예측 결과에서 가장 높은 확률을 가지는 클래스의 인덱스를 찾는다.

  #! 해당 러닝 중 중간 이미지와 결과값이 출력이 되지 않는 문제가 있다. 해당 문제에 대해 해결해야 한다.