# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt #pip install matplotlib 설치필요

print(tf.__version__) # todo: Tensorflow version 확인

#? 패션 MNIST 데이터셋 임포트하기
fashion_mnist = tf.keras.datasets.fashion_mnist # keras 모듈에서 Fashin MNIST 데이터셋을 불러오기 위한 메서드 작성
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data() # fashion_mnist 객체의 load_data() 메소드를 호출하여 데이터셋을 로드한다.
# Todo: 데이터셋을 로드하여 훈련 이미지, 훈련 레이블, 테스트 이미지, 테스트 레이블로 분할하여 각각 변수에 할당하는 과정임.

class_names = ['T-shirt/top', 'Trouser','Pullover', 'Dress','Coat','Sandal','Shirt','Sneaker','bag','Ankle boot']
# Todo: 의류에 대한 이름을 문자열로 저장.

#? 데이터 탐색
train_images.shape

# len(train_labels)

# train_labels

# test_images.shape

# len(test_labels)

# # 데이터 전처리
# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

# train_images = train_images / 255.0
# test_images = test_images / 255.0

# plt.figure(figsize=(10,10))
# for i in range(25):
#   plt.subplot(5,5,i+1)
#   plt.xticks([])
#   plt.yticks([])
#   plt.grid(False)
#   plt.imshow(train_images[i], cmap=plt.cm.binary)
#   plt.xlabel(class_names[train_labels[i]])
# plt.show()

# # 모델구성
# # 층 설정
# model = tf.keras.Sequential([
#   tf.keras.layers.Flatten(input_shape=(28,28)),
#   tf.keras.layers.Dense(128,activation='relu'),
#   tf.keras.layers.Dense(10)
# ])

# # 모델 컴파일
# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])

# # 모델 훈련
# model.fit(train_images, train_labels, epochs=10)

# # 정확도 평가
# test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
# print('\nTest accuracy :', test_acc)

# # 예측하기
# probability_model = tf.keras.Sequential([model,
#                                         tf.keras.layers.Softmax()])
# predictions = probability_model.predict(test_images)

# predictions[0]

# np.argmax(predictions[0])

# test_labels[0]

# # 그래프를 통한 표현
# def plot_image(i, predictions_array, true_label, img):
#   true_label, img =true_label[i], img[i]
#   plt.xticks([])
#   plt.yticks([])

#   plt.imshow(img,cmap=plt.cm.binary)

#   predicted_label = np.argmax(predictions_array)
#   if predicted_label == true_label:
#     color = 'blue'
#   else:
#     color='red'

#   plt.xlabel("{}{:2.of}%({})".format(class_names[predicted_label],
#                               100*np.max(predictions_array),
#                               class_names[true_label]),
#                               color=color)
  
# def plot_value_array(i, predictions_array, true_label):
#   true_label = true_label[i]
#   plt.xticks(range(10))
#   plt.yticks([])
#   thisplot = plt.bar(range(10), predictions_array, color="#777777")
#   plt.ylim([0,1])
#   predicted_label = np.argmax(predictions_array)

#   thisplot[predicted_label].set_color('red')
#   thisplot[true_label].set_color('blue')

# # 예측 확인
#   i=0
#   plt.figure(figsize=(6,3))
#   plt.subplot(1,2,1)
#   plot_image(i, predictions[i], test_labels, test_images)
#   plt.subplot(1,2,2)
#   plot_value_array(i, predictions[i], test_labels)
#   plt.show()

#   i=12
#   plt.figure(figsize=(6,3))
#   plt.subplot(1,2,1)
#   plot_image(i, predictions[i], test_labels, test_images)
#   plt.subplot(1,2,2)
#   plot_value_array(i,predictions[i], test_labels)
#   plt.show()

# # Plot the first X test images, their predicted labels, and the true labels.
# # Color correct predict predictions in blue and incorrect predictions in red.
#   num_rows = 5
#   num_cols = 3
#   num_images = num_rows*num_cols
#   plt.figure(figsize=(2*2*num_cols, 2*num_rows))
#   for i in range(num_images):
#     plt.subplot(num_rows, 2*num_cols,2*i+1)
#     plot_image(i, predictions[i], test_labels, test_images)
#     plt.subplot(num_rows, 2*num_cols, 2*i+2)
#     plot_value_array(i, predictions[i], test_labels)
#   plt.tight_layout()
#   plt.show()

# # 훈련된 모델 사용하기
# # Grab an image from the test dataset.
#   img = test_images[1]
#   print(img.shape)

# # Add the image to a batch where it's the only member.
#   img = (np.expand_dims(img,0))
#   print(img.shape)

#   predictions_single = probability_model.predict(img)
#   print(predictions_single)

#   plot_value_array(1, predictions_single[0], test_labels)
#   _ = plt.xticks(range(10), class_names, rotation=45)
#   plt.show()

#   np.argmax(predictions_single[0])