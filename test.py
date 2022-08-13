# https://colab.research.google.com/drive/1oX6hPlMPW_CNKwpBtmREDaz5jzBGL13E#scrollTo=BoXXdwulH5wB
import os
import cv2
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

classes = ["cat", "dog"]
model = keras.models.load_model('CatsDogsModel.h5')

images = []
for image_name in os.listdir("images"):
    # print(image_name)
    img = cv2.imread(r"images/" + image_name)
    img = cv2.resize(img, (224, 224))

    data = np.array([img])
    pred = model.predict(data)

    label = np.argmax(pred)
    print(f"{image_name}: {classes[label]}")

    cv2.putText(img, f"Class: {classes[label]}", (5, 25), cv2.FONT_HERSHEY_PLAIN, 2, (200, 20, 50), 2)
    images.append(img)
    # cv2.imshow(image_name, img)
    # cv2.waitKey(0)

for i in range(18):
    plt.subplot(3, 6, i + 1)
    plt.imshow(images[i], cmap='gray')
plt.show()


