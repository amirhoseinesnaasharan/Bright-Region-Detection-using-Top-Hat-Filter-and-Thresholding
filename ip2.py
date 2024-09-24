import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import disk, white_tophat


image_path = 'C:\\Users\\Amir\\Desktop\\python\\code python\\rice.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


 
radius = 15

# ایجاد عنصر سازنده به شکل دیسک
structuring_element = disk(radius)

# اعمال فیلتر Top-Hat
tophat_image = white_tophat(image, structuring_element)

# اعمال آستانه گذاری سراسری با مقدار اولیه آستانه 100
_, thresholded_image = cv2.threshold(tophat_image, 100, 255, cv2.THRESH_BINARY)

# نمایش تصاویر
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Top-Hat Image')
plt.imshow(tophat_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Thresholded Image')
plt.imshow(thresholded_image, cmap='gray')
plt.axis('off')

plt.show()