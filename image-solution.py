import cv2
import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文显示
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# 原始图像与幅度谱：
# 原始图像显示房屋等建筑结构。
# 幅度谱显示图像的频率分布，中心区域较亮（低频），周围区域较暗（高频）。
# 低通滤波后图像：
# 图像变得模糊，边缘和细节减少。
# 整体结构仍然可见，但不够清晰。
# 高通滤波后图像：
# 主要显示边缘和细节。
# 背景变得暗淡或消失，只剩下轮廓和边缘线。
# Sobel X 和 Sobel Y 方向：
# Sobel X 方向显示水平方向上的边缘。
# Sobel Y 方向显示垂直方向上的边缘。
# 边缘线出现在图像中亮度变化较大的地方。
# https://www.woshicver.com/

#半径
r = 50

# 读取图像
image = cv2.imread('../house.bmp', cv2.IMREAD_GRAYSCALE)
plt.figure(figsize=(8, 4))
plt.subplot(121)
plt.imshow(image, cmap='gray'), plt.title('原始图像')
plt.axis('off')

# 二维傅里叶变换
f = fft2(image)
fshift = fftshift(f)

# 显示频谱图
magnitude_spectrum = 20 * np.log(np.abs(fshift))
plt.subplot(122)
plt.imshow(magnitude_spectrum, cmap='gray'), plt.title('幅度谱')
plt.axis('off')
plt.show()

# 设定截止频率
rows, cols = image.shape
crow, ccol = rows // 2, cols // 2

# 创建低通滤波器
low_pass_filter = np.zeros((rows, cols), np.uint8)
cv2.circle(low_pass_filter, (ccol, crow), r, 1, -1)

# 应用低通滤波器
f_low_pass = fshift * low_pass_filter
f_low_pass_ishift = ifftshift(f_low_pass)
img_back_low_pass = ifft2(f_low_pass_ishift)
img_back_low_pass = np.abs(img_back_low_pass)

# 显示低通滤波后的图像
plt.figure(figsize=(8, 4))
plt.subplot(121), plt.imshow(img_back_low_pass, cmap='gray'), plt.title('低通滤波后图像')
plt.axis('off')

# 创建高通滤波器
high_pass_filter = np.ones((rows, cols), np.uint8) - low_pass_filter

# 应用高通滤波器
f_high_pass = fshift * high_pass_filter
f_high_pass_ishift = ifftshift(f_high_pass)
img_back_high_pass = ifft2(f_high_pass_ishift)
img_back_high_pass = np.abs(img_back_high_pass)

# 显示高通滤波后的图像
plt.subplot(122), plt.imshow(img_back_high_pass, cmap='gray'), plt.title('高通滤波后图像')
plt.axis('off')
plt.show()

# 计算二维一阶差分结果
sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)

# 显示Sobel X方向的结果
plt.figure(figsize=(8, 4))
plt.subplot(121), plt.imshow(sobelx, cmap='gray'), plt.title('Sobel X 方向')
plt.axis('off')

# 显示Sobel Y方向的结果
plt.subplot(122), plt.imshow(sobely, cmap='gray'), plt.title('Sobel Y 方向')
plt.axis('off')
plt.show()



