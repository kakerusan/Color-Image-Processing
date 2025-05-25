import cv2
import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取图像
image = cv2.imread('house.bmp', cv2.IMREAD_GRAYSCALE)

# 二维傅里叶变换
f = fft2(image)
fshift = fftshift(f)
magnitude_spectrum = 20 * np.log(np.abs(fshift))

# 动态计算截止频率 阈值 energy_threshold 可调整
def calculate_cutoff_frequency(magnitude_spectrum, energy_threshold=0.8):
    #获取频谱图的行数 (rows) 和列数 (cols)，并计算频谱中心点坐标 (crow, ccol)。
    #解释：频谱图的低频能量集中在中心，需以中心为原点计算各点到中心的距离
    rows, cols = magnitude_spectrum.shape
    crow, ccol = rows // 2, cols // 2
    #生成网格坐标矩阵
    Y, X = np.ogrid[0:rows, 0:cols]
    #计算每个像素点到频谱中心的距离矩阵 radius。
    radius = np.sqrt((X - ccol)**2 + (Y - crow)**2)
    #将 radius 展平为一维数组后按值从小到大排序，并返回排序后的索引。
    sorted_indices = np.argsort(radius, axis=None)
    #获取排序后的能量矩阵
    sorted_energy = magnitude_spectrum.flatten()[sorted_indices]
    #计算截止频率 归一化为总能量的百分比
    cumulative_energy = np.cumsum(sorted_energy) / np.sum(sorted_energy)
    """
        cumulative_energy > energy_threshold：找到累积能量首次超过阈值的位置。
        np.argmax(...)：返回第一个满足条件的索引。
        通过索引从 radius 中提取对应的半径值作为截止频率 D0。
    """
    D0 = radius.flatten()[sorted_indices][np.argmax(cumulative_energy > energy_threshold)]
    return D0

D0 = calculate_cutoff_frequency(magnitude_spectrum)

# Butterworth低通滤波器
def butterworth_lowpass(rows, cols, crow, ccol, D0, n=2):
    u, v = np.meshgrid(np.arange(cols), np.arange(rows))
    D = np.sqrt((u - ccol)**2 + (v - crow)**2)
    return 1 / (1 + (D / D0)**(2*n))

rows, cols = image.shape
crow, ccol = rows // 2, cols // 2
low_pass_filter = butterworth_lowpass(rows, cols, crow, ccol, D0, n=2)

# 应用低通滤波器
f_low_pass = fshift * low_pass_filter
f_low_pass_ishift = ifftshift(f_low_pass)
img_back_low_pass = ifft2(f_low_pass_ishift)
img_back_low_pass = np.abs(img_back_low_pass)

# Butterworth高通滤波器
high_pass_filter = 1 - low_pass_filter

# 应用高通滤波器
f_high_pass = fshift * high_pass_filter
f_high_pass_ishift = ifftshift(f_high_pass)
img_back_high_pass = ifft2(f_high_pass_ishift)
img_back_high_pass = np.abs(img_back_high_pass)
img_back_high_pass = cv2.normalize(img_back_high_pass, None, 0, 255, cv2.NORM_MINMAX)

# Sobel边缘检测
sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)

# 显示结果
plt.figure(figsize=(12, 6))
plt.subplot(231), plt.imshow(image, cmap='gray'), plt.title('原始图像')
plt.subplot(232), plt.imshow(magnitude_spectrum, cmap='gray'), plt.title('幅度谱')
plt.subplot(233), plt.imshow(img_back_low_pass, cmap='gray'), plt.title(f'低通滤波 (D0={D0:.1f})')
plt.subplot(234), plt.imshow(img_back_high_pass, cmap='gray'), plt.title('高通滤波')
plt.subplot(235), plt.imshow(sobelx, cmap='gray'), plt.title('Sobel X')
plt.subplot(236), plt.imshow(sobely, cmap='gray'), plt.title('Sobel Y')
plt.show()