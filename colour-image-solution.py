import sys
import cv2
import os
import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog, ttk
from concurrent.futures import ThreadPoolExecutor

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False



class ImageProcessingApp:
    """
        图像处理应用程序类，用于加载、处理和显示彩色图像。

        参数：
        root (tk.Tk): Tkinter 主窗口对象。
        image_path (str): 默认加载的图像路径（可选）。

        功能：
        - 提供图形界面加载图像。
        - 对图像进行傅里叶变换、低通滤波、高通滤波等处理。
        - 显示原始图像、处理后的图像以及幅度谱等。
        """
    def __init__(self, root, image_path=None):
        # 初始化主窗口和图像处理参数
        self.root = root
        self.root.title('彩色图像处理')
        self.image = None  # 存储当前处理图像
        self.original_img = None  # 存储原始图像（修改初始化）
        self.filter_radius = 50  # 低通滤波器初始半径
        self.threads = 3  # 多线程处理的线程池大小

        # 创建GUI组件
        self.create_widgets()

        # 自动加载默认图像
        if image_path:
            self.load_image(image_path)  # 使用动态路径加载默认图像

    def create_widgets(self):
        """
               创建图形界面组件，包括按钮、滑块和图像显示区域。

               功能：
               - 添加文件选择按钮，用于加载图像。
               - 添加滤波器半径调节滑块，用于动态调整低通滤波器的半径。
               - 使用matplotlib创建图像显示画布，用于显示原始图像、处理后的图像。
        """
        self.btn_load = tk.Button(self.root, text='选择图像', command=self.load_image)
        self.btn_load.grid(row=0, column=0, padx=10, pady=5)

        # 滤波器半径调节滑块
        self.lbl_radius = tk.Label(self.root, text='低通滤波器半径:')
        self.lbl_radius.grid(row=0, column=1, padx=10, pady=5)
        self.scale_radius = tk.Scale(self.root, from_=10, to=200,
                                     orient=tk.HORIZONTAL, command=self.update_radius) # 创建滑动条组件 root为主窗口 最小值10 最大值200 绑定事件
        self.scale_radius.set(50)  #设置初始值为50
        self.scale_radius.grid(row=0, column=2, padx=10, pady=5)  #设置边框

        # 手动输入框
        self.radius_entry = tk.Entry(self.root, width=10)
        self.radius_entry.insert(0, str(self.filter_radius))  # 初始化输入框值
        self.radius_entry.grid(row=0, column=3, padx=10, pady=5)

        # 回车键事件绑定
        self.radius_entry.bind('<Return>', self.on_entry_submit)

        # 使用matplotlib创建图像显示画布
        self.fig, self.axes = plt.subplots(2, 3, figsize=(12, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)

        # 修改画布布局行号
        self.canvas.get_tk_widget().grid(row=2, column=0, columnspan=3, padx=10, pady=10)

    def on_entry_submit(self, event=None):
        """
        处理用户在输入框中输入的新半径值。
        如果输入无效，则恢复为当前滑块的值。
        """
        try:
            new_radius = int(self.radius_entry.get())
            if 10 <= new_radius <= 200:  # 确保数值范围有效
                self.filter_radius = new_radius
                self.scale_radius.set(new_radius)  # 同步更新滑块位置
                if self.original_img is not None:
                    self.process_and_display()  # 更新图像处理结果
            else:
                raise ValueError("超出范围")
        except ValueError:
            # 输入无效时恢复当前半径值
            self.radius_entry.delete(0, tk.END)
            self.radius_entry.insert(0, str(self.filter_radius))

    def load_image(self, file_path=None):
        """
               加载图像文件。

               参数：
               file_path (str): 图像文件路径。通过文件选择对话框选择图像。

               功能：
               - 使用cv2.imdecode读取图像，支持中文路径。
               - 加载成功后，调用process_and_display方法进行图像处理和显示。
         """
        # 文件对话框选择图像文件
        if file_path is None:
            file_path = filedialog.askopenfilename(filetypes=[('图像文件', '*.bmp;*.png;*.jpg;*.jpeg')])
        if file_path:
            # 使用imdecode处理包含中文路径的文件
            try:
                with open(file_path, 'rb') as f:
                    img_data = np.frombuffer(f.read(), dtype=np.uint8)
                    # 以彩色模式读取图像
                    self.original_img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            except Exception as e:
                print("图像读取失败:", e)
                self.original_img = None

            # 加载后立即处理显示
            if self.original_img is not None:
                # 创建并初始化进度条
                self.progress = ttk.Progressbar(
                    self.root,
                    length=400,
                    mode='determinate',
                    maximum=100,
                    value=0
                )
                self.progress.grid(row=1, column=0, columnspan=3, padx=10, pady=5)
                self.process_and_display()

    def process_and_display(self):
        """
                对图像进行处理并显示结果。

                功能：
                - 分阶段进行图像预处理、通道处理和结果显示。
                - 使用多线程处理图像的每个通道，提高处理效率。
                - 显示原始图像、幅度谱、低通滤波结果、高通滤波结果以及Sobel边缘检测结果。
        """
        # 阶段1：图像预处理
        for ax in self.axes.flat:
            ax.cla()
        self.progress['value'] = 20
        self.progress.update()

        # 显示原始彩色图像
        self.axes[0, 0].imshow(cv2.cvtColor(self.original_img, cv2.COLOR_BGR2RGB))
        self.axes[0, 0].set_title('原始图像')
        self.axes[0, 0].axis('off')
        self.progress['value'] = 30
        self.progress.update()

        # 阶段2：通道处理
        with ThreadPoolExecutor(max_workers=self.threads) as executor:
            """
            将多通道彩色图像拆分为独立的单通道数组。在 OpenCV 中：
            cv2.split() 会将 [H, W, C] 格式的图像分解为 C 个 [H, W] 的二维数组
            对于彩色图像（默认由 cv2.imdecode 加载），返回值顺序为：(B_channel, G_channel, R_channel)
            """
            channels = cv2.split(self.original_img)
            futures = []
            for channel in channels:
                futures.append(executor.submit(self.process_channel, channel))

            results = []
            for future in futures:
                results.append(future.result())

            fshifts, low_pass_results, high_pass_results = zip(*results)

            # 合并处理结果
            fshift = cv2.merge(fshifts)
            img_low_pass = cv2.merge(low_pass_results)
            img_high_pass = cv2.merge(high_pass_results)
            img_low_pass = np.clip(img_low_pass, 0, 255).astype(np.uint8)
            img_high_pass = np.clip(img_high_pass, 0, 255).astype(np.uint8)
            self.progress['value'] = 50
            self.progress.update()

        # 阶段3：结果显示
        magnitude_spectrum = 20 * np.log(np.abs(fshift).clip(1e-6))
        self.axes[0, 1].imshow(magnitude_spectrum.astype(np.uint8), cmap='gray')
        self.axes[0, 1].set_title('幅度谱')
        self.axes[0, 1].axis('off')

        img_low_pass = img_low_pass.astype(np.uint8)
        self.axes[0, 2].imshow(cv2.cvtColor(img_low_pass, cv2.COLOR_BGR2RGB))
        self.axes[0, 2].set_title(f'低通滤波（半径{self.filter_radius}）')
        self.axes[0, 2].axis('off')

        img_high_pass = img_high_pass.astype(np.uint8)
        self.axes[1, 0].imshow(cv2.cvtColor(img_high_pass, cv2.COLOR_BGR2RGB))
        self.axes[1, 0].set_title(f'高通滤波（半径{self.filter_radius}）')
        self.axes[1, 0].axis('off')

        # Sobel处理
        gray_img = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=5)
        self.progress['value'] = 80
        self.progress.update()

        self.axes[1, 1].imshow(sobelx, cmap='gray')
        self.axes[1, 1].set_title('Sobel X方向')
        self.axes[1, 1].axis('off')
        self.axes[1, 2].imshow(sobely, cmap='gray')
        self.axes[1, 2].set_title('Sobel Y方向')
        self.axes[1, 2].axis('off')

        # 完成处理
        self.progress['value'] = 100
        self.progress.update()

        # 隐藏进度条
        self.progress.grid_remove()
        self.canvas.draw()

    def update_radius(self, value):
        """
        更新低通滤波器半径参数。
        参数：
        value (str): 滑块的值（字符串形式）。
        功能：
        - 更新低通滤波器半径。
        - 如果有原始图像，则重新处理图像并显示结果。
        """

        # 更新低通滤波器半径参数
        self.filter_radius = int(value)
        if self.original_img is not None:
            # 参数变化后重新处理图像
            self.process_and_display()

    # def process_channel(self, channel):
    #     """
    #        对单通道图像进行傅里叶变换、低通滤波和高通滤波处理。
    #        参数：
    #        channel (numpy.ndarray): 输入的单通道图像数据。
    #        返回值：
    #        fshift_abs (numpy.ndarray): 傅里叶变换的幅值谱。
    #        img_low_pass (numpy.ndarray): 低通滤波后的图像。
    #        img_high_pass (numpy.ndarray): 高通滤波后的图像。
    #        功能：
    #        - 对输入的单通道图像进行二维傅里叶变换。
    #        - 将零频率分量移到频谱中心。
    #        - 创建低通滤波器掩膜，并应用低通滤波器。
    #        - 创建高通滤波器掩膜，并应用高通滤波器。
    #        - 逆傅里叶变换还原图像。
    #
    #        空间域 → 傅里叶变换 → 频域中心化 → 掩膜相乘 → 逆变换 → 空间域结果
    #        ↑           ↑            ↑            ↑
    #     channel     fft2+shift    低/高通滤波   ifft+abs
    #
    #        """
    #
    #     # 单通道图像处理流程
    #     # 二维傅里叶变换
    #     f = fft2(channel)
    #
    #     # 将零频率分量移到频谱中心 将频域矩阵原点从左上角(0,0)移动到中心位置，便于观察低频集中区域
    #     fshift = fftshift(f)
    #
    #     # 创建低通滤波器掩膜
    #     #1.获取输入通道的尺寸
    #     rows, cols = channel.shape
    #     #2.计算图像中心点坐标
    #     crow, ccol = rows // 2, cols // 2
    #     #3.创建低通滤波器掩膜 @return全零矩阵 (0表示抑制频率，1表示保留)
    #     low_pass_filter = np.zeros((rows, cols), np.uint8)
    #     # 绘制圆形滤波器区域
    #     # low_pass_filter：目标矩阵
    #     # (ccol, crow)：圆心坐标
    #     # self.filter_radius：半径（由滑块或者输入控制）
    #     # 1：圆内值
    #     # -1：填充整个圆
    #     #return 中心为1的圆形区域（允许低频通过）
    #     cv2.circle(low_pass_filter, (ccol, crow), self.filter_radius, (1,), -1)
    #
    #     # 应用低通滤波器
    #     #频域点乘（保留掩膜范围内的频率分量）
    #     f_low_pass = fshift * low_pass_filter
    #     # 逆傅里叶变换还原图像
    #         # ifftshift()：撤销fftshift操作，恢复频率分布
    #         # ifft2()：二维逆傅里叶变换（复数结果）
    #         # np.abs()：取模得到实数值图像 只有实数值才能输出
    #         # 输出：低通滤波后的图像矩阵（模糊效果）
    #     img_low_pass = np.abs(ifft2(ifftshift(f_low_pass)))
    #
    #     # 创建高通滤波器（通过取反低通滤波器） 通过取反低通掩膜获得（1-低通掩膜）
    #     high_pass_filter = np.ones((rows, cols), np.uint8) - low_pass_filter
    #     # 应用高通滤波器
    #     f_high_pass = fshift * high_pass_filter
    #     # 逆变换获得高通结果
    #     img_high_pass = np.abs(ifft2(ifftshift(f_high_pass)))
    #
    #     # 返回傅里叶变换的幅值谱、低通和高通结果
    #     # 注意：返回fshift的幅值而非复数本身,无法加载出虚数部分
    #     return np.abs(fshift), img_low_pass, img_high_pass
    def process_channel(self, channel):
        rows, cols = channel.shape
        crow, ccol = rows // 2, cols // 2

        # 高斯低通滤波器
        sigma = self.filter_radius
        X, Y = np.meshgrid(np.arange(cols) - ccol, np.arange(rows) - crow)
        low_pass_filter = np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2))  # 高斯核

        # 应用滤波器
        f = fft2(channel)
        fshift = fftshift(f)
        f_low_pass = fshift * low_pass_filter
        img_low_pass = np.abs(ifft2(ifftshift(f_low_pass)))

        # 高通滤波器（1 - 低通）
        high_pass_filter = 1 - low_pass_filter
        f_high_pass = fshift * high_pass_filter
        img_high_pass = np.abs(ifft2(ifftshift(f_high_pass)))

        return np.abs(fshift), img_low_pass, img_high_pass


if __name__ == '__main__':
    # 动态获取资源路径（用于打包后访问house.bmp）
    if getattr(sys, 'frozen', False):
        # 如果是打包后的环境，使用_MEIPASS路径
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(
            '../../python learning/sginalSystemImageSolution/imageBetter/signal-system-homework')

    default_image_path = os.path.join(base_path, 'house.bmp')

    root = tk.Tk()
    app = ImageProcessingApp(root, default_image_path)  # 传递默认图片路径参数
    root.mainloop()
