import numpy as np
import matplotlib.pyplot as plt


# 1. 像素点的理解
def test01():

    # 构建200x200，像素值全为0的图像
    img = np.zeros([200, 300])
    print(img)
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.show()

    # 构建200x200，像素值全为255的图像
    img = np.full([200, 200], 255)
    print(img)
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.show()


# 2. 图像通道的理解
def test02():

    # 从磁盘中读取彩色图片
    img = plt.imread('data/彩色图片.png')
    print(img.shape)  # (640, 640, 4)  表示图像具有4个通道：RGBA

    img = np.transpose(img, [2, 0, 1])
    print(img.shape)

    for channel in img:
        print(channel)
        plt.imshow(channel)
        plt.show()


    # 透明通道
    print(img[3])
    # 修改透明通道值
    img[3] = 0.5
    print(img[3])
    # 显示图像：(H, W, C)
    img = np.transpose(img, [1, 2, 0])
    plt.imshow(img)
    plt.show()








if __name__ == '__main__':
    test02()