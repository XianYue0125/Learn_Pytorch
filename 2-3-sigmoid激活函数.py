import torch
import matplotlib.pyplot as plt


def test():

    _, axes = plt.subplots(1, 2)

    # 绘制函数图像
    x = torch.linspace(-20, 20, 1000)
    y = torch.sigmoid(x)

    axes[0].plot(x, y)
    axes[0].grid()
    axes[0].set_title('sigmoid 函数图像')

    # 绘制导数图像
    x = torch.linspace(-20, 20, 1000, requires_grad=True)
    torch.sigmoid(x).sum().backward()

    axes[1].plot(x.detach(), x.grad)
    axes[1].grid()
    axes[1].set_title('sigmoid 导数图像')


    plt.show()


if __name__ == '__main__':
    test()