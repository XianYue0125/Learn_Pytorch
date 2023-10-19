import torch


def test():

    t1 = torch.tensor(30)
    t2 = torch.tensor([30])
    t3 = torch.tensor([[30]])

    print(t1.shape)
    print(t2.shape)
    print(t3.shape)

    print(t1.item())
    print(t2.item())
    print(t3.item())

    # 注意: 张量中只有一个元素，如果有多个元素的话，使用 item 函数可能就会报错
    # ValueError: only one element tensors can be converted to Python scalars
    # t4 = torch.tensor([30, 40])
    # print(t4.item())


if __name__ == '__main__':
    test()