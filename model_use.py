# 如何使用让别人使用你的模型, 或者说使用 hugging face 上的模型

"""
1. 定义相同模型, 可以自己写也可以使用开源仓库中的
"""

import random  # 导入随机数库

import matplotlib.pyplot as plt  # 导入绘图库，用于显示图片
import torch  # 导入PyTorch深度学习框架
from torch import nn  # 从torch中导入神经网络模块
from torchvision import datasets, transforms  # 导入数据集和图像变换工具


# 定义与训练时相同的神经网络结构
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


"""
2. 下载并加载模型权重  "miniocean404/simple-nn"
"""
# 从 hugging face 下载模型权重文件
# model_path = hf_hub_download(repo_id="miniocean404/simple-nn", filename="simple.bin")
model_path = "simple.bin"  # 指定本地模型文件路径, 也可以从 hf 下载
model = SimpleNN()  # 创建模型实例（此时权重是随机的）
model.load_state_dict(torch.load(model_path))  # 加载训练好的模型权重
print("加载模型成功")  # 打印成功信息

# 使用模型
model.eval()  # 将模型设置为评估模式

"""
3. 使用加载的模型预测 5 张图片
"""

test_set = datasets.MNIST(  # 加载MNIST测试数据集
    root="./data",  # 数据保存路径
    train=False,  # 加载测试集（不是训练集）
    download=True,  # 如果数据不存在就下载
    transform=transforms.ToTensor(),  # 将图片转换为张量
)


def show_random_predictions(model: SimpleNN, data_set: datasets.MNIST):
    model.eval()  # 确保模型处于评估模式
    fig, axes = plt.subplots(1, 5, figsize=(10, 2))
    for i in range(5):
        idx = random.randint(0, len(data_set) - 1)
        image, label = data_set[idx]
        with torch.no_grad():  # 不计算梯度（推理时不需要）
            output = model(image.unsqueeze(0))  # 将图片输入模型获得预测
            pred = torch.argmax(output, dim=1).item()  # 找到概率最大的数字
        axes[i].imshow(image.squeeze(), cmap="gray")  # 显示灰度图片
        axes[i].set_title(f"Predicted: {pred}")
        axes[i].axis("off")

    plt.show()


show_random_predictions(model, test_set)  # 调用函数进行预测并显示结果
