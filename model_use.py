# 如何使用让别人使用你的模型, 或者说使用 hugging face 上的模型

"""
1. 定义相同模型, 可以自己写也可以使用开源仓库中的
"""

import random

import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision import datasets, transforms


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # 输入层 -> 隐藏层1
        self.fc2 = nn.Linear(128, 64)  # 隐藏层1 -> 隐藏层2
        self.fc3 = nn.Linear(64, 10)  # 隐藏层2 -> 输出层

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # 展平成一维
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # 不加 softmax 交叉熵会自动做
        return x


"""
2. 下载并加载模型权重  "miniocean404/simple-nn"
"""
# model_path = hf_hub_download(repo_id="miniocean404/simple-nn", filename="simple.bin")
model_path = "simple.bin"
model = SimpleNN()
model.load_state_dict(torch.load(model_path))
print("加载模型成功")
# 使用模型
model.eval()

"""
3. 使用加载的模型预测 5 张图片
"""

test_set = datasets.MNIST(
    root="./data", train=False, download=True, transform=transforms.ToTensor()
)


def show_random_predictions(model, data_set):
    model.eval()
    fig, axes = plt.subplots(1, 5, figsize=(10, 2))
    for i in range(5):
        idx = random.randint(0, len(data_set) - 1)
        image, label = data_set[idx]
        with torch.no_grad():
            output = model(image.unsqueeze(0))
            pred = torch.argmax(output, dim=1).item()
        axes[i].imshow(image.squeeze(), cmap="gray")
        axes[i].set_title(f"Predicted: {pred}")
        axes[i].axis("off")

    plt.show()


show_random_predictions(model, test_set)
