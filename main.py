# 训练图像识别神经网络模型
import random

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

"""
1. 数据准备, 下载的数据本质上是 28*28 的灰度图片
"""
transform = transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,)),
    ]
)
train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, transform=transform, download=True
)
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=64, shuffle=True
)


test_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, transform=transform, download=True
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=64, shuffle=False
)


"""
2. 简单三层神经网络
"""


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


model = SimpleNN()

"""
3. 用随机初始化的模型权重随机预测五张图片
"""


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


print("随机预测五张图片")
show_random_predictions(model, test_dataset)

"""
4. 训练模型
"""
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

epochs = 5
for epoch in range(epochs):
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(
        f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f} loss2: {running_loss / len(train_loader):.4f}"
    )


"""
5. 用训练后的模型权重再次预测
"""
print("随机预测五张图片")
show_random_predictions(model, test_dataset)

# debugger 时候可以查看模型权重, 可以打印参数: model.fc1.weight、model.fc2.weight、model.fc3.weight
# 保存模型权重为 bin 文件
torch.save(model.state_dict(), "simple.bin")


# """
# 6. 保存模型参数到 hugging face hub
# """

# # 上传到 huggung face hub
# api = HfApi()
# upload_file(
#     path_or_fileobj="simple.bin",
#     path_in_repo="simple.bin",
#     repo_id="miniocean404/simple-nn",
#     repo_type="model",
# )
