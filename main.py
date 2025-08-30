# 训练图像识别神经网络模型
import random  # 导入随机数库，用于随机选择图片

import matplotlib.pyplot as plt  # 导入绘图库，用于显示图片
import torch  # 导入PyTorch深度学习框架
import torch.nn as nn  # 导入神经网络模块
import torchvision  # 导入计算机视觉库
import torchvision.transforms as transforms  # 导入图像变换工具

"""
1. 数据准备, 下载的数据本质上是 28*28 的灰度图片
"""
# 定义图像预处理步骤
transform = transforms.Compose(
    [
        torchvision.transforms.ToTensor(),  # 将图片转换为张量（数字矩阵）
        torchvision.transforms.Normalize((0.5,), (0.5,)),  # 将像素值标准化到-1到1之间
    ]
)
# 下载并加载 MNIST 训练数据集（手写数字0-9）
train_dataset = torchvision.datasets.MNIST(
    root="./data",  # 数据保存路径
    train=True,  # 加载训练集
    transform=transform,  # 应用上面定义的预处理
    download=True,  # 如果数据不存在就下载
)
# 创建训练数据加载器
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,  # 使用上面的训练数据集
    batch_size=64,  # 每次训练使用64张图片
    shuffle=True,  # 随机打乱数据顺序
)


# 下载并加载MNIST测试数据集
test_dataset = torchvision.datasets.MNIST(
    root="./data",  # 数据保存路径
    train=False,  # 加载测试集
    transform=transform,  # 应用预处理
    download=True,  # 如果数据不存在就下载
)
# 创建测试数据加载器
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,  # 使用测试数据集
    batch_size=64,  # 每次处理64张图片
    shuffle=False,  # 不打乱测试数据顺序
)


"""
2. 简单三层神经网络
"""


class SimpleNN(nn.Module):  # 定义一个简单的神经网络类
    def __init__(self):  # 初始化函数
        super(SimpleNN, self).__init__()  # 调用父类初始化
        self.fc1 = nn.Linear(28 * 28, 128)  # 输入层 -> 隐藏层1 : 784个输入 -> 128个输出
        self.fc2 = nn.Linear(128, 64)  # 隐藏层1 -> 隐藏层2 ：128个输入 -> 64个输出
        self.fc3 = nn.Linear(
            64, 10
        )  # 隐藏层2 -> 输出层 ：64个输入 -> 10个输出（0-9数字）

    def forward(self, x):  # 定义前向传播过程
        x = x.view(-1, 28 * 28)  # 将 28x28 的图片展平成 784 个数字的一维数组
        x = torch.relu(self.fc1(x))  # 通过第一层并应用ReLU激活函数
        x = torch.relu(self.fc2(x))  # 通过第二层并应用ReLU激活函数
        x = self.fc3(x)  # 通过第三层（输出层，不加激活函数）不加 softmax 交叉熵会自动做
        return x  # 返回预测结果


model = SimpleNN()  # 创建模型实例

"""
3. 用随机初始化的模型权重随机预测五张图片
"""


def show_random_predictions(
    model: SimpleNN, data_set: torchvision.datasets.MNIST
):  # 定义显示随机预测的函数
    model.eval()  # 将模型设置为评估模式（不训练）
    fig, axes = plt.subplots(1, 5, figsize=(10, 2))  # 创建 1 行 5 列的 ui 图表
    for i in range(5):
        idx = random.randint(0, len(data_set) - 1)
        image, label = data_set[idx]  # 获取图片和真实标签
        with torch.no_grad():  # 不计算梯度（节省内存）
            output = model(image.unsqueeze(0))  # 将图片输入模型获得预测
            pred = torch.argmax(output, dim=1).item()  # 找到概率最大的类别
        axes[i].imshow(image.squeeze(), cmap="gray")  # 显示 UI 灰度图片
        axes[i].set_title(f"Predicted: {pred}")  # 设置 UI 标题显示预测结果
        axes[i].axis("off")  # 隐藏坐标轴

    # 显示 UI 图表
    plt.show()


print("随机预测五张图片")
show_random_predictions(model, test_dataset)

"""
4. 训练模型
"""
criterion = nn.CrossEntropyLoss()  # 定义损失函数（交叉熵损失）
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 定义优化器（随机梯度下降）

epochs = 5  # 设置训练轮数为5轮
for epoch in range(epochs):  # 循环训练5轮
    running_loss = 0.0  # 初始化累计损失
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()  # 清零梯度
        outputs = model(images)  # 将图片输入模型得到预测
        loss = criterion(outputs, labels)  # 计算预测与真实标签的损失
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新模型参数
        running_loss += loss.item()  # 累加损失值

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
# api = HfApi()  # 创建Hugging Face API实例
# upload_file(  # 上传文件到Hugging Face
#     path_or_fileobj="simple.bin",  # 本地文件路径
#     path_in_repo="simple.bin",  # 仓库中的文件名
#     repo_id="miniocean404/simple-nn",  # 仓库ID
#     repo_type="model",  # 仓库类型为模型
# )
