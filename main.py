# 训练图像识别神经网络模型
import random  # 导入随机数库，用于随机选择图片

import matplotlib.pyplot as plt  # 导入绘图库，用于显示图片
import torch  # 导入 PyTorch 深度学习框架
import torch.nn as nn  # 导入神经网络模块
import torchvision  # 导入计算机视觉库
import torchvision.transforms as transforms  # 导入图像变换工具

"""
1. 数据准备，下载的数据本质上是 28*28 的灰度图片
"""
# 定义图像预处理步骤
transform = transforms.Compose(
    [
        torchvision.transforms.ToTensor(),  # 将图片转换为张量（数字矩阵）
        torchvision.transforms.Normalize(
            (0.5,), (0.5,)
        ),  # 将像素值标准化到 - 1 到 1 之间
    ]
)
# 下载并加载 MNIST 训练数据集（手写数字 0-9）
train_dataset = torchvision.datasets.MNIST(
    root="./data",  # 数据保存路径
    train=True,  # 加载训练集
    transform=transform,  # 应用上面定义的预处理
    download=True,  # 如果数据不存在就下载
)
# 创建训练数据加载器
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,  # 使用上面的训练数据集
    batch_size=64,  # 每次训练使用 64 张图片
    shuffle=True,  # 随机打乱数据顺序
)


# 下载并加载 MNIST 测试数据集
test_dataset = torchvision.datasets.MNIST(
    root="./data",  # 数据保存路径
    train=False,  # 加载测试集
    transform=transform,  # 应用预处理
    download=True,  # 如果数据不存在就下载
)
# 创建测试数据加载器
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,  # 使用测试数据集
    batch_size=64,  # 每次处理 64 张图片
    shuffle=False,  # 不打乱测试数据顺序
)


"""
2. 简单三层神经网络
"""


class SimpleNN(nn.Module):  # 定义一个简单的神经网络类
    fc1: nn.Linear
    fc2: nn.Linear
    fc3: nn.Linear

    def __init__(self):  # 初始化函数
        super(SimpleNN, self).__init__()  # 调用父类初始化
        self.fc1 = nn.Linear(
            28 * 28, 128
        )  # 输入层 -> 隐藏层 1 : 784 个输入 -> 128 个输出
        self.fc2 = nn.Linear(128, 64)  # 隐藏层 1 -> 隐藏层 2 ：128 个输入 -> 64 个输出
        self.fc3 = nn.Linear(
            64, 10
        )  # 隐藏层 2 -> 输出层 ：64 个输入 -> 10 个输出（0-9 数字）

    # 定义前向传播过程，(个人理解用于计算概率)
    def forward(self, x):
        x = x.view(-1, 28 * 28)  # 将 28x28 的图片展平成 784 个数字的一维数组
        """
        ReLU（Rectified Linear Unit，修正线性单元）
        # 引入非线性
        神经网络如果只有线性变换（矩阵乘法），无论有多少层，本质上都等价于一个线性函数
        ReLU 引入非线性，使神经网络能够学习复杂的非线性关系

        # 解决梯度消失问题
        传统激活函数（如 sigmoid 、tanh ）在深层网络中容易出现梯度消失
        ReLU 在正区间的梯度恒为 1，有效缓解了梯度消失问题，使得深层网络的训练更加稳定

        # 计算效率高
        ReLU 只需要一个简单的阈值操作： max (0, x), 相比 sigmoid 等需要指数运算的函数，计算速度更快，节省训练时间和计算资源

        # 稀疏激活
        ReLU 会将负值 "杀死"（置为 0），产生稀疏的激活模式，这种稀疏性有助于网络学习更有效的特征表示，减少了神经元之间的相互依赖

        # 在您的 MNIST 项目中的具体作用
        在您的手写数字识别项目中：
        1. 第一层 ReLU：处理从 784 个像素输入到 128 个隐藏单元的变换，帮助网络学习基本的图像特征
        2. 第二层 ReLU：进一步提取和组合特征，从 128 维降到 64 维
        3. 输出层不使用 ReLU：因为需要输出 10 个类别的原始分数，供交叉熵损失函数处理

        # 为什么选择 ReLU
        相比其他激活函数：
        1. 比 sigmoid 更好：避免梯度消失，计算更快
        2. 比 tanh 更好：同样避免梯度消失，且输出范围更适合
        3. 比 Leaky ReLU 更简单：在大多数情况下效果相当，但更简单
        """
        x = torch.relu(self.fc1(x))  # 通过第一层输入层并应用 ReLU 激活函数
        x = torch.relu(self.fc2(x))  # 通过第二层隐藏层并应用 ReLU 激活函数
        x = self.fc3(
            x
        )  # 通过第三层（输出层，不加激活函数）不加 softmax 交叉熵 CrossEntropyLoss 会自动做
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
            # 将图片输入模型获得预测,
            # unsqueeze 用于将二维数组图数据 [[1, 2], [3, 4]] 片转化为四维数组 [[[1, 2], [3, 4]]], 之前：[高度, 宽度, 通道数] -> [224, 224, 3] (这是一张图片)，之后：[批次大小, 高度, 宽度, 通道数] -> [1, 224, 224, 3] (这是一个包含一张图片的批次)
            # unsqueeze 参数代表在哪个梯度进行扩展dim=0: 在最前面加维度、dim=1: 在原来的第0维和第1维之间加维度、dim=-1: 在最后面加维度（非常常用）
            output = model(image.unsqueeze(0))
            pred = torch.argmax(output, dim=1).item()  # 找到概率最大的类别的索引的值

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

epochs = 5  # 设置训练轮数为 5 轮
for epoch in range(epochs):  # 循环训练 5 轮
    running_loss = 0.0  # 初始化累计损失
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()  # 清除上一次迭代的梯度，确保每次迭代使用的是当前批次的梯度
        outputs = model(images)  # 将图片输入模型得到预测
        loss = criterion(outputs, labels)  # 计算预测与真实标签的损失

        # 反向传播过程，(用于根据损失函数计算自己定义的神经网络在什么梯度下可以识别出来)
        """
        # 反向传播计算梯度（Backward）的作用
        反向传播是深度学习的核心算法，用于计算损失函数相对于网络中每个参数的梯度（偏导数）
        ## 为什么需要梯度
        ### 梯度的含义：
        1. 梯度告诉我们：如果稍微改变某个参数，损失函数会如何变化
        2. 梯度的方向指向损失函数增长最快的方向
        3. 梯度的大小表示损失函数变化的速率
        ### 参数更新的依据：
        1. 我们想要减小损失，所以要朝梯度的反方向更新参数
        2. 这就是梯度下降算法的基本思想

        # 前向传播（计算预测）
        输入图像 → fc1 → ReLU → fc2 → ReLU → fc3 → 输出预测
        # 反向传播（计算梯度）
        损失函数 ← ∂L/∂fc3 ← ∂L/∂fc2 ← ∂L/∂fc1 ← 梯度传播

        ## 具体的计算过程
        反向传播使用链式法则计算复合函数的导数：∂Loss/∂fc1_weight = ∂Loss/∂output × ∂output/∂fc3 × ∂fc3/∂fc2 × ∂fc2/∂fc1 × ∂fc1/∂fc1_weight
        逐层计算：
        1. 输出层：计算损失相对于输出层参数的梯度
        2. 隐藏层2：利用输出层的梯度，计算隐藏层2参数的梯度
        3. 隐藏层1：利用隐藏层2的梯度，计算隐藏层1参数的梯度

        loss.backward()  # 计算所有参数的梯度
        自动计算：
        - model.fc1.weight.grad  # fc1权重的梯度
        - model.fc1.bias.grad    # fc1偏置的梯度
        - model.fc2.weight.grad  # fc2权重的梯度
        - model.fc2.bias.grad    # fc2偏置的梯度
        - model.fc3.weight.grad  # fc3权重的梯度
        - model.fc3.bias.grad    # fc3偏置的梯度
        """
        loss.backward()
        optimizer.step()  # 使用梯度更新参数
        running_loss += loss.item()  # 累加损失值

    print(
        f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f} loss2: {running_loss / len(train_loader):.4f}"
    )


"""
5. 用训练后的模型权重再次预测
"""
print("随机预测五张图片")
show_random_predictions(model, test_dataset)

# debugger 时候可以查看模型权重，可以打印参数: model.fc1.weight、model.fc2.weight、model.fc3.weight
# 保存模型权重为 bin 文件
torch.save(model.state_dict(), "simple.bin")


# """
# 6. 保存模型参数到 hugging face hub
# """

# # 上传到 huggung face hub
# api = HfApi ()  # 创建 Hugging Face API 实例
# upload_file (  # 上传文件到 Hugging Face
#     path_or_fileobj="simple.bin",  # 本地文件路径
#     path_in_repo="simple.bin",  # 仓库中的文件名
#     repo_id="miniocean404/simple-nn",  # 仓库 ID
#     repo_type="model",  # 仓库类型为模型
# )
