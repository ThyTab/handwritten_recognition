import tkinter as tk
from tkinter import ttk, messagebox
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from PIL import Image, ImageDraw
import os

# 设备配置（GPU优先）
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 画布尺寸（可视化尺寸，实际识别会缩放到28x28）
CANVAS_SIZE = 280  # 10倍放大便于手绘
PIXEL_SIZE = CANVAS_SIZE // 28  # 每个MNIST像素对应画布的像素数
# 模型保存路径
MODEL_PATH = "手写数字识别/model_save_LeNet.pth"


# 1.加载数据集
def load_data():
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))])
    train_dataset = datasets.MNIST(root='./data',train=True,transform=transform,download=True)
    test_dataset = datasets.MNIST(root='./data',train=False,transform=transform,download=True)
    train_iter = DataLoader(train_dataset,batch_size=256,shuffle=True)
    test_iter = DataLoader(test_dataset,batch_size=256,shuffle=False)
    return train_iter,test_iter

# 2.定义模型
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2,2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(16*4*4,120),
            nn.Sigmoid(),
            nn.Linear(120,84),
            nn.Sigmoid(),
            nn.Linear(84,10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 16*4*4)  # 展平
        x = self.fc_layers(x)
        return x
    
# 3.训练模型
def train(train_iter,test_iter):
    model = LeNet().to(DEVICE)
    criterion = nn.CrossEntropyLoss()   #自带softmax
    optimizer = optim.SGD(model.parameters(),lr=0.01,momentum=0.9)
    #开始训练
    epochs,loss_list = 50,[]
    model.train()
    for epoch in range(epochs):
        total_loss,total_sample = 0,0
        for train_x,train_y in train_iter:
            train_x,train_y = train_x.to(DEVICE),train_y.to(DEVICE)
            #前向传播
            y_pred = model(train_x)
            #计算损失
            loss = criterion(y_pred,train_y)
            total_loss += loss.item()*train_x.shape[0]
            total_sample += train_x.shape[0]
            #梯度清零 + 反向传播 + 优化参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_list.append(total_loss/total_sample)
    return loss_list,model

# 4.模型评估
def evaluate(model,test_iter):
    model.eval()   #切换模型模式
    correct,total = 0,0
    with torch.no_grad():   #关闭梯度计算
        for test_x,test_y in test_iter:
            test_x,test_y = test_x.to(DEVICE),test_y.to(DEVICE)
            y_pred = model(test_x)
            y_ans = torch.argmax(y_pred,dim=1)   #取最大值索引
            correct += (y_ans==test_y).sum().item()   #只用.sum()得出的是一个一阶张量
            total += test_x.shape[0]
    accuracy = correct/total
    return accuracy

# 5.训练并保存模型
def train_and_save_model():
    train_iter,test_iter = load_data()
    loss_list,model = train(train_iter,test_iter)
    accuracy = evaluate(model,test_iter)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"模型训练完成，测试集准确率: {accuracy*100:.2f}%")

# 6.加载预训练模型
def load_model():
    """加载预训练模型（无模型则自动训练）"""
    model = LeNet().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("已加载预训练模型")
        accuracy = evaluate(model, load_data()[1])
        print(f"测试集准确率: {accuracy*100:.2f}%")
    else:
        messagebox.showinfo("提示", "未找到预训练模型,开始训练(首次运行约1分钟)...")
        train_and_save_model()
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()  # 切换到评估模式
    return model

# 7.定义GUI应用
class HandwritingRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PyTorch手写数字识别")
        self.root.geometry(f"{CANVAS_SIZE + 200}x{CANVAS_SIZE + 80}")  # 画布+右侧结果区
        self.root.resizable(False, False)

        # 加载模型
        self.model = load_model()

        # 初始化画布相关变量
        self.canvas = None
        self.image = Image.new("L", (28, 28), 0)  # 28x28灰度图（MNIST规格）
        self.draw = ImageDraw.Draw(self.image)
        self.last_x, self.last_y = None, None

        # 构建界面
        self._build_ui()

    def _build_ui(self):
        """构建界面布局"""
        # 1. 手绘画布
        self.canvas = tk.Canvas(
            self.root, 
            width=CANVAS_SIZE, 
            height=CANVAS_SIZE, 
            bg="white",
            relief="solid",
            bd=2
        )
        self.canvas.grid(row=0, column=0, rowspan=3, padx=10, pady=10)
        
        # 画布绑定鼠标事件
        self.canvas.bind("<B1-Motion>", self._draw_on_canvas)  # 按住拖动
        self.canvas.bind("<ButtonRelease-1>", self._reset_last_xy)  # 释放鼠标

        # 2. 右侧控件区
        # 识别结果标签
        self.result_label = ttk.Label(
            self.root, 
            text="识别结果: \n置信度: ",
            font=("微软雅黑", 14)
        )
        self.result_label.grid(row=0, column=1, padx=10, pady=20)

        # 清除按钮
        clear_btn = ttk.Button(
            self.root, 
            text="清除画布", 
            command=self._clear_canvas,
            width=15
        )
        clear_btn.grid(row=1, column=1, padx=10, pady=10)

        # 识别按钮
        recognize_btn = ttk.Button(
            self.root, 
            text="识别数字", 
            command=self._recognize_digit,
            width=15
        )
        recognize_btn.grid(row=2, column=1, padx=10, pady=10)

    def _draw_on_canvas(self, event):
        """在画布上绘制(同步到28x28图像)"""
        # 1. 在可视化画布上绘制
        if self.last_x and self.last_y:
            self.canvas.create_line(
                self.last_x, self.last_y, event.x, event.y,
                width=15, fill="black", capstyle=tk.ROUND, smooth=tk.TRUE
            )
        
        # 2. 同步绘制到28x28的MNIST规格图像
        x = event.x // PIXEL_SIZE
        y = event.y // PIXEL_SIZE
        if 0 <= x < 28 and 0 <= y < 28:
            # 绘制一个3x3的区域（增强手绘效果）
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    if 0 <= x+dx < 28 and 0 <= y+dy < 28:
                        self.draw.point((x+dx, y+dy), 255)  # 白色为笔迹（MNIST是黑底白字）
        
        self.last_x, self.last_y = event.x, event.y

    def _reset_last_xy(self, event):
        """重置鼠标坐标"""
        self.last_x, self.last_y = None, None

    def _clear_canvas(self):
        """清空画布和图像"""
        self.canvas.delete("all")
        self.image = Image.new("L", (28, 28), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="识别结果: \n置信度: ")

    def _recognize_digit(self):
        """识别手绘数字"""
        try:
            # 1. 图像预处理（适配MNIST）
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            img_tensor = transform(self.image).unsqueeze(0).to(DEVICE)  # 增加batch维度

            # 2. 模型推理
            with torch.no_grad():
                output = self.model(img_tensor)
                probabilities = torch.softmax(output, dim=1)  # 转换为概率
                pred_digit = torch.argmax(probabilities, dim=1).item()
                pred_confidence = probabilities[0][pred_digit].item() * 100

            # 3. 更新结果显示
            self.result_label.config(
                text=f"识别结果: {pred_digit}\n置信度: {pred_confidence:.2f}%"
            )

        except Exception as e:
            messagebox.showerror("错误", f"识别失败: {str(e)}")

# 运行应用
if __name__ == "__main__":
    root = tk.Tk()
    app = HandwritingRecognitionApp(root)
    root.mainloop()
