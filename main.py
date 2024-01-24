import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from transformers import AdamW
from utils import CustomDataset, ImageTextTransform, MultimodalModel

parser = argparse.ArgumentParser()

# 添加命令行参数
parser.add_argument('--only', type=str, default='', help='text or image')
parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
parser.add_argument('--max_input_len', type=int, default=32, help='Maximum text length')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--num_classes', type=int, default=3, help='Number of classes')
parser.add_argument('--bert_hidden_size', type=int, default=768, help='BERT hidden size')
parser.add_argument('--image_hidden_size', type=int, default=768, help='Image hidden size')
parser.add_argument('--update_parameters', type=bool, default=False, help='Update vgg and bert parameters')
parser.add_argument('--fusion_chosen', type=int, default=0, help='Choose fusion model, 0=gmu, 1=linear weight, 2=bilstm')

# 解析命令行参数
args = parser.parse_args()
only = args.only
max_input_len = args.max_input_len
batch_size = args.batch_size
epochs = args.epochs
num_classes = args.num_classes
bert_hidden_size = args.bert_hidden_size
image_hidden_size = args.image_hidden_size
update_parameters = args.update_parameters
fusion_chosen = args.fusion_chosen
lr = args.lr

# 读取txt文件
file_path = 'train.txt'  # 替换成你的文件路径
df = pd.read_csv(file_path)

# 划分训练集和验证集
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

train_dataset = CustomDataset(data_folder='data', dataframe=train_df, max_input_len=max_input_len, only=only,
                              transform=ImageTextTransform)
val_dataset = CustomDataset(data_folder='data', dataframe=val_df, max_input_len=max_input_len, only=only,
                            transform=ImageTextTransform)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建多模态模型
multimodal_model = MultimodalModel(text_input_dim=bert_hidden_size, image_input_dim=image_hidden_size,
                                   output_dim=num_classes, update_parameters=update_parameters,
                                   fusion_chosen=fusion_chosen)
# multimodal_model = MultimodalModel()
multimodal_model.to(device)

# 定义优化器和损失函数
optimizer = AdamW(multimodal_model.parameters(), lr=lr)
criterion = torch.nn.CrossEntropyLoss()

# 训练循环
for epoch in range(epochs):
    # 训练
    total_loss = 0.0
    total_samples = 0
    correct_predictions = 0
    for batch in tqdm(train_dataloader, desc="Training", leave=False):
        inputs = {key: value.to(device) for key, value in batch.items() if key != 'guid'}
        optimizer.zero_grad()
        # 前向传播
        outputs = multimodal_model(inputs)
        # 计算损失
        loss = criterion(outputs, inputs['labels'])
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == inputs['labels']).sum().item()
        total_samples += len(inputs['labels'])
    accuracy = correct_predictions / total_samples
    print(f"Train Accuracy: {accuracy * 100:.2f}%")

    # 验证
    total_val_loss = 0.0
    total_samples = 0
    correct_predictions = 0
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Evaluating", leave=False):
            inputs = {key: value.to(device) for key, value in batch.items() if key != 'guid'}
            # 前向传播
            outputs = multimodal_model(inputs)
            # 计算损失
            loss = criterion(outputs, inputs['labels'])
            total_val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == inputs['labels']).sum().item()
            total_samples += len(inputs['labels'])

    accuracy = correct_predictions / total_samples
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")
    print(
        f"Epoch {epoch + 1}/{epochs} - Train Loss: {total_loss / len(train_dataloader):.4f} - Val Loss: {total_val_loss / len(val_dataloader):.4f}")

torch.save(multimodal_model.state_dict(), 'multimodal_model.pth')
