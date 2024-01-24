import os
import re
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision.models import vgg16
from transformers import BertTokenizer, BertModel
import torch
from torchvision import transforms

ImageTextTransform = transforms.Compose([
    transforms.Resize((224, 224)),  # 将图像resize为224*224
    transforms.ToTensor(),  # 将图像转换为PyTorch张量
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化图像
])


# class SpecialTokens:
#     PAD_TOKEN = "[PAD]"
#     UNK_TOKEN = "[UNK]"
#     SOS_TOKEN = "[SOS]"
#     EOS_TOKEN = "[EOS]"
#     NON_TOKEN = "[NON]"
#     BRG_TOKEN = "[BRG]"
#     OPT_TOKEN = "[OPT]"
#     CLS_TOKEN = "[CLS]"
#     SEP_TOKEN = "[SEP]"
#     MASK_TOKEN = "[MASK]"
#     NUM_TOKEN = '[NUM]'
#     USER_TOKEN = '[USER]'
#     TAG_TOKEN = '[TAG]'


class CustomDataset(Dataset):
    def __init__(self, data_folder, dataframe, max_input_len, only, transform=None):
        self.data_folder = data_folder
        self.dataframe = dataframe
        self.transform = transform
        self.pretrained_tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased')
        self.vocab_size = self.pretrained_tokenizer.vocab_size + 13
        # self.special_tokens = [SpecialTokens.__dict__[k] for k in SpecialTokens.__dict__ if not re.search('^\_', k)]
        # self.pretrained_tokenizer.add_special_tokens({'additional_special_tokens': self.special_tokens})
        # self.special_tokens.sort()
        self.max_input_len = max_input_len
        self.only = only

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        guid = self.dataframe.iloc[idx]['guid']
        guid=str(int(guid))
        txt_filepath = os.path.join(self.data_folder, f"{guid}.txt")
        img_filepath = os.path.join(self.data_folder, f"{guid}.jpg")
        # print(txt_filepath)
        try:
            with open(txt_filepath, encoding="utf-8") as f:
                text_data = f.read()
        except:
            with open(txt_filepath, 'r', encoding='utf-8', errors='ignore') as f:
                text_data = f.read()

        if self.only == 'text':
            image_data = Image.new(mode='RGB', size=(224, 224), color=(0, 0, 0))
        else:
            image_data = Image.open(img_filepath).convert('RGB')

        if self.only == 'img':
            word_list = []
        else:
            word_list = text_data.strip('\n').strip().replace("#", "").split(' ')
        words_result = []

        for word in word_list:
            if len(word) < 1:
                continue
            elif (len(word) >= 4 and 'http' in word) or word[0] == '@':
                continue
            else:
                words_result.append(word)
        sequence = " ".join(words_result)
        if self.transform:
            image_data = self.transform(image_data)

        # 第一次成功
        # image_data = Image.open(img_filepath)
        # img = image_data.resize((224, 224), Image.Resampling.LANCZOS)
        # img = np.asarray(img, dtype='float32')
        # image_data = img.transpose(2, 0, 1)
        # image_data = torch.Tensor(image_data)
        # word_list = text_data.replace("#", "").split(" ")
        # words_result = []
        # for word in word_list:
        #     if len(word) < 1:
        #         continue
        #     elif (len(word) >= 4 and 'http' in word) or word[0] == '@':
        #         continue
        #     else:
        #         words_result.append(word)
        # sequence = " ".join(words_result)

        # if self.only == 'text':
        #     image_data = Image.new(mode='RGB', size=(224, 224), color=(0, 0, 0))
        # else:
        #     image_data = Image.open(img_filepath).convert('RGB')
        #
        # if self.only == 'img':
        #     text_data = ''
        # else:
        #     text_data = text_data.strip('\n').strip().split(' ')
        #
        # # 图片数据预处理
        # if self.transform:
        #     image_data = self.transform(image_data)
        # # 文本数据预处理

        result = self.pretrained_tokenizer.batch_encode_plus(batch_text_or_text_pairs=[sequence], truncation=True,
                                                             padding='max_length', max_length=self.max_input_len,
                                                             return_tensors='pt')
        token_ids = result['input_ids'][0]
        # print(token_ids.shape)
        attention_mask = result['attention_mask'][0]

        label = self.dataframe.iloc[idx]['tag']
        if label == "positive":
            tag = 2
        elif label == "negative":
            tag = 0
        elif label == "neutral":
            tag = 1
        else:
            tag = -1
        # print(token_ids.shape, attention_mask.shape, image_data.shape)
        return {'guid': guid, 'input_ids': token_ids, 'attention_mask': attention_mask, 'image': image_data,
                'labels': torch.tensor(tag)}


class GMU(nn.Module):
    def __init__(self, input_size_text, input_size_image, hidden_size):
        super(GMU, self).__init__()
        self.hidden_size = hidden_size
        self.linear_text = nn.Linear(input_size_text, hidden_size)
        self.linear_image = nn.Linear(input_size_image, hidden_size)
        self.z_gate = nn.Linear(input_size_text + input_size_image, hidden_size)

    def forward(self, text, image):
        h_text = torch.tanh(self.linear_text(text))
        h_image = torch.tanh(self.linear_image(image))
        z = torch.sigmoid(self.z_gate(torch.cat([text, image], dim=1)))
        return z * h_text + (1 - z) * h_image


# gmu融合
class MultimodalFusionLayer(nn.Module):
    def __init__(self, input_size_text, input_size_image, num_classes, hidden_size=100):
        super(MultimodalFusionLayer, self).__init__()
        self.gmu = GMU(input_size_text, input_size_image, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, text, image):
        combined = self.gmu(text, image)
        output = self.fc(combined)
        return output


# 线性加权融合
class MultimodalFusionLayer1(nn.Module):
    def __init__(self, input_size_text, input_size_image, num_classes):
        super(MultimodalFusionLayer1, self).__init__()
        self.img_weight = nn.Linear(input_size_image, 1)
        self.txt_weight = nn.Linear(input_size_text, 1)
        self.fc = nn.Linear(input_size_text, num_classes)

    def forward(self, text, image):
        img_weight = self.img_weight(image)
        txt_weight = self.txt_weight(text)
        output = img_weight * image + txt_weight * text
        output = self.fc(output)
        return output


# bilstm
class MultimodalFusionLayer2(nn.Module):
    def __init__(self, input_size_text, input_size_image, num_classes, hidden_size=100):
        super(MultimodalFusionLayer2, self).__init__()
        self.bilstm = torch.nn.LSTM(input_size=input_size_text + input_size_image, hidden_size=hidden_size,
                                    bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, text, image):
        inputs = torch.cat([text, image], dim=1)
        outputs, (hidden, cell) = self.bilstm(inputs)
        output = self.fc(outputs)
        return output


class MultimodalModel(nn.Module):
    def __init__(self, text_input_dim, image_input_dim, output_dim, update_parameters=False, fusion_chosen=0):
        super(MultimodalModel, self).__init__()

        # 文本模型
        self.text_model = BertModel.from_pretrained('./bert-base-uncased')
        self.text_trans = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.text_model.config.hidden_size, text_input_dim),
            nn.ReLU(inplace=True)
        )
        if update_parameters:
            for param in self.text_model.parameters():
                param.requires_grad = True
        # 图片模型
        self.full_model = vgg16(pretrained=True)
        self.image_model = nn.Sequential(
            *(list(self.full_model.children())[:-1]),
            nn.Flatten()
        )
        self.image_trans = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(25088, image_input_dim),
            nn.ReLU(inplace=True)
        )
        if update_parameters:
            for param in self.full_model.parameters():
                param.requires_grad = True

        # 融合层
        if fusion_chosen == 0:
            self.fusion_layer = MultimodalFusionLayer(text_input_dim, image_input_dim, output_dim)
        elif fusion_chosen == 1:
            self.fusion_layer = MultimodalFusionLayer1(text_input_dim, image_input_dim, output_dim)
        elif fusion_chosen == 2:
            self.fusion_layer = MultimodalFusionLayer2(text_input_dim, image_input_dim, output_dim)

    def forward(self, inputs):
        # 文本模型的前向传播
        text_outputs = self.text_model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        pooler_out = text_outputs['pooler_output']
        # text_logits = text_outputs.last_hidden_state[:, 0, :]
        text_embedding = self.text_trans(pooler_out)
        # 图片模型的前向传播
        image_outputs = self.image_model(inputs['image'])
        image_embedding = self.image_trans(image_outputs)
        # 融合层的前向传播
        fused_representation = self.fusion_layer(text_embedding, image_embedding)

        return fused_representation
