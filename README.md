# AI5

### 执行环境

- windows 11
- python 3.8
- CUDA 11.6

使用包在requirements.txt文件中：

```
pandas~=1.5.3
torch~=1.13.1
scikit-learn~=1.3.2
tqdm~=4.64.1
transformers~=4.36.2
pillow~=9.4.0
torchvision~=0.14.1
```

引用的bert预训练文件在`bert-base-uncased`文件夹下

首次执行会自动下载vgg预训练模型

### 文件结构

```
│   main.py		主程序，训练并验证
│   predict.py		加载保存的.pth文件并预测
│   README.md		本文件
│   requirements.txt		执行本项目的环境中需要的包
│   test_without_label.txt		测试集
│   train.txt		训练集
│   utils.py		模型训练和预测所需要的工具，包括数据预处理和模型定义
│   
├───bert-base-uncased	bert预训练模型
│   
└───data		图片和文本数据
```



### 执行方法

将实验数据放在data文件夹中，在 https://huggingface.co/bert-base-uncased 下载bert预训练模型（本仓库中也含有）

安装需求包，并执行以下命令：

```bash
python main.py
```

可指定的参数列表（可以使用命令`python main.py -h` 或者在文件`main.py`中查看）

| 参数                | 默认值 | 说明                                                         |
| ------------------- | ------ | ------------------------------------------------------------ |
| --only              | ''     | 默认为空字符串，指定为'text'或'image'，表示只使用文本或图像模态。 |
| --lr                | 1e-5   | 学习率，控制模型权重更新的步长。                             |
| --max_input_len     | 32     | 文本输入的最大长度，用于截断或填充。                         |
| --batch_size        | 32     | 每个训练批次的样本数量。                                     |
| --epochs            | 10     | 训练的轮数，表示模型遍历整个训练数据的次数。                 |
| --num_classes       | 3      | 分类任务的类别数量。                                         |
| --bert_hidden_size  | 768    | BERT模型的特征大小。                                         |
| --image_hidden_size | 768    | 图像模型的特征大小。                                         |
| --update_parameters | False  | 是否更新VGG和BERT模型的参数。设为True表示更新，False表示不更新。 |
| --fusion_chosen     | 0      | 选择融合模型的方式，0表示GMU，1表示线性加权，2表示BiLSTM。   |


### 参考库

无

### 参考资料

https://blog.csdn.net/m0_60964321/article/details/128363479