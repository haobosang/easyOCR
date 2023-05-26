# 1. 创建日文假数据标签
```
$ python createjson.py  # 可修改 num_data 生成发票标签的数量
```
运行结束后，会在 ```fakeLabel``` 文件夹中生成对应的 ```json``` 文件和 ```label.csv``` 文件
# 2. 创建发票图片
首先以 ```template_img/2.png``` 作为模板生成数据，根据上述得到的 ```json``` 文件生成日文发票假数据。
```
$ python fakeImageGenerator.py
```
运行结束后，在 ```fakeImage``` 中生成对应个数的图片
# 3. 创建 lmdb 格式的数据集
```
$ cd deep-text-recognition
```
首先运行 ```create_lmdb_dataset.py``` 文件中 ```InvoiceDataset('..')``` 函数，将生成的假发票数据依据 ```../fakeLabel/label.csv``` 文件，按照坐标切割发票图片为小样本数据，输出到 ```./output/images/```，并生成每一部分的标签 ```./output/gt.txt```。
```
# 注释掉 InvoiceDataset('..');
$ python create_lmdb_dataset.py
```
若没有切割过发票数据，两个可以一起运行。最终输出到 ```./output/data.mdb``` 和 ```./output/lock.mdb```
# 4. dataset.py 的修改
在 180 行左右注释掉：
```
...
        out_of_char = f'[^{self.opt.character}]'
        if re.search(out_of_char, label.lower()):
            continue 
...
```
该部分的代码的作用是过滤掉不在 character 中的字符，原先是的字符集只限制在 ```0123456789abcdefghijklmnopqrstuvwxyz``` 中，日文的训练我们不采用这种过滤方式。
# 5. 日文字符集
日文的所有字符已在 ```ja_char.txt``` 中
# 6. train.py 的修改
可以定义两种模型结构：
- 轻量模型：```None-VGG-BiLSTM-CTC```
- 大模型：```TPS-ResNet-BiLSTM-Attn```

在第一次训练时，不能使用 ```opt.saved_model``` 参数，否则会出现原有模型输出尺寸和现在的字符分类任务的字符分类数对不上。```train.py``` 现已做出修改：
### 93 行：
```
""" setup loss """
if 'CTC' in opt.Prediction:
    # if opt.baiduCTC:
    #     need to install warpctc. see our guideline.
    #     from warpctc_pytorch import CTCLoss
    #     criterion = CTCLoss()
    # else:
    criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
else:
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0
# loss averager
loss_avg = Averager()
```
```
在 parser 参数解析器中已删除所有选项的 required 参数
```
### 新添加：
```
# -------------------------------------- PM ----------------------------------------- #
cfg.train_data = "./lmdb/training"
cfg.valid_data = "./lmdb/validation"
cfg.Transformation = "TPS"  # [ None | TPS ]
cfg.FeatureExtraction = "ResNet"  # [ VGG | ResNet ]
cfg.SequenceModeling = "BiLSTM"  
cfg.Prediction = "Attn"  # [ CTC | Attn ]
cfg.num_iter = 10000
cfg.valInterval = 50
cfg.FT = True

with open('./ja_char.txt', 'r') as f:
    text = f.read()
    cfg.character = "0123456789.abcdefghijklmnopqrstuvwxyz" + text
# ------------------------------------- END ----------------------------------------- #
```

```test.py```, ```demo.py``` 均已做出同样的修改。
# 7. 测试结果
| Model | num_iter | Acc |
|-------|----------|-----|
| TPS-ResNet-BiLSTM-Attn-Seed1111 | 10,000 | 100.000 |
| None-VGG-BiLSTM-CTC-Seed1111 | 10,000 | 74.290 |
| None-VGG-BiLSTM-CTC-Seed1111 | 100,000 | 74.439 |

结果日志文件均在 ```result``` 文件中。