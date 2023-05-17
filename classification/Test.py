
import net
from datapre import *
from Myutils import *



num_classes = 14
model_new = init_model(net.InvoiceClassifier(num_classes), restore="abc/InvoiceClassifier1.pt")
if torch.cuda.is_available():
    model_new.cuda()
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

# 定义输入图片的转换方式
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # 将图片缩放到指定大小
    transforms.ToTensor(),  # 将 PIL 图像转换为 PyTorch 张量
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 定义分类函数
def predict_class():
    # 获取图片地址
    image_path = input('请输入图片地址：')
    # 加载图片
    image = Image.open(image_path)
    # 转换图片格式
    image = transform(image)
    # 将图片扩展为4维张量，第一维为batch_size
    image = image.unsqueeze(0)
    image = image.to(device)
    # 预测分类
    output = model_new(image)
    # 获取分类结果
    _, predicted = torch.max(output.data, 1)
    # 返回分类结果
    return predicted.item()

# 调用分类函数
result = predict_class()
print('分类结果为：', result)