
import torch.nn as nn
import torch.optim as optim
import net
from datapre import *
from Myutils import *
from Test import predict_class

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 14
dataset = InvoiceDataset(img_dir, label_dir)
TestDataset = InvoiceDataset(img_dir1,label_dir)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
dataloader1 = DataLoader(TestDataset, batch_size=1, shuffle=False)

def train(classifier):
    # 根据发票版本数量调整

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    for epoch in range(50):
        for i, (images, labels) in enumerate(dataloader):



            images = images.to(device)
            labels = torch.tensor(labels).to(device)



            optimizer.zero_grad()
            outputs = classifier(images).to(device)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch + 1}/{50}], Loss: {loss.item():.4f}")
    save_model(classifier, "InvoiceClassifier1.pt")
    return classifier

def main():
  init_random_seed(None)

  model_new = init_model(net.InvoiceClassifier(num_classes).to(device), restore="abc/InvoiceClassifier1.pt")
  # model_new = train(model)
  model_new.eval()

  correct = 0
  total = 0

  result = predict_class().to(device)
  print('分类结果为：', result)
  # with torch.no_grad():
  #     for images, labels in dataloader1:
  #         images, labels = images.to(device), labels.to(device)
  #         outputs = model_new(images)
  #         _, predicted = torch.max(outputs.data, 1)
  #         total += labels.size(0)
  #         correct += (predicted == labels).sum().item()
  #
  #         # 打印预测标签和实际标签
  #         for pred, actual in zip(predicted, labels):
  #             print(f"Predicted: {pred.item()}, Actual: {actual.item()}")
  #
  #
  #
  #
  #
  # print(f"Test Accuracy: {100 * correct / total:.2f}%")




# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    main()


