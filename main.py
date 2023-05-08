
import torch.nn as nn
import torch.optim as optim
import net
from datapre import *
from Myutils import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 31
def train(classifier):
    

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    for epoch in range(50):
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = classifier(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch + 1}/{50}], Loss: {loss.item():.4f}")
    save_model(classifier, "InvoiceClassifier.pt")
    return classifier

def main():
  init_random_seed(None)

  model = init_model(net.InvoiceClassifier(num_classes).to(device), restore=None)
  model_new = train(model)
  model.eval()
  correct = 0
  total = 0
  with torch.no_grad():
      for images, labels in test_loader:
          images, labels = images.to(device), labels.to(device)
          outputs = model(images)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()

  print(f"Test Accuracy: {100 * correct / total:.2f}%")




if __name__ == '__main__':
    main()


