import torch
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

torch.cuda.device(0)

def data_load():
    train_set = torchvision.datasets.ImageFolder(
        root='./TinyImageNet/train',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
        ])
    )

    test_set = torchvision.datasets.ImageFolder(
        root='./TinyImageNet/val',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
        ])
    )

    train_loader = DataLoader(train_set, shuffle=True, batch_size=50, num_workers=8)

    test_loader = DataLoader(test_set, shuffle=True)

    return train_loader, test_loader

def new_plot(title, xlabel, ylabel, data_1, data_2, data_1_label, data_2_label):
    plt.figure(figsize=(10,5))
    plt.suptitle(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(data_1, label=data_1_label)
    plt.plot(data_2, label=data_2_label)
    plt.legend()
    
    plt.show()



def main():
    train_loader, test_loader = data_load()
    epochs = 20
    PATH = './TinyImageNet/result2.pt'
    load_model = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(torch.cuda.get_device_name(device))
    
    train_loss_list = []
    val_loss_list = []
    train_accuracy_list = []
    val_accuracy_list = []

    model = VGG_Net()
    if load_model:
        model.load_state_dict(torch.load(PATH))
        model = model.to(device)
    else:
        model = model.to(device)

        criterion = torch.nn.CrossEntropyLoss().cuda()
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)

        for epoch in range(epochs):
            running_loss = 0.0
            total_train = 0
            correct_train = 0
            
            for i, img in enumerate(train_loader, 0):
                inputs, labels = img
                inputs, labels = inputs.to(device), labels.to(device)

                opt.zero_grad()
                y_pred, feature = model(inputs)

                loss = criterion(y_pred, labels)

                loss.backward()
                opt.step()
                
                _, predicted = torch.max(y_pred, 1)
                total_train += labels.size(0)
                correct_train += predicted.eq(labels.data).sum().item()

                running_loss += loss.item()
                train_accuracy = (correct_train / total_train) * 100
                
                if i % 50 == 49:
                    print('[%d, %5d] loss: %.3f  accuracy : %.3f' %
                          (epoch + 1, i + 1, running_loss / 50, train_accuracy))
                    running_loss = 0.0
                
                train_accuracy_list.append(train_accuracy)
                train_loss_list.append(running_loss)
                
                
            print("validation ===============================")
            correct_val = [0] * 100
            total_val = [0] * 100
            val_loss = 0
            count = 0

            with torch.no_grad():
                for test_data in test_loader:
                    img, labels = test_data
                    img, labels = img.to(device), labels.to(device)

                    out, _ = model(img)
                    _, pred = torch.max(out, 1)
                    pred_item = pred.item()
                    label_item = labels.item()
                    pred_acc = (pred_item == label_item)
                    if pred_acc:
                        correct_val[label_item] += 1
                    total_val[label_item] += 1
                    
                    loss = criterion(out, labels)
                    val_loss += loss.item()
                    
                    count+=1
                    
            val_loss /= count

            accuracy_sum = 0
            for i in range(100):
                temp = 100 * correct_val[i] / total_val[i]
                accuracy_sum += temp
            print('Validation Accuracy: ', accuracy_sum / 100, "Validation Loss: ", val_loss)
            
            val_accuracy_list.append(accuracy_sum)
            val_loss_list.append(val_loss)
            
            print("==========================================")
            
    correct = [0]*100
    total = [0]*100

    with torch.no_grad():
        for test_data in test_loader:
            img, labels = test_data
            img = img.cuda()

            out, _ = model(img)
            _, pred = torch.max(out, 1)
            pred_item = pred.item()
            label_item = labels.item()
            pred_acc = (pred_item == label_item)
            if pred_acc:
                correct[label_item] += 1
            total[label_item] += 1

    accuracy_sum = 0
    for i in range(100):
        temp = 100 * correct[i] / total[i]
        print('Accuracy of %5s : %2d %%' % (
            i, temp))
        accuracy_sum += temp
    print('Accuracy average: ', accuracy_sum / 100)

    new_plot('Train Loss & Validation Loss', 'epochs', 'Traing loss', train_loss_list, val_loss_list, 'train', 'validation')
    new_plot('Train Accuracy & Validation Accuracy', 'epochs', 'Accuracy', train_accuracy_list, val_accuracy_list, 'train', 'validation')


if __name__ == '__main__':
    main()
