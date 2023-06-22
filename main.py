import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import MyDataset
from models import ResNet

# set environment parser
Parser = argparse.ArgumentParser()
Parser.add_argument("-b", "--batch_size", default=512, type=int, help="batch size")
Parser.add_argument("-d", "--device", default="cpu", type=str, help="device")
Parser.add_argument("-e", "--epochs", default=300, type=int, help="training epochs")
Parser.add_argument("-l", "--lr", default=0.0005, type=float, help="learning rate")
Parser.add_argument("-s", "--save_path", default="runs", type=str, help="save path")
Parser.add_argument("-w", "--num_workers", default=8, type=int, help="number of workers")
args = Parser.parse_args()


def train(model, dataloader, device, train):
    if train:
        model.train()
    else:
        model.eval()
    process_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    fc_params = list(model.resnet.fc.parameters())
    other_params = [param for name, param in model.resnet18.named_parameters() if "fc" not in name]
    optimizer = optim.Adam([
        {"params": fc_params, "lr": args.lr},
        {"params": other_params, "lr": args.lr / 10},
    ])

    criterion = nn.CrossEntropyLoss()

    running_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, labels) in process_bar:
        imgs, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        if train:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = 100 * correct / total

    return epoch_loss, epoch_accuracy


if __name__ == '__main__':
    train_csv = os.path.join('data', 'train.csv')
    test_csv = os.path.join('data', 'test.csv')
    train_dataset = MyDataset(train_csv)
    test_dataset = MyDataset(test_csv)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                                  num_workers=args.num_workers)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size,
                                 num_workers=args.num_workers)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    elif device == "cuda":
        device = "cuda:0"


    model = ResNet()
    model.to(device)
    writer = SummaryWriter()

    best = -1
    train_ac = -1
    test_ac = -1
    best_epoch = -1
    best_ac = -1

    for epoch in range(args.epochs):
        loss, accuracy = train(model, train_dataloader, device, True)
        test_loss, test_accuracy = train(model, test_dataloader, device, False)

        print(
            f"Epoch {args.epochs}/{epoch + 1}, Loss: {loss:.4f}, Accuracy: {accuracy:.2f}% Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}")
        writer.add_scalar("Training Loss", loss, epoch)
        writer.add_scalar("Training Accuracy", accuracy, epoch)
        writer.add_scalar("Test Loss", test_loss, epoch)
        writer.add_scalar("Test Accuracy", test_accuracy, epoch)

        save_root = args.save_path
        if best < test_accuracy:
            best_epoch = epoch
            best = test_accuracy
            train_ac = accuracy
            best_ac = test_accuracy
            test_ac = test_accuracy
            save_path = os.path.join(save_root, 'best.pth')
            torch.save(model.state_dict(), save_path)

        writer.close()

    print(f"train_acc: {train_ac}, best val_acc: {best_ac}, test_acc: {test_ac}, best epoch: {best_epoch}")
