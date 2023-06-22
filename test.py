import os
import torch
import argparse
import pandas as pd
from dataset import MyDataset
from torch.utils.data.dataloader import DataLoader
from models import ResNet

# set environment parser
Parser = argparse.ArgumentParser()
Parser.add_argument("-b", "--batch_size", default=512, type=int, help="batch size")
Parser.add_argument("-d", "--device", default="cpu", type=str, help="device")
Parser.add_argument("-s", "--save_path", default="runs", type=str, help="save path")
Parser.add_argument("-w", "--num_workers", default=8, type=int, help="number of workers")
args = Parser.parse_args()

if __name__ == '__main__':
    test_csv = os.path.join('data', 'test.csv')
    test_dataset = MyDataset(test_csv)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size,
                                 num_workers=args.num_workers)
    data = {'label': [], 'pred': []}

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    elif device == "cuda":
        device = "cuda:0"

    dnn_path = os.path.join('runs', 'best.pth')

    model = ResNet()
    state_dict = torch.load(dnn_path, map_location)
    model.load_state_dict(state_dict=state_dict)
    model.to(device)

    s = 0

    for _, (imgs, labels) in enumerate(test_dataloader):
        model.eval()
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs)
        labels = labels.cpu().numpy().tolist()
        preds = torch.max(preds, 1)[1].cpu().numpy().tolist()
        l = len(labels)
        for i in range(l):
            if labels[i] != preds[i]:
                s += 1
        data['label'].extend(labels)
        data['pred'].extend(preds)
    df = pd.DataFrame(data)
    df.to_csv('test.csv')
    print((3360 - s) / 3360)
