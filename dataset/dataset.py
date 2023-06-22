import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class MyDataset(Dataset):
    def __init__(self, csv_path):
        super(MyDataset, self).__init__()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        img = Image.open(self.df[self.df['id'] == item + 1]['path'].item())
        img = self.transform(img)
        label = self.df[self.df['id'] == item + 1]['label'].item()

        return img, label

