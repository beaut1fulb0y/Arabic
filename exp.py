from dataset import MyDataset

dt = MyDataset('data/test.csv')
img, label = dt[0]
print(label)