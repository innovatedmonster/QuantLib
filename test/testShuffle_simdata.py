from torch.utils.data import Dataset, DataLoader


class DataSet(Dataset):
    def __init__(self, n):
        self.n = n
        self.data = [i for i in range(n)]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.data[idx]


data_set = DataSet(16)
data_loader = DataLoader(data_set, batch_size=8, shuffle=True)
for num in data_loader:
    print(num, end='\t')
print()
for num in data_loader:
    print(num, end='\t')
print()
for num in data_loader:
    print(num, end='\t')
print()
