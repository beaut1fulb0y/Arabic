import os
import pandas as pd

train_path = os.path.join('data', 'train')
test_path = os.path.join('data', 'test')

if __name__ == '__main__':
    for this_path in [train_path, test_path]:
        real_path = os.path.join('..', this_path)
        data = {'id': [], 'path': [], 'label': []}
        for _, dirs, files in os.walk(real_path):
            for file in files:
                file_path = os.path.join(this_path, file)
                file_name = file.split('.')[0]
                file_id, file_label = file_name.split('_')[1], file_name.split('_')[3]
                data['id'].append(int(file_id))
                data['path'].append(file_path)
                data['label'].append(int(file_label) - 1)
        df = pd.DataFrame(data)
        df = df.sort_values('id')

        if this_path == train_path:
            df.to_csv(os.path.join('..', 'data', 'train.csv'))
        else:
            df.to_csv(os.path.join('..', 'data', 'test.csv'))
