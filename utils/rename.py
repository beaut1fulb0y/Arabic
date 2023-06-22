import os

if __name__ == '__main__':
    log_path = os.path.join('..', 'runs', '1')
    for _, dirs, files in os.walk(log_path):
        for file in files:
            name_list = file.split('.')
            name_list[-1] = str(int(name_list[-1]) + 100)
            name_list[-2] = '3768'
            file_new = '.'.join(name_list)
            src = os.path.join(log_path, file)
            dst = os.path.join(log_path, file_new)
            os.rename(src, dst)
