import urllib.request
import gzip
import numpy as np

url_base = 'http://yann.lecun.com/exdb/mnist/'
key_dict = {
    'train_img':'train-images-idx3-ubyte.gz',
    'train_label':'train-labels-idx1-ubyte.gz',
    'test_img':'t10k-images-idx3-ubyte.gz',
    'test_label':'t10k-labels-idx1-ubyte.gz'
}
save_dir = '../data'

def download_mnist():
    for k, v in key_dict.items():
        print(f'Downloading {k}...')
        file_path = save_dir + '/' + v
        urllib.request.urlretrieve(url_base + v, file_path)
    print('Done!')

def load_mnist():
    file_path = save_dir + '/' + key_dict['train_img']
    with gzip.open(file_path, 'rb') as f:
        x_train = np.frombuffer(f.read(), np.uint8, offset=16)
    x_train = x_train.reshape([-1, 784])

    file_path = save_dir + '/' + key_dict['test_img']
    with gzip.open(file_path, 'rb') as f:
        x_test = np.frombuffer(f.read(), np.uint8, offset=16)
    x_test = x_test.reshape([-1, 784])
    
    file_path = save_dir + '/' + key_dict['train_label']
    with gzip.open(file_path, 'rb') as f:
        y_train = np.frombuffer(f.read(), np.uint8, offset=8)
    y_train = np.eye(10)[y_train]
    
    file_path = save_dir + '/' + key_dict['test_label']
    with gzip.open(file_path, 'rb') as f:
        y_test = np.frombuffer(f.read(), np.uint8, offset=8)
    y_test = np.eye(10)[y_test]
    
    return x_train, y_train, x_test, y_test

    