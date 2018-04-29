import os
import glob
import re


def clean_input_files(datapath):
    for filename in glob.glob(os.path.join(datapath, '*.csv')):
        with open(filename, 'r+') as f:
            text = f.read()
            text = re.sub('Epoch ','',text)
            text = re.sub('Train accuracy ', '', text)
            text = re.sub('Test accuracy', '', text)
            text = re.sub(' ', '', text)
            f.seek(0)
            f.write(text)
            f.truncate()


if __name__ == '__main__':
    datapath = '../data'
    clean_input_files(datapath)
