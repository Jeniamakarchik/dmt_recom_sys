import sklearn
import pandas as pd

from data import read_processed_train

def train_model():
    data = read_processed_train()
    print(data.head())


if __name__ == '__main__':
    pass