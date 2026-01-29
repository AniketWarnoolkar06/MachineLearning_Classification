import pandas as pd

def load_bank_data(path):
    return pd.read_csv(path, sep=";")
