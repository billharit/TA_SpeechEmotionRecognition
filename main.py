import os
import pandas as pd
from dataset import load_to_dataframe

train_df, test_df = load_to_dataframe('dataset/train', 'dataset/test')
