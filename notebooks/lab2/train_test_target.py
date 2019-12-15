import pathlib

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

path_bank_df = pathlib.Path("../data/raw/bank-full.csv")
bank_df = pd.read_csv(path_bank_df, delimiter=";")

bank_df.describe(include="all").T
bank_df.describe(include="object").T
bank_df.describe(include="number").T

bank_df.count()
bank_df.dropna().count()
le_y = LabelEncoder()

bank_df["target"] = le_y.fit_transform(bank_df["y"])
bank_df["target"].describe()

bank_data_df = bank_df.drop('y', axis=1)

set(bank_df.columns).difference(bank_data_df.columns)

X = bank_data_df.drop('target', axis=1)
y = bank_data_df['target']

seed = 111
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=seed, stratify=y)

np.mean(y_train)
np.mean(y_test)

df_train = pd.concat([X_train, pd.DataFrame({"target": y_train})], axis=1)
df_test = pd.concat([X_test, pd.DataFrame({"target": y_test})], axis=1)

df_train.to_csv(pathlib.Path("../data/interim/bank_train.csv"), index=False)
df_test.to_csv(pathlib.Path("../data/interim/bank_test.csv"), index=False)