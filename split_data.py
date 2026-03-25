"""
Split licence_plate_price_cleaned.csv → train.csv (80%) / test.csv (20%)
"""

import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('licence_plate_price_cleaned.csv')

train, test = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)

print(f"Total : {len(df)}")
print(f"Train : {len(train)} → train.csv")
print(f"Test  : {len(test)}  → test.csv")
