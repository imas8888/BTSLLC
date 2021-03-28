import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import sys
from joblib import load

model = load('LLC.pkl')

for line in sys.stdin:
    inputD = np.array([float(i) for i in line.split(",")])[-50:].reshape(1,-1)
    print(int(model.predict(inputD)[0]))
