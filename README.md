# io.py

reads csv files and gives back a dict where each column name is a key and the values are numpy arrays. missing values get replaced with NaN by default but you can change that.

**read_csv(filepath, delimiter=",", has_header=True, fill_value=np.nan, skip_rows=0, dtype=None, chunk_size=None)**

```python
from numcompute.io import read_csv

data = read_csv("students.csv")
print(data["grade"])

data = read_csv("data.tsv", delimiter="\t")

data = read_csv("data.csv", fill_value=0.0)

for chunk in read_csv("big.csv", chunk_size=1000):
    print(chunk["price"].mean())
```

---

# preprocessing.py

scalers and encoders for cleaning up data. all of them have fit(), transform() and fit_transform() so they work the same way.

## StandardScaler

makes each column have mean 0 and std 1. wont break on constant columns or NaN values.

```python
from numcompute.preprocessing import StandardScaler

sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)
X_original = sc.inverse_transform(X_train_scaled)
```

## MinMaxScaler

scales everything into a range, default is [0, 1]. handles NaN and constant columns fine.

```python
from numcompute.preprocessing import MinMaxScaler

mm = MinMaxScaler(feature_range=(0, 1))
X_scaled = mm.fit_transform(X_train)
X_test_scaled = mm.transform(X_test)
```

## OneHotEncoder

converts categorical columns into binary columns. one column per unique value seen during fit().

```python
from numcompute.preprocessing import OneHotEncoder
import numpy as np

X = np.array([["cat"], ["dog"], ["cat"], ["bird"]])
enc = OneHotEncoder()
X_encoded = enc.fit_transform(X)
print(enc.get_feature_names_out(["animal"]))
```

---

# pipeline.py

chains preprocessing steps together so you dont have to call fit and transform on each one separately. throws an error early if something is wrong with the steps.

```python
from numcompute.pipeline import Pipeline
from numcompute.preprocessing import StandardScaler, OneHotEncoder

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("encode", OneHotEncoder()),
])

X_out = pipe.fit_transform(X_train)
X_test_out = pipe.transform(X_test)
pipe.set_params(scale__copy=False)
```

can also put a model at the end:

```python
pipe = Pipeline([
    ("scale", StandardScaler()),
    ("model", MyClassifier()),
])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
```
