"""Download and preprocess the 23 OpenML data sets into data/{name}.npz."""
import os, json, warnings, numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from paths import DATA
warnings.filterwarnings('ignore')
IDS = {'balance': 11, 'banknote': 1462, 'blood': 1464, 'climate': 1467, 'diabetes': 37, 'heart': 53, 'ilpd': 1480,
       'ionosphere': 59, 'iris': 61, 'kc1': 1067, 'magic': 1120, 'optdigits': 28, 'pc1': 1068, 'phoneme': 1489,
       'qsar': 1494, 'segment': 36, 'sonar': 40, 'spambase': 44, 'steel': 1504, 'vehicle': 54, 'wdbc': 1510,
       'wilt': 1570, 'wine': 187}

if __name__ == '__main__':
    os.makedirs(DATA, exist_ok=True); meta = {}
    for name, i in IDS.items():
        d = fetch_openml(data_id=i, as_frame=True, data_home=os.path.join(DATA, 'cache'))
        X = d.data.apply(lambda c: c.astype('category').cat.codes if str(c.dtype) in ('category', 'object') else c).astype(float)
        m = X.notna().all(axis=1); X = X[m].values; y = d.target[m].astype('category').cat.codes.values
        keep = np.where(np.bincount(y)[y] >= 10)[0]; X, y = X[keep], y[keep]   # drop classes with < 10 examples
        if name == 'steel': X = np.delete(X, range(27, 33), axis=1)              # these six fault flags determine the label
        if len(y) > 600: X, _, y, _ = train_test_split(X, y, train_size=600, stratify=y, random_state=0)
        np.savez(f'{DATA}/{name}.npz', X=X, y=y); meta[name] = dict(n=len(y), d=X.shape[1], k=int(y.max() + 1)); print(name, meta[name], flush=True)
    json.dump(meta, open(f'{DATA}/meta.json', 'w'), indent=1)
