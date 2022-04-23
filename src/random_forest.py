import numpy as np
from tqdm import tqdm

from src.tree import DTree
from utils.helpers import bootstrap_sample, vote

def oob_error(predictions, trainlabels):
    num_wrong_pred = 0
    for k in range(len(trainlabels)):
        if predictions[k] != trainlabels[k]:
            num_wrong_pred +=1
    
    return num_wrong_pred / len(trainlabels)    

class RForest:
    def __init__(self, ntrees=100, min_size=1, max_depth=15, n_vars="auto", seed=None):
        # self.X, self.y = X, y
        self.id = "random-forest"
        self.ntrees=ntrees
        self.min_size = min_size
        self.max_depth = max_depth
        self.n_vars = n_vars
        self.trees, self.data = [], []
        self.oob_score = None
        if seed is not None: np.random.seed(seed)
        else:  np.random.seed(42)
 
        
    def subsamples(self, X, y, ratio=1.0):
        for i in tqdm(range(self.ntrees), ncols=60):
            yield i, bootstrap_sample(X, y, int(len(X)*ratio))

    def fit(self, X, y):
        # self.X, self.y = X, y
        if self.n_vars is "auto": 
            if X.shape[1]%2: m = int(np.sqrt(X.shape[1]))
            else:            m = int(np.floor(np.log(X.shape[1])+1))
        elif self.n_vars is "all": m = X.shape[1]
        else: m = self.n_vars

        for i, (xs, ys) in self.subsamples(X, y):
            r = np.sort(np.random.choice(range(X.shape[1]), size=m, replace=False))
            # print(f"\nTree: {i}, {r}, {X.shape[1]}, {m}")
            tr = DTree(min_size=self.min_size, max_depth=self.max_depth, id=i).grow(xs[:, r], ys, features=r)
            self.trees.append(tr)
        return self
            
    def predict(self, X):
        if len(self.trees) is 0: raise "fit data !!!"
        preds = np.array([tree.predict(X) for tree in self.trees]).T
        return np.array([vote(p) for p in preds])


