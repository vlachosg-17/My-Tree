import numpy as np
from helpers import load_iris, load_mushroom
from tqdm import tqdm

class Gini:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.classes = np.sort(np.unique(y))
        self.types=[float, np.float16, np.float32, np.float64,\
                    int, np.int32, np.int16, np.int8, np.int64]
        

    def powerset(self, s, remove=True):
        a = []
        x = len(s)
        for i in range(1, 2**x+1):
            a.append([s[j] for j in range(x) if (i & (2**j))])
        
        if remove:
            return a[:-2]
        else:
            return a

    def make_thresholds(self, x):
        def tp(x):
            return type(x[0])
        if tp(x) in self.types:
            v = np.sort(np.unique(x))
            thrs = np.array([(v[k-1]+v[k]) / 2 for k in range(1, len(v))])
        else: 
            thrs = self.powerset(np.unique(x))   
        return thrs 

    def make_groups(self, x, thresh):
        def reverse_bool(vector):
            return [not v for v in vector]

        if isinstance(thresh, float):
            t = np.array([True if x_i < thresh else False for x_i in x])
        else:
            t = np.array([True if x_i in thresh else False for x_i in x])
        return [np.where(t), np.where(reverse_bool(t)), thresh]

    def gini(self, D):
        """
        C = classes: the unique labels of the dataset
        Gini(D) = 1 - sum_{i}^{|C|}(p_i)^2
        p_i = #(D == i) / |D|, i=0,...,|C|
        """
        prop_sum = 0.0
        keys, counts = np.unique(D, return_counts=True)
        for _, count in zip(keys, counts):
            p = count/len(D)
            prop_sum += p * p
        return 1.0 - prop_sum

    def gini_for_thrs(self, groups, y):
        """
        Gini_{A}(D) = (|D_1| / |D|) * Gini(D_1) + (|D_2| / |D|) * Gini(D_2)
        """
        left_idx, right_idx, thresh = groups
        left_labs, right_labs = y[left_idx], y[right_idx]
        coef = left_labs.shape[0]/(left_labs.shape[0]+right_labs.shape[0])
        return {
            "gini": coef * self.gini(left_labs) + (1 - coef) * self.gini(right_labs), 
            "thres": thresh,
            "obs": groups
            }   

    def optimal_split(self, features=None, state="dict"):
        if features is None: features = list(range(self.X.shape[1]))
        best_gini, best_groups, best_var, best_thres = float("inf"), None, None, None
        for i, attr in enumerate(self.X[:, features].T):
            thrs = self.make_thresholds(attr)
            for thr in thrs:
                f = self.gini_for_thrs(groups=self.make_groups(attr, thr), y=self.y)
                if f["gini"] < best_gini: # choose the smallest gini possible
                    best_gini, best_thres, best_var = f["gini"], thr, features[i] 
                    best_groups=[{"X": self.X[f["obs"][0]], "y": self.y[f["obs"][0]]}, {"X": self.X[f["obs"][1]], "y": self.y[f["obs"][1]]}]
        
        if state is "dict":
            return {"gini": best_gini, "thres": best_thres, "var": best_var, "groups": best_groups}
        else:
            return best_gini, best_thres, best_var, best_groups 

if __name__ == "__main__":
    P = np.array([[1,'A',2,3,6,8,0],
                  [9,'A',4,1,5,7,0],
                  [3,'B',5,3,6,4,1],
                  [8,'C',3,8,4,6,0],
                  [2,'C',2,5,5,9,1],
                  [5,'A',9,4,6,7,0],
                  [4,'C',3,9,1,2,1],
                  [9,'B',7,6,1,4,1],
                  [0,'A',1,9,9,8,1]],dtype = object)
    # print(P[0, 0], type(P[0, 0]), type(P[0, 0]) in [float, int])
    # print(Gini(P[:, :-1], P[:, -1]).optimal_split())
    
    data, labels = load_mushroom("data/agaricus-lepiota.data")
    # m=np.random.choice(range(data.shape[1]), size=2*data.shape[1]//3, replace=False)
    # print(best_gini_split(data[:, m], labels))
    print(Gini(data, labels).optimal_split())

    data, labels = load_iris("data/iris.data", shuffle_data=True)
    # print(Gini(data, labels).optimal_split())
    


