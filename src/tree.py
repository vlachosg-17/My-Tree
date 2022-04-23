import numpy as np
from tqdm import tqdm
from src.gini_index import Gini
from utils.helpers import vote

class DTree:
    """ 
    This object represents any node of 
    the tree and will contain about its children (also nodes)
    and about state and position inside the tree.
    """
    def __init__(self, depth=0, parent=None, state="root", min_size=0, max_depth=10, id=0): # X, y, max_features=None,
        # self.X = X
        # self.Y = y
        # self.max_features = max_features
        self.id = f"decision-tree-{id}"
        self.time = None
        self.parent = parent
        self.depth = depth
        self.state = state
        self.min_size = min_size
        self.max_depth = max_depth
        self.size=None
        self.leaf=None
        self.thread = None
        self.x = -1
        self.y = depth
        self.pos=(self.x, self.y)
        self.mod = 0    

    def __repr__(self):
        if self.leaf is not None:
            r = (self.depth, self.state, self.size, str(self.leaf))
            return "Node(depth:%d, state:%s, size:%d, leaf:%s)" % r
        else:
            r = (self.depth, self.state) 
            return "Node(depth:%d, state:%s)" % r
    
    def __call__(self, X):
        return self.predict(X)

    def grow(self, X, y, features=None):
        # if self.state == "root":
        #     self.time = tqdm(ncols=70, leave=False)
        # else:
        #     self.time = self.parent.time
        # self.time.update(1)
        self.size = X.shape[0]
        self.leaf = self.is_terminal(y, self.min_size, self.max_depth)
        if not self.leaf:
            self.label = None
            self.children, Xy = self.best_split(X, y, features)
            for child, (xs, ys) in zip(self.children, Xy):
                child.grow(xs, ys)
        else:
            self.label = vote(y)
            self.children = []
        return self

    def best_split(self, X, y, features=None):
        if features is None: features = list(range(X.shape[1]))
        self.gini_prob, self.thres, self.var, (left, right) = \
            Gini(X, y).optimal_split(state="list", features=features)
        if len(left["X"]) != 0 and len(right["X"]) != 0:
            self.left_child = DTree(self.depth+1, parent=self, state="left", min_size=self.min_size, max_depth=self.max_depth)
            self.right_child = DTree(self.depth+1, parent=self, state="right", min_size=self.min_size, max_depth=self.max_depth)
            return [self.left_child, self.right_child], [(left["X"], left["y"]), (right["X"], right["y"])]
        else:
            if len(left["X"]) == 0:
                self.left_child = None
                self.right_child = DTree(self.depth+1, parent=self, state="right", min_size=self.min_size, max_depth=self.max_depth)
                return [self.right_child], [(right["X"], right["y"])]
            if len(right["X"]) == 0:
                self.left_child = DTree(self.depth+1, parent=self, state="left", min_size=self.min_size, max_depth=self.max_depth)
                self.right_child = None
                return [self.left_child], [(left["X"], left["y"])]    

    def is_terminal(self, y, min_size, max_depth):
        if all(y == vote(y)):
            return True

        if y.size == 0:
            return True
        
        if self.size <= min_size:
            return True

        if self.depth >= max_depth:
            return True

        return False
    
    @ staticmethod
    def is_left(x, node):
        if isinstance(node.thres, list):
            if x[node.var] in node.thres: return True
            else: return False
        else:
            if x[node.var] < node.thres: return True
            else: return False

    def pred(self, node, x):
            if node.leaf:
                return node.label
            else:
                if self.is_left(x, node):
                    if node.left_child is not None:
                        return self.pred(node.left_child, x)
                    else:
                        return self.pred(node.right_child, x)
                else:
                    if node.right_child is not None:
                        return self.pred(node.right_child, x)
                    else:
                        return self.pred(node.left_child, x)

    def predict(self, X):
        if X.ndim == 1: X = np.expand_dims(X, axis=0)
        return np.array([self.pred(self, x) for x in X])


