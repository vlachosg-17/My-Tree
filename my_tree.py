import numpy as np
from tqdm import tqdm
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from helpers import load_mushroom, load_iris, load_abalone, numerize_class, split_sets
from metric import *
from gini_index import Gini
from layouts import TreeLayout
from helpers import overSample, vote
# from plot_tree import layout
from plot_tree import set_layout, axes_off
import matplotlib.pyplot as plt


class Node:
    """ 
    This object represents any node of 
    the tree and will contain about its childern (also nodes)
    and about state and position inside the tree.
    """
    def __init__(self, data, labels, depth):
        self.X = data
        self.Y = labels
        self.depth = depth
        self.size = data.shape[0]
        self.state = None # left or right node
        self.label = None # leaf is False then labels is None
        self.leaf = False 
        self.thres, self.gini, self.var = None, None, None
        self.parent = None
        self.left_child, self.right_child = None, None
        self.right_data, self.left_data = None, None
        self.dict = {"parent": self}

    def __repr__(self):
        r = (self.depth, self.size, str(self.leaf))
        return "Node(depth=%d, size=%d, leaf=%s)" % r

    def best_split(self, features=None):
        if features is None: features = list(range(self.X.shape[1]))
        T = Gini(self.X, self.Y).optimal_split(features=features, state="list")
        self.gini, self.thres, self.var, (self.left_data, self.right_data) = T
        self.left_child = Node(self.left_data["X"], self.left_data["y"], self.depth+1)
        self.right_child = Node(self.right_data["X"], self.right_data["y"], self.depth+1)
        self.left_child.state, self.right_child.state = "left", "right"
        self.left_child.parent, self.right_child.parent = self, self
        return self.left_child, self.right_child

class Tree:
    """ 
    This object contain all the subtrees of the 
    tree (parent, left child, right child) along with infromation
    about the position of the subtree and inforation that will help 
    us to plot the tree. This will help to mostly plot the tree because
    we need to arange the subtrees to the right position in order to
    not intercept.
    """
    def __init__(self, tree, parent=None, depth=0, number=1):
        self.x = -1
        self.y = depth
        self.mod = 0
        self.subtree = tree
        self.node = tree["parent"]
        self.children = [Tree(t, self, depth+1) for key, t in tree.items() if key is not "parent"]
        if len(self.children) == 0:
            self.left_child, self.right_child = False, False
        else:    
            self.left_child, self.right_child = self.children
        self.parent = parent
        self.thread = None
        self.offset = 0
        self.ancestor = self
        self.change = self.shift = 0
        self._lmost_sibling = None
        self.number = number
    
    def __repr__(self):
        if len(self.children) != 0:
            z = [self.node] + [child.node for child in self.children]
        else:
            z = [self.node]+[None]*2
        return "[p:%s, l: %s, r: %s]" % tuple(z)


class MyDecisionTreeClassifier:
    def __init__(self, X, y, max_depth = float("inf"), node_size=20, max_features=None, folds=None, label=1):
        self.type = "decisiontree"
        self.tree_label = label
        self.max_depth = max_depth
        self.max_num_nodes = 1 + max_depth * (max_depth - 1)
        self.K = folds
        self.node_size = node_size
        if max_features is None: max_features = X.shape[1]
        self.features = np.sort(np.random.choice(range(max_features), size=max_features, replace=False))
        self.cv_error, self.trian_error = None, None
        if self.K is not None:
            self.cv_error = self.cross_validation(X, y, k=self.K)
        self.tree = self.fit_(X, y)
        self.train_error = accuracy_score(myconfusion_matrix(y, self.predict(X)))
       
    def __repr__(self):
        if self.max_depth == float("inf"): self.max_depth=int(1000)
        r = (self.tree_label, self.max_depth, self.node_size)
        return "DecisionTree(tree: %d, max_depth: %d, minibucket: %d)" % r
        
    def terminal(self, node):
        node.leaf = True
        return vote(node.Y)
        
    def is_terminal(self, node):
        if all(node.Y == vote(node.Y)):
            node.leaf = True
            return True

        if node.Y.size is 0:
            node.leaf = True
            return True
        
        if node.size <= self.node_size:
            node.leaf = True
            return True

        return False

    def grow(self, node, store, bar=None):
        if bar is not None: bar.update(1)
        left_child, right_child = node.best_split(features=self.features)
        store["left"], store["right"] = left_child.dict, right_child.dict

        if node.depth >= self.max_depth:
            left_child.label = self.terminal(left_child)
            store["left"] = left_child.dict
            right_child.label = self.terminal(right_child)
            store["right"] = right_child.dict
            return 
        
        if self.is_terminal(left_child):
            left_child.label = self.terminal(left_child)
            store["left"] = left_child.dict
        else:
            self.grow(left_child, store["left"], bar=bar)
        
        if self.is_terminal(right_child):
            right_child.label = self.terminal(right_child)
            store["right"] = right_child.dict
        else:
            self.grow(right_child, store["right"], bar=bar)
         
    def fit_(self, data, labels):
        self.root = Node(data, labels, depth=0)
        tree = {"parent": self.root}
        bar=tqdm(total=self.max_num_nodes, ncols=50, leave=False)
        self.grow(self.root, tree, bar=bar)
        tree = Tree(tree)
        return tree

    def fit(self, data, labels, label=1):
        return MyDecisionTreeClassifier(
                                X=data, 
                                y=labels, 
                                max_depth=self.max_depth, 
                                node_size=self.node_size,
                                label=label, 
                                folds=self.K
                                )

    def iter_kfolds(self, data, labels, k):
        data, labels = shuffle(data, labels)
        num_obs_fold = data.shape[0]//k
        for i in range(1, k):
            train_X, train_y = data[i*num_obs_fold:], labels[i*num_obs_fold:]
            test_X, test_y = data[:i*num_obs_fold], labels[:i*num_obs_fold]
            yield (train_X, train_y), (test_X, test_y)
    
    def cross_validation(self, data, labels, k=10):
        acc = []
        for (trX, trY), (teX, teY) in self.iter_kfolds(data, labels, k=k):
            self.tree = self.fit_(trX, trY)
            predY = self.predict(teX)
            cm = myconfusion_matrix(teY, predY)
            acc.append(accuracy_score(cm))
        
      
        return sum(acc) / len(acc)

    def pred(self, node, obs):
        if node.node.leaf:
            return node.node.label
        else:
            if isinstance(node.node.thres, list):
                if obs[node.node.var] in node.node.thres:
                    return self.pred(node.left_child, obs)   
                else:
                    return self.pred(node.right_child, obs)  
            else:
                if obs[node.node.var] < node.node.thres:
                    return self.pred(node.left_child, obs)   
                else:
                    return self.pred(node.right_child, obs)
    
    def predict(self, X):
        if X.ndim == 1: X = np.expand_dims(X, axis=0)
        return [self.pred(self.tree, x) for x in X]

if __name__ == "__main__":
    # #################### Fit tree with Iris Data ##########################
    # data, labels = load_iris("data/iris.data", shuffle_data=True, seed=0)
    # (trainX, trainY), (teX, teY) = split_sets(data, labels, test_ratio=0.2)
    # # evaluate the dicision with my tree
    # dt = MyDecisionTreeClassifier(trainX, trainY, max_depth=15, node_size=10, folds=10)
    # mypredY = dt.predict(teX)
    # m = metrics(mypredY, teY, model=dt)
    # print_metrics(m)
    # # evaluate the decision tree with sklearn tree
    # tr = DecisionTreeClassifier(max_depth=15, min_samples_split=10)
    # tr.fit(trainX, trainY)
    # skpredY = tr.predict(teX)
    # m = metrics(skpredY, teY)
    # print_metrics(m)

    # # plot the decision my tree
    # tree = TreeLayout().layout(dt.tree)
    # ax = axes_off(set_layout(tree))
    # plt.show()
    #######################################################################
    
    #################### Fit tree with Mussroom Data ######################
    data, labels = load_mushroom("data/agaricus-lepiota.data")
    (trainX, trainY), (teX, teY) = split_sets(data, labels, test_ratio=0.2)

    # evaluate the dicision with my tree
    dt = MyDecisionTreeClassifier(trainX, trainY, max_depth=10, node_size=round(trainX.shape[0]/10), folds=10)
    predY = dt.predict(teX)
    m = metrics(predY, teY, model=dt)
    print_metrics(m)

    # plot the decision my tree
    tree = TreeLayout().layout(dt.tree)
    ax = axes_off(set_layout(tree))
    plt.show()

    #######################################################################

    #################### Fit tree with Abalone Data #######################
    # data, labels = load_abalone("data/abalone19.dat")
    # (trainX, trainY), (teX, teY) = split_sets(data, labels, test_ratio=0.2)
    # # print((trainY=="positive").sum(), (trainY=="negative").sum())
    # trainX, trainY = overSample(trainX, trainY,\
    #                 k=(trainY=="negative").sum()-(trainY=="positive").sum(),
    #                 l="positive")
    # # evaluate the dicision with my tree
    # dt = MyDecisionTreeClassifier(trainX, trainY, max_depth=10, node_size=round(trainX.shape[0]/10))
    # predY = dt.predict(teX)
    # m = metrics(predY, teY, model=dt)
    # print_metrics(m)
    # # plot the decision my tree
    # tree = TreeLayout().layout(dt.tree)
    # ax = axes_off(set_layout(tree))
    # plt.show()
    ########################################################################
    
    
    