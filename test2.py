from utils.helpers import load_mushroom, split_sets
from graph_tree.plot_tree import dt_graph
from matplotlib import pyplot as plt
from src.tree import DTree
from src.random_forest import RForest
from utils.metric import *

data, labels = load_mushroom("data/agaricus-lepiota.data")
(trainX, trainY), (teX, teY) = split_sets(data, labels, test_ratio=0.2)
tree = DTree(min_size=20, max_depth=8)
tree = tree.grow(trainX, trainY)
predY = tree.predict(teX)
m = metrics(predY, teY)
print_metrics(m)
# dt_graph(tree)
# plt.show()

rf = RForest(ntrees=55, min_size=10, max_depth = 30, n_vars=6)
rf = rf.fit(trainX, trainY)
predY = rf.predict(teX)
m = metrics(predY, teY)
print_metrics(m)