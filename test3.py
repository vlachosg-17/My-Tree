from matplotlib import pyplot as plt

from utils.helpers import load_abalone, split_sets, over_sample
from graph_tree.plot_tree import dt_graph
from src.tree import DTree
from src.random_forest import RForest
from utils.metric import *

data, labels = load_abalone("data/abalone19.dat")
(trainX, trainY), (teX, teY) = split_sets(data, labels, test_ratio=0.5)
ostrainX, ostrainY = over_sample(trainX, trainY, k=(trainY=="negative").sum()-(trainY=="positive").sum(), l="positive")

tree = DTree(min_size=20, max_depth=15)
tree = tree.grow(ostrainX, ostrainY)
predY = tree.predict(teX)
m = metrics(predY, teY)
print_metrics(m)
# dt_graph(tree)
# plt.show()

rf = RForest(ntrees=50, min_size=1, max_depth = 30, n_vars=6)
rf = rf.fit(ostrainX, ostrainY)
predY = rf.predict(teX)
m = metrics(predY, teY)
print_metrics(m)
