import vptree as vp 
import numpy as np

np.random.seed(0)
data = np.random.rand(2, 100)

tree = vp.build_tree(data)

vp.visualize(tree)