import vptree as vp 
import numpy as np

np.random.seed(3)
data = np.random.rand(100, 2)

tree = vp.build_tree(data)

vp.visualize(tree)