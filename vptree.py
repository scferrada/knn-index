import numpy as np
from heapq import heappush, heappushpop
import matplotlib.pyplot as plt

'''
This Class holds the nodes of a vantage-point tree.
It contains pointers to the left and right childs,
the vantage-point that characterizes the node,
and boundaries of the distances of the elements of the subtree
with respect to the parent node's vantage-point
'''
class Node:

	def __init__(self, L, R, vp, mu, minp, maxp):
		self.L = L
		self.R = R
		self.vp = vp
		self.mu = mu
		self.minp = minp
		self.maxp = maxp
	
	def is_leaf(self):
		return False
		
class Leaf:
	def __init__(self, points, minp, maxp):
		self.points = points
		#self.vp = points
		self.minp = minp
		self.maxp = maxp
	
	def is_leaf(self):
		return True

def manhattan(D, x):
	return np.sum(np.abs(D-x), axis=1)
	
def euclidean(D, x):
	return np.sqrt(np.sum(np.power(D - x, 2), axis=1))
	
def manhattan2(x, y):
	return np.sum(np.abs(y[1:]-x[1:]))
	
def euclidean2(x, y):
	return np.sqrt(np.sum(np.power(p2[1:] - p1[1:], 2)))
	
class Res:
    def __init__(self, obj, dist):
        self.dist = -dist
        self.obj = obj

    def __lt__(self, other):
        return self.dist > other.dist

    def __eq__(self, other):
        return self.dist == other.dist and self.obj == other.obj

    def __str__(self):
        return str(self.obj) + "; " + str(self.dist)
		
'''
chooses a better-than-random vantage-point,
attempting to assure it divides the space evenly
'''
def select_vp(data, distance):
	N = int(min(np.ceil(len(data)*0.05), 100))
	P = data[np.random.choice(data.shape[0], N, replace=False)]
	best_spread = 0
	vp = P[0]
	for pi in P:
		D = data[np.random.choice(data.shape[0], N, replace=False)]
		distances = distance(D[:,1:], pi[1:])
		mu = np.median(distances)
		spread = np.std(distances - mu)
		if spread > best_spread:
			best_spread = spread
			vp = pi
	return vp, data[data[:, 0] != vp[0]]
		
'''
This method buids a vantage-point tree, given a numpy matrix.
It also receives the number of vectors that can be stored at the leaves (1 by default)
'''
def build_tree(data, c=1, distance=euclidean):
	idx = np.arange(len(data)).reshape(len(data), 1)
	matrix = np.hstack((idx, data))
	return build_tree_internal(matrix, c, distance)

def build_tree_internal(data, c=1, distance=euclidean, minP=0, maxP=0):
	if len(data) <= c:
		return Leaf(data[0], minP, maxP)
	vp, data = select_vp(data, distance)
	distances = distance(data[:,1:], vp[1:])
	mu = np.median(distances)
	L =None
	R =None
	targetL = distances[distances < mu]
	if len(targetL) > 0:
		_minL = np.min(targetL)
		_maxL = np.max(targetL)
		L = build_tree_internal(data[distances < mu], c, distance, _minL, _maxL)
	targetR = distances[distances >= mu]
	if len(targetR) > 0:
		_minR = np.min(targetR)
		_maxR = np.max(targetR)
		R = build_tree_internal(data[distances >= mu], c, distance, _minR, _maxR)
	return Node(L, R, vp, mu, minP, maxP)
	
'''
This method searches for the object k nearest neighbours of 
x on a vp-tree starting at node n
'''
def search(n, x, k, distance=euclidean2):
	return search_internal(n, x, k, distance, [])
	
def search_internal(n, x, k, distance, results):
	if n.is_leaf():
		for p in n.points:
			d = distance(x, p)
			if len(results) < k:
				heappush(results, Res(p[0], d))
			elif d < results[0].dist:
				heappushpop(results, Res(p[0], d))
	else:
		d = distance(n.vp, x)
		if len(results) < k:
			heappush(results, Res(p[0], d))
		elif d < results[0]:
			heappushpop(results, Res(p[0], d))		
		middle = (n.L.maxp + n.R.minp)/2
		if d < middle:
			if n.L.minp - results[0].dist < d and d < n.L.maxp + results[0].dist:
				search_internal(n.L, x, k, distance, results)
			if n.R.minp - results[0].dist < d and d < n.R.maxp + results[0].dist:
				search_internal(n.R, x, k, distance, results)
		else:
			if n.R.minp - results[0].dist < d and d < n.R.maxp + results[0].dist:
				search_internal(n.R, x, k, distance, results)
			if n.L.minp - results[0].dist < d and d < n.L.maxp + results[0].dist:
				search_internal(n.L, x, k, distance, results)
	
def visualize(tree):
	points = []
	circles = []
	visualize_internal(tree, points, circles)
	ax = plt.gca()
	ax.set_xlim((0, 1))
	ax.set_ylim((0, 1))
	ax.scatter(np.array(points)[:,1],np.array(points)[:,2])
	for c in circles:
		ax.add_artist(c)
	plt.show()
	
def visualize_internal(tree, points, circles):
	if tree is None: return
	if tree.is_leaf():
		points.append(tree.points)
	else:
		points.append(tree.vp)
		print(tree.mu)
		circles.append(plt.Circle((tree.vp[1], tree.vp[2]), radius=tree.mu, fill=False))
		visualize_internal(tree.L, points, circles)
		visualize_internal(tree.R, points, circles)
	