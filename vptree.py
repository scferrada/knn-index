import numpy as np
from heapq import heappush, heappushpop

'''
This Class holds the nodes of a vantage-point tree.
It contains pointers to the left and right childs,
the vantage-point that characterizes the node,
and boundaries of the distances of the elements of the subtree
with respect to the parent node's vantage-point
'''
class Node:

	def __init__(self, L, R, vp, mu, minp, maxp):
		self.L = None
		self.R = None
		self.vp = vp
		self.mu = mu
		self.minp = minp
		self.maxp = maxp
	
	def is_leaf(self):
		return False
		
class Leaf:
	def __init__(self, points, minp, maxp)
		self.points = points
		self.minp = minp
		self.maxp = maxp
	
	def is_leaf(self):
		return True

def manhattan(D, x):
	return np.sum(np.abs(D-x), axis=1)
	
def manhattan2(x, y):
	return np.sum(np.abs(y[1:]-x[1:]))
	
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
	P = np.random.choice(data, np.ceil(len(data)*0.05), False)
	best_spread = 0
	vp = P[0]
	for pi in P:
		D = np.random.choice(data, np.ceil(len(data)*0.05), False)
		distances = distance(D[:,1:], pi[1:])
		mu = np.median(distances)
		spread = np.std(distances - mu)
		if spread > best_spread:
			best_spread = spread
			vp = pi
	return vp, np.vstack((data[:vp[0]], data[vp[0]+1:])) #no funciona
		
'''
This method buids a vantage-point tree, given a numpy matrix.
It also receives the number of vectors that can be stored at the leaves (1 by default)
'''
def build_tree(data, c=1, distance=manhattan):
	idx = np.arange(len(data)).reshape(len(data), 1)
    matrix = np.hstack((idx, data))
	return build_tree_internal(matrix, c, distance)

def build_tree_internal(data, c=1, distance=manhattan, minP=0, maxP=0):
	if len(data) <= c:
		return Leaf(data, minP, maxP)
	vp, data = select_vp(data, distance)
	distances = distance(data[:,1:], vp[1:])
	mu = np.median(distances)
	_minL = np.min(distances[distances < mu]
	_maxL = np.max(distances[distances < mu]
	_minR = np.min(distances[distances >= mu]
	_maxR = np.max(distances[distances >= mu]
	L = build_tree_internal(data[distances < mu], c, distance, _minL, _maxL)
	R = build_tree_internal(data[distances >= mu], c, distance, _minR, _maxR)
	return Node(L, R, vp, mu, minP, maxP)
	
'''
This method searches for the object k nearest neighbours of 
x on a vp-tree starting at node n
'''
def search(n, x, k, distance=manhattan2):
	return search_internal(n, x, k, distance, [])
	
def search_internal(n, x, k, distance, results):
	if n.is_leaf():
		for p in n.points:
			d = distance(x, p)
			if len(results) < k:
				heappush(results, Res(p[0], d))
			elif d < results[0].dist:
				heappushpop(results, Res(p[0], d)
		#return results
	d = distance(n.vp, x)
	if len(results) < k:
		heappush(results, Res(p[0], d))
	elif d < results[0]:
		heappushpop(results, Res(p[0], d)		
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
	#return results