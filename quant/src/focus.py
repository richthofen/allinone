#!/usr/bin/python 
# coding=UTF-8
import datetime

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.finance import quotes_historical_yahoo_ochl
from matplotlib.collections import LineCollection

from sklearn import cluster, covariance, manifold

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

###############################################################################
# Retrieve the data from Internet

# Choose a time period reasonably calm (not too long ago so that we get
# high-tech firms, and before the 2008 crash)
d1 = datetime.datetime(2015, 1, 1)
d2 = datetime.datetime(2016, 1, 1)

# 上证50成分股
symbol_dict = {
    "600000": "浦发银行",
    "600010": "包钢股份",
    "600015": "华夏银行",
    "600016": "民生银行",
    "600018": "上港集团",
    "600028": "中国石化",
    "600030": "中信证券",
    "600036": "招商银行",
    "600048": "保利地产",
    "600050": "中国联通",
    "600089": "特变电工",
    "600104": "上汽集团",
    "600109": "国金证券",
    "600111": "北方稀土",
    "600150": "中国船舶",
    "600256": "广汇能源",
    "600406": "国电南瑞",
    "600518": "康美药业",
    "600519": "贵州茅台",
    "600583": "海油工程",
    "600585": "海螺水泥",
    "600637": "东方明珠",
    "600690": "青岛海尔",
    "600837": "海通证券",
    "600887": "伊利股份",
    "600893": "中航动力",
    "600958": "东方证券",
    "600999": "招商证券",
    "601006": "大秦铁路",
    "601088": "中国神华",
    "601166": "兴业银行",
    "601169": "北京银行",
    "601186": "中国铁建",
    "601288": "农业银行",
    "601318": "中国平安",
    "601328": "交通银行",
    "601390": "中国中铁",
    "601398": "工商银行",
    "601601": "中国太保",
    "601628": "中国人寿",
    "601668": "中国建筑",
    "601688": "华泰证券",
    "601766": "中国中车",
    "601800": "中国交建",
    "601818": "光大银行",
    "601857": "中国石油",
    "601901": "方正证券",
    "601988": "中国银行",
    "601989": "中国重工",
    "601998": "中信银行"}

symbols, names = np.array(list(symbol_dict.items())).T

quotes = [quotes_historical_yahoo_ochl(symbol+".ss", d1, d2, asobject=True)
          for symbol in symbols]

open = np.array([q.open for q in quotes])
print open
# try:  
    # open = np.array([q.open for q in quotes]).astype(np.float)
# except Exception,e:  
    # print Exception,":",e
close = np.array([q.close for q in quotes])

# 每日价格浮动包含了重要信息！
variation = close - open

###############################################################################
# Learn a graphical structure from the correlations
edge_model = covariance.GraphLassoCV()

# standardize the time series: using correlations rather than covariance
# is more efficient for structure recovery
X = variation.copy().T
X /= X.std(axis=0)
edge_model.fit(X)

###############################################################################
# Cluster using affinity propagation

_, labels = cluster.affinity_propagation(edge_model.covariance_)
n_labels = labels.max()

for i in range(n_labels + 1):
    print('Cluster %i: %s' % ((i + 1), ', '.join(names[labels == i])))

###############################################################################
# Find a low-dimension embedding for visualization: find the best position of
# the nodes (the stocks) on a 2D plane

# We use a dense eigen_solver to achieve reproducibility (arpack is
# initiated with random vectors that we don't control). In addition, we
# use a large number of neighbors to capture the large-scale structure.
node_position_model = manifold.LocallyLinearEmbedding(
    n_components=2, eigen_solver='dense', n_neighbors=6)

embedding = node_position_model.fit_transform(X.T).T

###############################################################################
# Visualization
plt.figure(1, facecolor='w', figsize=(10, 8))
plt.clf()
ax = plt.axes([0., 0., 1., 1.])
plt.axis('off')

# Display a graph of the partial correlations
partial_correlations = edge_model.precision_.copy()
d = 1 / np.sqrt(np.diag(partial_correlations))
partial_correlations *= d
partial_correlations *= d[:, np.newaxis]
non_zero = (np.abs(np.triu(partial_correlations, k=1)) > 0.02)

# Plot the nodes using the coordinates of our embedding
plt.scatter(embedding[0], embedding[1], s=100 * d ** 2, c=labels,
            cmap=plt.cm.spectral)

# Plot the edges
start_idx, end_idx = np.where(non_zero)
#a sequence of (*line0*, *line1*, *line2*), where::
#            linen = (x0, y0), (x1, y1), ... (xm, ym)
segments = [[embedding[:, start], embedding[:, stop]]
            for start, stop in zip(start_idx, end_idx)]
values = np.abs(partial_correlations[non_zero])
lc = LineCollection(segments,
                    zorder=0, cmap=plt.cm.hot_r,
                    norm=plt.Normalize(0, .7 * values.max()))
lc.set_array(values)
lc.set_linewidths(15 * values)
ax.add_collection(lc)

# Add a label to each node. The challenge here is that we want to
# position the labels to avoid overlap with other labels
for index, (name, label, (x, y)) in enumerate(
        zip(names, labels, embedding.T)):

    dx = x - embedding[0]
    dx[index] = 1
    dy = y - embedding[1]
    dy[index] = 1
    this_dx = dx[np.argmin(np.abs(dy))]
    this_dy = dy[np.argmin(np.abs(dx))]
    if this_dx > 0:
        horizontalalignment = 'left'
        x = x + .002
    else:
        horizontalalignment = 'right'
        x = x - .002
    if this_dy > 0:
        verticalalignment = 'bottom'
        y = y + .002
    else:
        verticalalignment = 'top'
        y = y - .002
    plt.text(x, y, name, size=10,
             horizontalalignment=horizontalalignment,
             verticalalignment=verticalalignment,
             bbox=dict(facecolor='w',
                       edgecolor=plt.cm.spectral(label / float(n_labels)),
                       alpha=.6))

plt.xlim(embedding[0].min() - .15 * embedding[0].ptp(),
         embedding[0].max() + .10 * embedding[0].ptp(),)
plt.ylim(embedding[1].min() - .03 * embedding[1].ptp(),
         embedding[1].max() + .03 * embedding[1].ptp())

plt.title('上证50成分股')
plt.show()