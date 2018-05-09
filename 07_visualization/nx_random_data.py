import matplotlib.pyplot as plt
import networkx as nx
import random as rand
import seaborn as sns
import numpy as np
import pandas as pd

d = []
for i in range(100):
    n = rand.randint(5, 20)
    p = rand.uniform(0.01, 0.8)
    G = nx.gnp_random_graph(n, p, seed=None, directed=False)
    density = nx.density(G)
    clustering = nx.average_clustering(G)
    d.append([i, density, clustering])

arr = np.array(d)
df = pd.DataFrame(arr)
df.columns = ["Index", "Density", "Clustering"]
df['Category'] = pd.qcut(df["Density"], 5, labels=False) + 1
df.to_csv("network_metrics.csv", sep="\t")

sns.stripplot(x="Category", y="Clustering", data=df);
plt.show(block=True)

ax = sns.pointplot(x="Category", y="Clustering", hue="Category", data=df)
plt.show(block=True)

sns.pairplot(data=df, hue="Category")
plt.show(block=True)

sns.violinplot(x="Category", y="Clustering", hue="Category", data=df, inner="quart")
plt.show()

sns.lvplot(x="Category", y="Clustering", hue="Category", data=df, linewidth=2.5)
plt.show(block=True)

sns.swarmplot(x="Category", y="Clustering", hue="Category", data=df)
plt.show(block=True)

sns.regplot(x="Density", y="Clustering", data=df)
plt.show(block=True)

# sns.lmplot(x="Density", y="Clustering", hue="Category", truncate=True, size=5, data=df)
# plt.show(block=True)

# sns.lmplot(x="Density", y="Clustering", hue="Category", size=5, data=df)
# plt.show(block=True)

"""
g = sns.factorplot(x="sex", y="total_bill",
...                    hue="smoker", col="time",
...                    data=tips, kind="bar",
...                    size=4, aspect=.7);



sns.factorplot(x="Category", y="Clustering", hata=df, kind="box", size=4, aspect=.5)

sns.jointplot(x="total_bill", y="tip", data=tips, kind="reg")

def hexbin(x, y, color, **kwargs):
    cmap = sns.light_palette(color, as_cmap=True)
    plt.hexbin(x, y, gridsize=15, cmap=cmap, **kwargs)

with sns.axes_style("dark"):
    g = sns.FacetGrid(tips, hue="time", col="time", size=4)
g.map(hexbin, "total_bill", "tip", extent=[0, 50, 0, 10])

g = sns.jointplot(x1, x2, kind="kde", size=7, space=0)


with sns.axes_style('white'):
    sns.jointplot("x", "y", data, kind='hex')

with sns.axes_style(style='ticks'):
    g = sns.factorplot("day", "total_bill", "sex", data=tips, kind="box")
    g.set_axis_labels("Day", "Total Bill");

sns.jointplot("total_bill", "tip", data=tips, kind='reg');
"""
