import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
import numpy as np

# Завантаження даних з набору Iris
iris = datasets.load_iris()
X = iris.data[:, :2]

# Вибираємо перші дві ознаки для візуалізації
Y = iris.target

# Використання методу KMeans для кластеризації даних
kmeans = KMeans(n_clusters=Y.max() + 1, init='k-means++', n_init=10, max_iter=300,
                tol=0.0001, verbose=0, random_state=None, copy_x=True, algorithm='auto')

kmeans.fit(X)

y_pred = kmeans.predict(X)

# Візуалізація результатів кластеризації
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.show()


# Визначення функції для знаходження кластерів
def find_clusters(X, n_clusters, rseed=2): rng = np.random.RandomState(rseed)


i = rng.permutation(X.shape[0])[:n_clusters]
centers = X[i]
while True:
    labels = pairwise_distances_argmin(X, centers)
new_centers = np.array([X[labels == i].mean(0) for i in
                        range(n_clusters)])
if np.all(centers == new_centers): break
centers = new_centers
return centers, labels

# Використання власної функції для знаходження кластерів та візуалізація
centers, labels = find_clusters(X, 3)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.show()

# Використання власної функції з іншим seed та візуалізація
centers, labels = find_clusters(X, 3, rseed=0)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.show()

# Використання KMeans без власної функції та візуалізація
labels = KMeans(3, random_state=0).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.show()
