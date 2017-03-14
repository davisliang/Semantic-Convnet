test
rt numpy as np
from sklearn import decomposition as dec
import matplotlib.pyplot as plt


num_samples = 10
sample_vectors = np.random.randn(num_samples,300)

pca_w2v = dec.PCA(n_components = 2)

pca_w2v.fit(sample_vectors)

pca_vectors = pca_w2v.transform(sample_vectors)

U, V = zip(*pca_vectors)
X = np.zeros(10)
Y = np.zeros(10)

plt.figure()
ax = plt.gca()
ax.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1,width = 0.005)
ax.set_xlim([-15, 15])
ax.set_ylim([-15, 15])
ax.annotate('dog', xy=(U[0], V[0]))
plt.draw()
plt.show()
