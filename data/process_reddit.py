import numpy as np
import scipy.sparse as sp

data = np.load("reddit.npz")
features = data['feats']

adj = sp.coo_matrix(sp.load_npz("reddit_adj.npz"))

print("Vertices:", adj.shape[0])
print("Edges:", len(adj.row))
assert len(adj.row) == len(adj.col)
assert features.shape[0] == adj.shape[0]

print("Features shape:", features.shape)
n = features.shape[0]

classes = [0] * n

y_train = data['y_train']
for i, ind in enumerate(data['train_index']):
    classes[ind] = int(float(y_train[i]))

y_val = data['y_val']
for i, ind in enumerate(data['val_index']):
    classes[ind] = int(float(y_val[i]))

y_test = data['y_test']
for i, ind in enumerate(data['test_index']):
    classes[ind] = int(float(y_test[i]))

assert all(x is not None for x in classes)

with open('reddit.content', 'w') as fout:
    for i in range(n):
        print(i, end=' ', file=fout)
        print(*list(features[i]), end=' ', file=fout)
        print(classes[i], file=fout)

with open('reddit.cites', 'w') as fout:
    for i, j in zip(adj.row, adj.col):
        print(i, j, file=fout)

