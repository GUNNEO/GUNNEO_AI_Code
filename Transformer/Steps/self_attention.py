# This file provides a clear insight of self_attention
# in the form of attention matrix
import numpy as np
import math

# length: 4, dimension of each word vector: 8
L, d_k, d_v = 4, 8, 8
# randomly generate the matrix in the form of query, key and value
q = np.random.randn((L, d_k))
k = np.random.randn((L, d_k))
v = np.random.randn((L, d_v))
print("type:", type(q), "shape:", q.shape)

# obtain the attention score with query and key
attention_score = np.matmul(q, k.T)
print(attention_score.shape)

# scaled it to decrease the discreteness of the score matrix
scaled = attention_score / math.sqrt(d_k)
print(attention_score.var(), scaled.var())

# mask
mask = np.tril(np.ones((L, L)))
mask[mask == 0] = -np.infty
mask[mask == 1] = 0
scaled = scaled + mask


# softmax
def softmax(x):
    return (np.exp(x).T / np.sum(np.exp(x), axis=-1)).T


attention = softmax(scaled)
print("after softmax:\n", attention)

# generate output for feedforward NN
new_v = np.matmul(attention, v)
print("shape of new_v:", new_v.shape, "\n", new_v)
