import numpy as np
import keras.backend as K

def pos_embedding(x):
    ''' Embedding layer as in Attention Is All You Need.
        Can be used this way :
        pos_embedding_layer = Lambda(lambda x: pos_embedding(x))(embedding_layer) '''
    
    dmodel = K.cast(K.shape(x)[-1], K.dtype(x))
    pos = K.cumsum(x * 0 + 1, axis=-2)
    i = K.cumsum(x * 0 + 1, axis=-1)
    even_mask = i % 2
    
    pos_emb = pos / K.pow(K.cast(10000, K.dtype(x)), 2 * i / dmodel)
    pos_emb = even_mask * K.sin(pos_emb) + (1 - even_mask) * K.cos(pos_emb)
    
    return pos_emb

tensor = np.random.uniform(size=40).reshape((10, 4))
tensor = K.variable(value=tensor, dtype='float32', name='example')
embeddings = pos_embedding(tensor)
print(K.eval(embeddings))
