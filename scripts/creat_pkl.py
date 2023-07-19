import pickle
import numpy as np

def create_dct(n=10):
    tmp = {}
    for i in range(1,n+1):
        key = 'patient0000'+str(i)+'/study'+str(1)+'/view1'+'_frontal'
        tmp[key] = {
            'pdesc': np.ones((1, 40, 768)), # of shape: size(equal to 1 for a single vector), sequence length, embedding dimension(match num of hidden layers)
            'bics': np.array([1, 2]),
            'bts': np.array([1, 2, 3, 4]),
            'label': np.array([1, 0, 1, 0, 1, 0, 1, 0])
        }
    return tmp

def create_pckl(dct, filename='./data/predev/dct.pkl'):
    with open(filename, 'wb') as handle:
        pickle.dump(dct, handle, protocol=pickle.HIGHEST_PROTOCOL)

new = create_dct(3)
create_pckl(new)

