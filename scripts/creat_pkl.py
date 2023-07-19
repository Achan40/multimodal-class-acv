import pickle

def create_dct(n=10):
    tmp = {}
    for i in range(n):
        key = 'a'+str(i)
        tmp[key] = {
            'pdesc': ['one two three'],
            'bics': [1, 2],
            'bts': [1, 2, 3, 4],
            'label': [1, 0, 1, 0, 1, 0, 1, 0]
        }
    
    return tmp

def create_pckl(dct, filename='./data/predev/dct.pickle'):
    with open(filename, 'wb') as handle:
        pickle.dump(dct, handle, protocol=pickle.HIGHEST_PROTOCOL)

new = create_dct(10)
create_pckl(new)

