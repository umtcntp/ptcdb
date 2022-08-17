import pandas as pd
import numpy as np
import pickle

mp = {}
ifl = np.genfromtxt("cora.content", dtype=np.dtype(str))
a = len(ifl[1,1:-1])
y = ifl[:,-1]
for i in y:
    if i in mp:
        mp[i] += 1
    else:
        mp[i] = 1

print(a)
print(len(mp.keys()))
