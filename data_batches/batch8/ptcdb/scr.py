import pandas as pd
import pickle
import numpy as np

mp = {}
with open("./ptcdb.absid","rb") as f:
    df = pickle.load(f)
y = df["y"]

ifl = np.genfromtxt("ptcdb.content", dtype=np.dtype(str))
a = len(ifl[1,1:-1])


for i in y:
    if i in mp:
        mp[i] += 1
    else:
        mp[i] = 1

print("features:"+str(a))
print(mp)