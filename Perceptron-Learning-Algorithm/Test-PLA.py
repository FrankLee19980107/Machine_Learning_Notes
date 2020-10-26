from PLA import BinaryClassifier

import numpy as np

# Load Datasets
Dat = open('./train.dat')
Data = Dat.read()
Dat.close()

# Turn into matrix
train_x = []
train_y = []

for data in Data.split('\n'):
    d = np.array(data.split('\t')).astype(float)
    train_x.append(d[:-1])
    train_y.append(d[-1])
    
PLA = BinaryClassifier()
PLA.LoadData(train_x, train_y)
PLA.PocketPLA()
