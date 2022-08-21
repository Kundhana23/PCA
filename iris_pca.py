import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

def PCA(data,k):
    mean_x = data - np.mean(data , axis = 0)
    cov = np.cov(mean_x , rowvar = False)
    e_val , e_vector = np.linalg.eigh(cov)
    p = np.argsort(e_val)[::-1]
    s_eval = e_val[p]
    s_evector = e_vector[:,p]
    evector_subnet = s_evector[:,0:k]
    xr = np.dot(evector_subnet.transpose() , mean_x.transpose() ).transpose()
    return xr

dataset = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", names=['sepal length','sepal width','petal length','petal width','target'])
features = dataset.iloc[:,0:4]
target = dataset.iloc[:,4]
result = PCA(features , 2)
final = pd.DataFrame(result , columns = ['PC1','PC2'])
final = pd.concat([final , pd.DataFrame(target)] , axis = 1)

plt.figure(figsize = (10,10))
sb.scatterplot(data = final , x = 'PC1',y = 'PC2' ,hue = 'target' ,  s = 60 , palette= 'rocket')