import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv('Iris.csv')
data = data.drop(['Species'], axis=1)
data.shape


# # Preview of Data
# - There are 150 observations with 4 features each (sepal length, sepal width, petal length, petal width).
# - There are no null values, so we don't have to worry about that.

# In[4]:


data.head()


# In[94]:


data.info()


# In[95]:


data.describe()


# # Datapoints before clustering

# In[99]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2).fit(data)
data = pca.transform(data)
plt.figure(figsize=(8,8))
plt.scatter(data[:,0], data[:,1], c='red', s=25)


# # Creating clusters using k-means from scikit-learn

# In[98]:


from sklearn.cluster import KMeans
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)
    
#Plotting the results onto a line graph, allowing us to observe 'The elbow'
plt.plot(range(1, 11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') #within cluster sum of squares
plt.show()


# # Applying kmeans to the dataset / Creating the kmeans classifier

# In[78]:


kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(data)
print(y_kmeans)


# # Visualising the clusters
# 

# In[90]:


x = data
plt.figure(figsize=(8,8))
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 25, c = 'red', label = 'Cluster 1')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 25, c = 'blue', label = 'Cluster 2')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 25, c = 'green', label = 'Cluster 3')

#Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 25, c = 'black', label = 'Centroids')

plt.legend()
plt.show()

# In[ ]:




