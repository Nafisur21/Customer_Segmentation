# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 18:05:09 2017

@author: Nafis
"""
#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

#loading dataset
dataset=pd.read_csv('Mall_Customers.csv')

#selecting features columns
df=dataset.iloc[:,3:5]

#Features Matrix
X=df.values

#Clustering Model
# First Step is to find the Number of Cluster
#1.Elbow Method
#2. Dendrograms

# Finding the optimal number of cluster using Elbow method
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,10):
    kmeans=KMeans(n_clusters=i,init='k-means++',)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,10),wcss)
plt.title('Elbow Method')
plt.xlabel('No. of cluster')
plt.ylabel('wcss: sum of dist. of sample to their closest cluster center' )


#Finding optimal number of cluster using dendrogram
import scipy
from scipy.cluster import hierarchy
dendro=hierarchy.dendrogram(hierarchy.linkage(X,method='ward'))

#Number of cluster 3 or 5

#1. KMeans Clustering Model with 3 cluster
kmeans_1=KMeans(n_clusters=3)
kmeans_1.fit(X)
cluster_pred=kmeans_1.predict(X)
cluster_pred_2=kmeans_1.labels_
cluster_center=kmeans_1.cluster_centers_

#Visualization using only 2 dimension (only y and z axis)
plt.scatter(X[cluster_pred==0,0],X[cluster_pred==0,1],s=100,c='r',label='cluster 1')
plt.scatter(X[cluster_pred==1,0],X[cluster_pred==1,1],s=100,c='g',label='cluster 2')
plt.scatter(X[cluster_pred==2,0],X[cluster_pred==2,1],s=100,c='b',label='cluster 3')
plt.scatter(cluster_center[:,0],cluster_center[:,1],s=300,c='y',label='centroid')


#2. KMeans Clustering Model with 5 cluster
kmeans_1=KMeans(n_clusters=5)
kmeans_1.fit(X)
cluster_pred=kmeans_1.predict(X)
cluster_pred_2=kmeans_1.labels_
cluster_center=kmeans_1.cluster_centers_

#Visualization using only 2 dimension (only y and z axis)
plt.scatter(X[cluster_pred==0,0],X[cluster_pred==0,1],s=100,c='r',label='cluster 1')
plt.scatter(X[cluster_pred==1,0],X[cluster_pred==1,1],s=100,c='g',label='cluster 2')
plt.scatter(X[cluster_pred==2,0],X[cluster_pred==2,1],s=100,c='b',label='cluster 3')
plt.scatter(X[cluster_pred==3,0],X[cluster_pred==3,1],s=100,c='c',label='cluster 4')
plt.scatter(X[cluster_pred==4,0],X[cluster_pred==4,1],s=100,c='m',label='cluster 5')
plt.scatter(cluster_center[:,0],cluster_center[:,1],s=300,c='y',label='centroid')


# Visualising the clusters
plt.scatter(X[cluster_pred==0,0],X[cluster_pred==0,1], s = 100, c = 'red', label = 'Standard')
plt.scatter(X[cluster_pred==1,0],X[cluster_pred==1,1], s = 100, c = 'blue', label ='Target' )
plt.scatter(X[cluster_pred==2,0],X[cluster_pred==2,1], s = 100, c = 'green', label = 'Careless')
plt.scatter(X[cluster_pred==3,0],X[cluster_pred==3,1], s = 100, c = 'cyan', label = 'Sensible')
plt.scatter(X[cluster_pred==4,0],X[cluster_pred==4,1], s = 100, c = 'magenta', label = 'Careful')
plt.scatter(cluster_center[:,0],cluster_center[:,1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Monthly Income ')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()






'''
#converting 3D ploblem into 2D using PCA, when data is linearly seperable
from sklearn.decomposition import PCA
#analysing no. of principal component
pca=PCA()
X_pca=pca.fit_transform(X)
exp_var=pca.explained_variance_ratio_
#we can see only two component can represent 85 percentage of the data
#choosing only 2 component because it explained 85% and also easy to visualization
pca=PCA(n_components=2)
X_pca=pca.fit_transform(X)
exp_var=pca.explained_variance_ratio_

#finding number of cluster
dendrogram=hierarchy.dendrogram(hierarchy.linkage(X_pca,method='ward'))
#no of cluster=5 by dendrogram
wcss=[]
for i in range(1,10):
    kmeans=KMeans(n_clusters=i,init='k-means++',)
    kmeans.fit(X_pca)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,10),wcss)
#no. of cluster=5 by elbow method

#1. KMeans Clustering Model with 5 cluster
kmeans_2=KMeans(n_clusters=5)
kmeans_2.fit(X_pca)
cluster_pred_2=kmeans_2.predict(X_pca)
cluster_center_2=kmeans_2.cluster_centers_
#Visualization using only 2 dimension (only y and z axis)
plt.scatter(X_pca[cluster_pred_2==0,0],X_pca[cluster_pred_2==0,1],s=100,c='r',label='cluster 1')
plt.scatter(X_pca[cluster_pred_2==1,0],X_pca[cluster_pred_2==1,1],s=100,c='b',label='cluster 2')
plt.scatter(X_pca[cluster_pred_2==2,0],X_pca[cluster_pred_2==2,1],s=100,c='g',label='cluster 3')
plt.scatter(X_pca[cluster_pred_2==3,0],X_pca[cluster_pred_2==3,1],s=100,c='c',label='cluster 4')
plt.scatter(X_pca[cluster_pred_2==4,0],X_pca[cluster_pred_2==4,1],s=100,c='m',label='cluster 5')
plt.scatter(cluster_center_2[:,0],cluster_center_2[:,1],s=300,c='y',label='centroid')
plt.legend()

'''


###################################################
#converting 3D into 2D using LDA linear discriminant analysis
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#lda=LinearDiscriminantAnalysis()
#X_lda=lda.fit_transform(X)
#lda.explained_variance_ratio_
#LDA is a Supervised model we need 'y' label also
#when data is linearly seperable
#########################################################

#Converting 3D into 2D when data is non-linear, when data is not linearly seperable Kernel PCA
from sklearn.decomposition import KernelPCA
kpca=KernelPCA(kernel='linear',n_components=2)
X_kpca=kpca.fit_transform(X)
dendrogram=hierarchy.dendrogram(hierarchy.linkage(X_kpca,method='ward'))
wcss=[]
for i in range(1,10):
    kmeans=KMeans(n_clusters=i,init='k-means++',)
    kmeans.fit(X_kpca)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,10),wcss)

kmeans_3=KMeans(n_clusters=5)
kmeans_3.fit(X_kpca)
cluster_pred_3=kmeans_3.predict(X_kpca)
cluster_center_3=kmeans_3.cluster_centers_
#Visualization using only 2 dimension (only y and z axis)
plt.scatter(X_kpca[cluster_pred_3==0,0],X_kpca[cluster_pred_3==0,1],s=100,c='r',label='cluster 1')
plt.scatter(X_kpca[cluster_pred_3==1,0],X_kpca[cluster_pred_3==1,1],s=100,c='b',label='cluster 2')
plt.scatter(X_kpca[cluster_pred_3==2,0],X_kpca[cluster_pred_3==2,1],s=100,c='g',label='cluster 3')
plt.scatter(X_kpca[cluster_pred_3==3,0],X_kpca[cluster_pred_3==3,1],s=100,c='c',label='cluster 4')
plt.scatter(X_kpca[cluster_pred_3==4,0],X_kpca[cluster_pred_3==4,1],s=100,c='m',label='cluster 4')

plt.legend()

###############################################################################

#Agglomerative H Clustering
from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=5)
hc.fit(X_pca)
cluster_pred_2=hc.labels_
hc.n_leaves_
hc.fit_predict(X_pca)

plt.scatter(X_pca[cluster_pred_2==0,0],X_pca[cluster_pred_2==0,1],s=100,c='r',label='cluster 1')
plt.scatter(X_pca[cluster_pred_2==1,0],X_pca[cluster_pred_2==1,1],s=100,c='b',label='cluster 2')
plt.scatter(X_pca[cluster_pred_2==2,0],X_pca[cluster_pred_2==2,1],s=100,c='g',label='cluster 3')
plt.scatter(X_pca[cluster_pred_2==3,0],X_pca[cluster_pred_2==3,1],s=100,c='c',label='cluster 4')
plt.scatter(X_pca[cluster_pred_2==4,0],X_pca[cluster_pred_2==4,1],s=100,c='m',label='cluster 5')


#visualization alternative one line code
plt.scatter(X_pca[:,0],X_pca[:,1],c=cluster_pred_2)