# Kaggle-Tabular-Playground-Series---Jul-2022
The Solution to Ensemble learning problem of the Tabular Playground Series - Jul 2022 Kaggle challenge
# 1. Read the csv file and drop irrelevant columns like 'id'
The shape of the dataset
The shape of the dataset after deleting the 1st row i.e., ‘id’ 

# 2. Feature Selection

From the heat map we can see that the relevant features are [f_7 to f_13] and [f_22 to f_28]. Variables f_07 - f_13 are integers (7 of
them), and f_22 - f_28 are floating point (also 7 of them). Floating point variables already appear to have a normal distribution, so
there is no need to transform them. But integer variables are not normally distributed. So, I power-transformed them into a more
'normal' shape and treated them as multivariate normal vars.
Removing the missing values:
this means they are no missing values in the dataset.

# 3. EDA: By plotting the histogram of 7 discrete and 22 continuous numerical features

# 4. Outlier Removing

Handling Rare values in discrete numerical features
I calculated the interquartile range (IQR) and outlier limits for each numerical feature in a list and replaced out-of-bounds values with
the bound. I then calculated the frequency of each discrete numerical feature in the same list, identifies values with a frequency below
0.0000001 (1%), and replaced them with the most frequent value.
Handling Outlier in the numerical features
Outlier detection and removal using the Z-score method where the data points with Z-scores greater than the threshold (=3) are
identified using numPy's "np.where()" function and removed.

# 5. Dimensionality reduction using PCA and SVD

• The heat map had shown that 14 features are important i.e., [f_7 to f_13] and [f_22 to f_28]. So I performed Principal
Component Analysis (PCA) on the clean train df data with 14 components using the PCA class from the
sklearn.decomposition module.
• Because the svd solver parameter is set to 'full,' the principal components will be computed using a full SVD (Singular Value
Decomposition). The random state parameter is set to 1001 to ensure reproducibility.
The explained variance ratio of 0.9094 indicates that the first 14 PCA principal components explain
90.94% of the total variance in the dataset. This implies that these 14 components can capture the majority of the relevant
information in the original dataset. To avoid overfitting and improve computational efficiency, it is common practise to choose the
number of components so that it explains a large percentage of the total variance while also keeping the number of components as low as possible.
The PCA-Transformed data

# 6. Forming the clusters

6.1 Finding the optimal number of k for kmeans clustering
6.1.1 The silhouette score method to obtain the optimum value of k for kmeans clustering
6.1.2 Elbow curve to obtain the optimum value of k for kmeans clustering
Thus, the highest silhouette score and the elbow point both are coming for k=2.
6.2. Forming the 2 clusters using kmeans for this pca-transformed dataset
To perform clustering on a dataset using K-means algorithm: first, I selected the relevant features from the dataset i.e., [f_7 to f_13]
and [f_22 to f_28] and applied a power transform to integer features. i.e., [f_22 to f_28]. Then, I applied Principal Component
Analysis (PCA) to extract two principal components from the data. Next, K-means algorithm with two clusters is applied to the PCA-
transformed data, and the resulting clusters are plotted on a scatter plot. The scatter plot shows the distribution of data points in the
two clusters.

#  7. Classification of all the rows into two clusters namely 0 and 1.

7.1. Using ICA to classify all the rows into these 2 clusters and returning the classification prediction query for Row no 42
I used the FastICA algorithm is to perform independent component analysis on the data after applying PCA. The transformed data is
then clustered using the KMeans algorithm with two clusters.
Finally, the cluster label of row 42 is obtained using fast ICA and printed i.e., 1.

7.2. Using LDA to classify all the rows into these 2 clusters and returning the classification prediction query for Row no 42
First, I applied the KMeans algorithm on the PCA transformed data to obtain two clusters. Then, LDA is used to classify the data
points into the two clusters based on their PCA coordinates. The number of rows in each cluster is counted and printed:
Finally, the cluster label of row 42 is obtained using LDA and printed i.e., 1.

7.3. KNN to classify all the rows into these 2 clusters and then returning the classification prediction query for Row no 42
First finding the optimum number of n_neighbors
I performed k-fold cross-validation to find the optimal number of neighbors for the K-Nearest Neighbors algorithm. The range of
neighbors considered goes from 1 to 10. For each number of neighbors, a KNeighborsClassifier is created and used to train and
evaluate the model using cross-validation with 10 folds. The accuracy score for each fold is computed and averaged across all folds.
The optimal number of neighbors is then selected as the one that resulted in the highest average accuracy score. Finally, the value of
the optimal number of neighbors is printed, i.e., 9.
Then, I applied the KMeans algorithm on the PCA transformed data to obtain two clusters. Then, is used to KNN classify the data
points into the two clusters based on their PCA coordinates. The number of rows in each cluster is counted and printed:
Finally, the cluster label of row 42 is obtained using LDA and printed i.e., 0.

# 8. Comparing the results of the above three models ICA, LDA and KNN.

The plots of the 5-fold CV Score for ICA, LDA and KNN as classifier
On comparing the three plots, KNN has the best accuracy of 99.90% followed by ICA at 99.88% and LDA at 96.67%.
Also, we can see that generally, on increasing the number of folds, the accuracy of the classifier first decreases then increases.
