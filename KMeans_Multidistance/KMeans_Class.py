import numpy as np
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt

class KMeans: 
    '''
        This class allows the calculation of the K-Means clustering algorithm using multiple distances. Standard libraries focus on the 
        euclidian distance, but a set of different distances are allowed in this situation to tackle specific problems. 
    '''
    
    def __init__(self, k, maxiter, distance='euclidean', seed=None, record_heterogeneity=None, verbose=False):
        '''
            The initializer function allows to define the parameters that are going to  be used to fit the model to the data that will be 
            passed. 
            Input parameters: 
                -k = number of clusters to find.
                -maxiter = maximum number of iterations to run.
                -distance = type of distance to apply during the fit of the model. Options allowed for this parameter are ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'] if the input is a sparse matrix. If the input is not a sparse matrix the options for the distance metric include all previous plus ['braycurtis', 'canberra', 'chebyshev', 'correlation','mahalanobis', 'minkowski', 'seuclidean', 'sqeuclidean'].
                -seed = seed for the random generation of the intital centroids of the clusters.
                -record_heterogeneity = (optional) a list to store the history of heterogeneity as a function of the number of iterations,
                                        if None, the heterogeneity history is not stored.
                -verbose = (optional) if set to True, print how many data points changed their cluster labels in each iteration. 
        '''
        self.k = k
        self.maxiter = maxiter
        self.record_heterogeneity = record_heterogeneity
        self.verbose = verbose
        self.distance = distance
        self.seed = seed
        
    
   
    def compute_heterogeneity(self, data, k, centroids, cluster_assignment, distance):
        '''
            Function to compute the heterogeneity of the iteration of the KNN fit process. 
            Inputs: 
                -data = all the points of the feature space to calculate the heterogeneity
                -k = number of clusters to compute
                -centroids = position of the baricenter of each cluster
                -cluster assignment = list of data points with their assignments within the clusters calculated for the iteration in place.
        '''
        heterogeneity = 0.0
        for i in range(k):

            # Select all data points that belong to cluster i. Fill in the blank
            member_data_points = data[cluster_assignment==i, :]

            if member_data_points.shape[0] > 0: # check if i-th cluster is non-empty

                # Compute distances from centroid to data points, based on the type of distance that is passed to the function
                distances = pairwise_distances(member_data_points, [centroids[i]], metric=distance)
                squared_distances = distances**2
                heterogeneity += np.sum(squared_distances)

        return heterogeneity

    
    def revise_centroids(self,data, k, cluster_assignment):
        '''
            After all points are assigned to a cluster, the centroids have to be revised to ensure that the distance between the points and the centroid is minimized. 
            Inputs:
                - data = all the points of the feature space
                - k = number of clusters to calculate
                - cluster_assignment = list of cluster ids with the current assignment of the data points to the clusters
        '''
        new_centroids = []
        for i in range(k):
            # Select all data points that belong to cluster i. Fill in the blank
            member_data_points = data[cluster_assignment == i]
            # Compute the mean of the data points. Fill in the blank
            centroid = data[cluster_assignment ==i].mean(axis=0)

            # Convert numpy.matrix type to numpy.ndarray type
            centroid = np.ravel(centroid)
            new_centroids.append(centroid)
        new_centroids = np.array(new_centroids)

        return new_centroids

    
    def assign_clusters(self,data, centroids, distance):
        '''
            Calculate the distance between each point to the centroids and decide to which cluster each point is assigned
            Inputs:
                - data = all the points of the feature space
                - centroids = baricenter points of the different clusters 
                - distance = type of distance selected to be used to calculate the distance between the points and the centroids
        '''
    
        # Compute distances between each data point and the set of centroids, based on the distance selected:
        distances_from_centroids = pairwise_distances(data, centroids, metric=distance)
        # Compute cluster assignments for each data point:
        cluster_assignment = np.argmin(distances_from_centroids, axis=1)

        return cluster_assignment
    
    
    def get_initial_centroids(self, data, k, seed=None):
        '''
            Randomly choose k data points as initial centroids
            Inputs: 
                - data = all the points of the feature space
                - k = number of clusters to calculate
                - seed = initial seed for the random number calculator
                
        '''
        if seed is not None: # useful for obtaining consistent results
            np.random.seed(seed)
        n = data.shape[0] # number of data points

        # Pick K indices from range [0, N).
        rand_indices = np.random.randint(0, n, k)

        # Keep centroids as dense format, as many entries will be nonzero due to averaging.
        centroids = data[rand_indices,:]

        return centroids
    
   
    def plot_heterogeneity(self):
        '''
            Function that allows to plot the evolution of the heterogeneity for each cluster with regards to the number of iterations
            Inputs:
                - heterogeneity = List of heterogeneity values calculated during the fit of the model
                - k = number of clusters that have been calculated
        '''
        plt.figure(figsize=(7,4))
        plt.plot(self.record_heterogeneity, linewidth=2)
        plt.xlabel('# Iterations')
        plt.ylabel('Heterogeneity')
        plt.title('Heterogeneity of clustering over time, K={0:d}'.format(self.k))
        plt.rcParams.update({'font.size': 12})
        plt.tight_layout()
    
    
    def fit(self,data):

        '''
            This function runs k-means on given data using a model of the class KNN.
            Output: 
                - centroids = Cluster centroids that define the clusters that have been generated by the algorithm
                - cluster assignments = List of points with their cluster id, defining which is the cluster they belong to.
        '''
        
        centroids = self.get_initial_centroids(data=data, k=self.k, seed=self.seed)
        prev_cluster_assignment = None

        for itr in range(self.maxiter):        
            if self.verbose:
                print("Iteration " + repr(itr) +". Calculating the cluster assignments for all data points.")

            # 1. Make cluster assignments using nearest centroids
            cluster_assignment = self.assign_clusters(data=data, centroids=centroids, distance=self.distance)

            # 2. Compute a new centroid for each of the k clusters, averaging all data points assigned to that cluster.
            centroids = self.revise_centroids(data=data, k=self.k, cluster_assignment=cluster_assignment)

            # Check for convergence: if none of the assignments changed, stop
            if prev_cluster_assignment is not None and (prev_cluster_assignment==cluster_assignment).all():
                break

            # Print number of new assignments 
            if prev_cluster_assignment is not None:
                num_changed = np.sum(prev_cluster_assignment!=cluster_assignment)
                if self.verbose:
                    print('    {0:5d} elements changed their cluster assignment during this assignment.'.format(num_changed))   

            # Record heterogeneity convergence metric
            if self.record_heterogeneity is not None:
                score = self.compute_heterogeneity(data=data, k=self.k, centroids=centroids, cluster_assignment=cluster_assignment, distance=self.distance)
                self.record_heterogeneity.append(score)

            prev_cluster_assignment = cluster_assignment[:]

        return centroids, cluster_assignment