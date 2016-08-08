# -*- coding: utf-8 -*-
"""
Created on Mon Jun 06 14:46:13 2016

@author: westerr

Implimentation of Gap Statistic using sklearn, pandas, and numpy with plotting 

Using: Initialize kmean_metrics class then call determine_best_k function

Sources: 
    https://datasciencelab.wordpress.com/2013/12/27/finding-the-k-in-k-means-clustering/
    http://web.stanford.edu/~hastie/Papers/gap.pdf
	
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
from math import sqrt

class kmeans_metrics():
    def __init__(self, B=10):
        self.B = B # number of monte carlo simulations to run
    
    def calc_log_Wk(self, kmeans, X):
        """
        Calculates the sum of squared euclidean distance for each cluster
        """        
        
        # First calculate squared distance between data and clusters
        X_dist_sqrd = kmeans.transform(X)**2
                  
        # Create dataframe with sqrd distance indexed by cluster labels
        df = pd.DataFrame(data=[i[kmeans.labels_[n]] for n, i in enumerate(X_dist_sqrd)],
                          index=kmeans.labels_)
        
        # Sum the squared distance by index (cluster number) to get inertia for each cluster
        cluster_inertia = df.groupby(level=0)[0].sum()
        
        # Return the log of the sum of the inertia divided by 2 times the cluster size
        return np.log(np.sum(cluster_inertia)) # Log Wk = log(sum(cluster_inertia))
        
    def __generate_reference_data(self, X):
        """
        Creates a reference dataset for Monte Carlo simulation
        
        Uses min/max for each feature and creates a resulting
        dataset of same size with random uniform variables
        """
        # Determine size of X
        rows, features = X.shape
        
        # Get min and max values for each feature
        feat_min = np.min(X, axis=0)
        feat_max = np.max(X, axis=0)
        
        # Initialized empty matrix of size feature, rows as ref_data
        ref_data = np.empty((features, rows))

        # Fill ref_data with random samples from uniform distribution
        for f in xrange(features):
            ref_data[f] = np.random.uniform(feat_min[f], feat_max[f], size=rows)
        
        # Transpose ref_data and return
        return ref_data.T
        
    def __reference_stats(self, X, kmeans_params):
        """
        Runs Monte Carlo simulation and calculates summary stats from simulation
        """
        # Create series indexed by iteration number
        ref_log_Wk = pd.Series(index=xrange(self.B))
        
        # Check if X is numpy array and if not convert to matrix
        if type(X).__module__.split('.')[0] != 'numpy':
            try:
                X = X.as_matrix()
            except AttributeError:
                print "X must be a numpy matrix or pandas dataframe"
            except:
                print "Unable to convert X to matrix"
                
        # Iterate for range of B 
        for b in xrange(self.B):
            # Random uniform sample 
            ref_data = self.__generate_reference_data(X)
            
            # Initialize a reference kmean class
            ref_kmeans = KMeans(**kmeans_params)

            # Fix clustering with reference data
            ref_kmeans.fit(ref_data)
            
            # Calculate log Wk 
            ref_log_Wk[b] = self.calc_log_Wk(ref_kmeans, ref_data)

        # Return the mean and std of the reference log(Wk)            
        return np.mean(ref_log_Wk), np.std(ref_log_Wk)
    
    def calc_gap_statistic(self, kmeans_actual, X, B=None):
        """
        Calculates gap statistics for an instance of kmeans evaluated 
        against the source data X
        """
        if B:
            self.B = B # if B, update self.B, else default is 10
        
        # Calculating log Wk for km actual
        log_Wk = self.calc_log_Wk(kmeans_actual, X)
        
        # Calculating reference stats
        ref_mean_log_Wk, ref_stdev_log_Wk = self.__reference_stats(X, kmeans_actual.get_params())
        
        # Calculate gap statistic
        gap_k = ref_mean_log_Wk - log_Wk
        
        # Calculate simulation error
        sk = sqrt(1+(1/float(self.B))) * ref_stdev_log_Wk
        
        # Return gap statistic and simulation error
        return log_Wk, ref_mean_log_Wk, gap_k, sk
        
    def determine_best_k(self, X, k_max=10, B=None, output_plot=False, test_silhouette=False, **kargs):
        """
        Function to determine the optimal number of clusters for kmean clustering as defined in the gap statistic 
        methodology.
        
        k_max is max number of clusters to consider

        B is the number of monte carlo simulations to run for each iteration of cluster numbers (1 to k_max), default is 10
        
        output_plots if True will create a plot where Gap(k) – Gap(k+1) + s(k+1) > 0 signifies good cluster separation

        test_silhouette if True includes silhouette score in k_summary_stats, else is populated with null values
        
        **kargs are args for sklearn.cluster.KMeans
        
        """        
        print "Determining best cluster size..."
        # Initialize k_summary_stats df 
        result_idx = np.arange(1, k_max+1)
        result_cols = ['gap_statistic', 'log_Wk', 'reg_mean_log_Wk', 'sim_error', 'inertia', 'silhouette_score']
        self.k_summary_stats = pd.DataFrame(columns=result_cols, index=result_idx)
        # Iterate for cluster numbers ranging from one to k_max
        for k in result_idx:
            # Initialize and fit clusters
            km_actual = KMeans(n_clusters=k, **kargs)
            km_actual.fit(X)
                
            # Determine gap statistics
            log_Wk, ref_mean_log_Wk, gap_k, sk = self.calc_gap_statistic(km_actual, X, B)

            # Determine silhouette score
            sil_score = metrics.silhouette_score(X, km_actual.labels_) if test_silhouette == True and k !=1 else np.nan # doesn't support k=1
            
            # Append results to results list
            self.k_summary_stats.loc[self.k_summary_stats.index==k, result_cols] = [gap_k, log_Wk, ref_mean_log_Wk, sk, km_actual.inertia_, sil_score]
        
        # Calculate gap_delta column 
        self.k_summary_stats['gap_delta'] = self.k_summary_stats['gap_statistic'] - \
                                            self.k_summary_stats['gap_statistic'].shift(-1) + \
                                            self.k_summary_stats['sim_error'].shift(-1)
         
        # Output chart
        if output_plot == True:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.bar(self.k_summary_stats.index-.25, self.k_summary_stats['gap_delta'], .5)
            plt.xticks(self.k_summary_stats.index)
            plt.xlabel('# of Clusters')
            plt.ylabel('Gap(k) - Gap(k+1) + s(k+1)')
            plt.title('Gap Statistic for k=1 to k=kmax-1') 
            
        try:
            # First try to get index (n_clusters) where gap_delta first becomes positive
            best_k = self.k_summary_stats[self.k_summary_stats['gap_delta'] >= 0].index.values[0]
        except IndexError:
            try:
                # if fails try to get the index (n_clusters) where gap_delta is the greatest
                best_k = self.k_summary_stats.sort_values('gap_delta', ascending=False).index.values[0]
            except IndexError:
                # else return 0
                print "Unable to determine best k"
                best_k = 0
        return best_k

# Example Use Case
if __name__ == '__main__':
    from sklearn.datasets.samples_generator import make_blobs
    from mpl_toolkits.mplot3d import Axes3D
    from sklearn.preprocessing import StandardScaler
    
    # Create sample dataset 
    X, y_true = make_blobs(n_samples=500, centers=3, n_features=3, cluster_std=1.1)
    
    # Scale using standard scaler
    X_scaled = StandardScaler().fit_transform(X)
    
    # Plotting clusters
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_scaled[:,0],
               X_scaled[:,1],
               X_scaled[:,2])
               
    # Initialize kmean_metrics
    km_metrics = kmeans_metrics()
    
    # Determine best k 
    best_k = km_metrics.determine_best_k(X_scaled, k_max=6, n_init=50, test_silhouette=True, output_plot=True)
    
    # Printing results
    print "Best Cluster Number: " + str(best_k)
    print km_metrics.k_summary_stats
    
