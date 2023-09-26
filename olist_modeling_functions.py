#! /usr/bin/env python3
# coding: utf-8

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join
import datetime

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_samples
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
import matplotlib.cm as cm

def plot_silhouette_analysis(X, k_min, k_max):
    for n_clusters in range(k_min,k_max):
        # Create a subplot with 1 row and 2 columns
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(8, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax.set_title("The silhouette plot for the various clusters.")
        ax.set_xlabel("The silhouette coefficient values")
        ax.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax.set_yticks([])  # Clear the yaxis labels / ticks
        ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = %d" % n_clusters),
                    fontsize=14, fontweight='bold')
                    
def k_means(df, clusters_number):
    
    """Implement k-means clustering on dataset
    
    INPUT:
        dataset : dataframe. Dataset for k-means to fit.
        clusters_number : int. Number of clusters to form.
        end : int. Ending range of kmeans to test.
    OUTPUT:
        Cluster results and t-SNE visualisation of clusters."""
    
    
    kmean = KMeans(n_clusters = clusters_number, random_state = 1)
    kmean.fit(df)

    # Extract cluster labels
    cluster_labels = kmean.labels_
        
    # Create a cluster label column in original dataset
    df_new = df.assign(Cluster = cluster_labels)
    
    # Initialise TSNE
    model = TSNE(random_state=1)
    transformed = model.fit_transform(df)
    
    # Plot t-SNE
    plt.title('Flattened Graph of {} Clusters'.format(clusters_number))
    sns.scatterplot(x=transformed[:,0], y=transformed[:,1], hue=cluster_labels, style=cluster_labels, palette="Set1")
    
    return df_new, cluster_labels

def snake_plot(normalised_df_rfm, df_rfm_kmeans, df_rfm_original):
    """Transform dataframe and plot snakeplot"""
    
    # Transform df_normal as df and add cluster column
    normalised_df_rfm = pd.DataFrame(normalised_df_rfm, 
                                       index=df_rfm_original.index, 
                                       columns=df_rfm_original.columns)
    normalised_df_rfm['Cluster'] = df_rfm_kmeans['Cluster']

    # Melt data into long format
    df_melt = pd.melt(normalised_df_rfm.reset_index(), 
                        id_vars=['customer_unique_id', 'Cluster'],
                        value_vars=['Recency', 'Frequency', 'Monetary'], 
                        var_name='Metric', 
                        value_name='Value')

    plt.xlabel('Metric')
    plt.ylabel('Value')
    
    return sns.pointplot(data=df_melt, x='Metric', y='Value', hue='Cluster')

def rfm_values(df):
    '''
    Calcualte average RFM values and size for each cluster

    '''
    df_new = df.groupby(['Cluster']).agg({
        'Recency': ['mean', 'min','max'],
        'Frequency': ['mean', 'min','max'],
        'Monetary': ['mean', 'min','max', 'count']
    }).round(0)
    
    return df_new

def df_1_year_n_months(df, n_months):
    
    date_1_year = df['order_purchase_timestamp'].min()+ pd.DateOffset(years=1)
    date_1_year_n_months = date_1_year + pd.DateOffset(weeks=n_months)
    # define the dataframe after 1 year and n months.
    df_1_year_n_months = df[df['order_purchase_timestamp']<=date_1_year_n_months]
    Now = df_1_year_n_months.order_purchase_timestamp.max()
    df_1_year_n_months = df_1_year_n_months.groupby('customer_unique_id').agg({'order_purchase_timestamp': ['max','count'],
                                                                'payment_value': 'sum'})                                                              
    # Rename of columns.
    df_1_year_n_months.columns = ['last_purchase_order','purchase_order_count', 'sum_payment_value']
    # Recency
    
    df_1_year_n_months['Recency'] = df_1_year_n_months.last_purchase_order.apply(lambda x:(Now-x).days)
    # We rename purchase_order_count by Frequency and sum_payment_value by Monetary.
    df_1_year_n_months.rename(columns = {'purchase_order_count':'Frequency', 'sum_payment_value':'Monetary'}, inplace = True)
    # We select only RFM features.
    df_1_year_n_months = df_1_year_n_months[['Recency', 'Frequency', 'Monetary']]
    # log transformation
    df_1_year_n_months = np.log(df_1_year_n_months +1)
    return df_1_year_n_months


      
        
        