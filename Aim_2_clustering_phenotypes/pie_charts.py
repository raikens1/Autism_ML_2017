import pandas as pd
import matplotlib.pyplot as plt
from utilities import *

def pie_charts_pipeline(labels, clusters):
    """
    Displays a pie chart of the label value proportions in each
        cluster for all labels associated with the individuals
        in a cluster result.
    Args:
        labels (DataFrame): The desired labels on the columns 
            with individuals on the rows. Each entry is the label 
            value for a particular individual.
        clusters (DataFrame): Clustering result with individuals 
            on rows and cluster ID's on columns. Must be in the 
            form of crisp results, where each entry is a 0 or 1 
            indicating membership in the cluster. An individual 
            may belong to more than one cluster.
    Returns:
        None
    """
    
    label_counts = {label: counts_for(labels[label], clusters) \
                    for label in labels.columns}
    for label in label_counts:
        display_pie_chart(label, label_counts[label])

def display_pie_chart(label, label_counts):
    """
    Displays a pie chart of the label value proportions in each
        cluster for a single label associated with the individuals
        in a cluster result.
    Arguments:
        label (str): The name of the label being graphed.
        label_counts (DataFrame): A matrix of counts with cluster 
            IDs on the rows and unique label values on the columns.
    Returns:
        None
    """
    
    num_clusters = len(label_counts.index)
    
    # Initialize figure
    fig = plt.figure(figsize=(3*(num_clusters+1), 3))
    
    # Plot pie charts for each cluster
    for cluster in range(1, num_clusters+1):
        counts = label_counts.iloc[cluster-1, :].values
        plt.subplot(1, num_clusters+1, cluster)
        plt.pie(counts, labels=None)
        plt.title('Cluster %d (%d)' % (cluster, int(counts.sum())))
        plt.axis('equal')
    
    # Add legend as another subplot by adding a pie chart with a
    # legend and then hiding the plot
    plt.subplot(1, num_clusters+1, num_clusters+1)
    pie = plt.pie(label_counts.iloc[0, :].values, labels=None)
    plt.legend(labels=label_counts.columns, loc='center')
    for group in pie:
        for wedge in group:
            wedge.set_visible(False)
        
    # Add title, adjust spacing, and display figure
    plt.suptitle('k = %d; label = %s' % (num_clusters, label))
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()