import pandas as pd
import matplotlib.pyplot as plt

def visualize_movement_pipeline(all_clusters):
    """
    Tracks and displays the movement of individuals between clusters 
        as the number of clusters increases. 
    Args:
        all_clusters (list): Contains DataFrames for each value of k 
            for clustering results. Has individuals on rows and cluster 
            ID's on columns. Must be in form of crisp results, where 
            each entry is a 0 or 1 indicating membership in cluster. 
            Must belong to only a single cluster.
    Returns:
        None
    """
    
    k_values = [len(cluster_result.columns) \
                for cluster_result in all_clusters]
    memberships = get_memberships(all_clusters)
    display_movement(memberships, k_values)

def get_memberships(all_clusters):
    """
    Gets the cluster ID for each of the individuals over all clustering 
        results.
    Args:
        all_clusters (list): Contains DataFrames for each value of k for 
            clustering results. Has individuals on rows and cluster ID's 
            on columns. Must be in form of crisp results, where each entry 
            is a 0 or 1 indicating membership in cluster. Must belong to a 
            single cluster.
    Returns:
        memberships (DataFrame): Contains individuals on the rows and 
            clustering results on the columns, where each entry is the ID 
            of the cluster that a specific individual belonged to for a 
            specific clustering result. Columns are sorted by the k value 
            used to generate the clustering result. 
    """
    
    k_values = [len(cluster_result.columns) \
                for cluster_result in all_clusters]
    
    memberships = []
    for index, cluster_result in enumerate(all_clusters):
        k = k_values[index]
        cluster_result.columns = range(k)
        cluster_result = cluster_result.idxmax(axis=1).to_frame(name=k)
        memberships.append(cluster_result)
        
    memberships = pd.concat(memberships, axis=1)
    
    return memberships.sort_index(axis=1)

def display_movement(memberships, k_values):
    """
    Shows k pie charts separated by cluster where each chart displays the
        membership of individuals at k+1
    Args:
        memberships (DataFrame): Contains individuals on the rows and 
            clustering results on the columns, where each entry is the ID 
            of the cluster that a specific individual belonged to for a 
            specific clustering result. Columns are sorted by the k value 
            used to generate the clustering result. 
        k_values (list): Contains integer values of k that were used to 
            generate the clustering results. Must be consecutive integers.
    Returns:
        None
    """
    
    # Get counts for plots
    for k in k_values[:-1]:
        subset_memberships = memberships[[k, k+1]]
        subset_memberships = pd.get_dummies(subset_memberships, columns=[k+1])
        counts = subset_memberships.groupby([k]).sum()
        
        # Plot the movement of individuals
        fig = plt.figure(figsize=(3*k, 3))
        for index, cluster_count in counts.iterrows():
            plt.subplot(1, k+1, index+1)
            plt.pie(cluster_count.values, labels=None)
            plt.title('Cluster %d (%d)' % (index+1, int(cluster_count.sum())))
            plt.axis('equal')
            
        # Add legend as another subplot by adding a pie chart with a
        # legend and then hiding the plot
        plt.subplot(1, k+1, k+1)
        pie = plt.pie(counts.iloc[0, :].values, labels=None)
        plt.legend(labels=[i+1 for i in range(k+1)], loc='center')
        for group in pie:
            for wedge in group:
                wedge.set_visible(False)
                
        # Add title, adjust spacing, and display figure
        plt.suptitle('Movement from k = %d to %d' % (k, k+1))
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.show()
