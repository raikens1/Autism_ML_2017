import pandas as pd

def single_cluster_membership(clusters):
    """
    Checks to see if each individual belongs to only a single cluster.
    Args: 
        clusters (DataFrame): Clustering result with examples on rows 
            and cluster ID's on columns. Must be in the form of crisp 
            results, where each entry is a 0 or 1 indicating membership 
            to cluster.
    Returns:
        (boolean): True if the each example belongs to only a single 
            cluster and False otherwise.
    """
    row_sums = clusters.sum(axis=1)
    return row_sums.min() == 1 and row_sums.max() == 1

def counts_for(label_data, clusters):
    """
    Counts the unique values for a label by cluster membership.
    Args:
        label_data (Series): A single label of interest  with individuals 
            on the rows. Each entry is the label value for a particular 
            individual.
        clusters (DataFrame): Clustering result with individuals on rows 
            and cluster ID's on columns. Must be in the form of crisp 
            results, where each entry is a 0 or 1 indicating membership 
            in a cluster. An individual may belong to more than one cluster.
    Returns:
        label_counts (DataFrame): A matrix of counts with cluster IDs on 
            the rows and unique label values on the columns.
    """
    
    # Catch invalid input
    if not isinstance(label_data, pd.Series):
        raise ValueError('label_data must be a Series.')
        
    # Change cluster indices to integers if they aren't
    clusters.columns = map(str, range(len(clusters.columns)))
    
    # Fill n/a values with 'missing' so they can be counted
    label_data = label_data.fillna('missing')
    
    # One hot encode the label and add cluster_id as a column
    label_data = pd.get_dummies(label_data).sort_index(axis=1)
    
    # Join label and cluster data into one DataFrame
    label_data = pd.concat([label_data, clusters], axis=1)
    
    # Subset label data by cluster membership
    labels_by_cluster = []
    for cluster in clusters.columns:
        cluster_labels = label_data.loc[label_data[cluster] == 1]
        cluster_labels = cluster_labels.drop(labels=clusters.columns, axis=1)
        labels_by_cluster.append(cluster_labels)
    
    # Sum columns for each cluster membership and append to DataFrame
    label_counts = pd.DataFrame()
    for cluster in range(len(clusters.columns)):
        cluster_count = pd.Series(\
                                  labels_by_cluster[cluster].sum(), \
                                  name=cluster\
                                 )
        label_counts = label_counts.append(cluster_count)
        
    return label_counts