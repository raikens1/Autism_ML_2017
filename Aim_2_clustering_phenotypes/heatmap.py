import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utilities import *

def heatmap_pipeline(data, sort_by=[], clusters=None, size=(15, 9)):
    """
    Displays the heat map of features with values scaled to be 
        in [0,1] for each column.
    Args:
        data (DataFrame): Unprocessed data. Contains examples on 
            rows and features of interest on columns, where each 
            entry in the DataFrame is the feature value for a 
            particular example. Only works on numerical feature 
            values.
        sort_by (list): Contains the feature names that the 
            individuals should be sorted by prior to generating
            the heat map. Sorting will occur in the order that 
            these values are listed.
        clusters (DataFrame): Clustering result with individuals 
            on rows and cluster ID's on columns. Must be in the 
            form of "crisp" results, where each entry is a 0 or 1 
            indicating membership in the cluster. An individual 
            may only belong to a single cluster. If provided, 
            will sort individuals in the feature set by cluster 
            membership before features in the list sort_by.
        size (tuple): The figure size for the heatmap.
    Returns:
        None
    """
    
    data = numeric_data(data)
    data = sort_data(data, sort_by, clusters)
    data = scale_data(data)
    display_heatmap(data, size)
    
def numeric_data(data):
    """
    Checks to see that all data are numeric. If not, notifies the 
        user that non-numeric columns were removed.
    Args:
        data (DataFrame): Unprocessed data. Contains examples on 
            rows and features of interest on columns, where each 
            entry in the DataFrame is the feature value for a 
            particular example.
    Returns:
        numeric_data (DataFrame): Contains only the columns with 
            numeric values from the inputted data.
    Note: If all features only contain numeric values, then 
        this function just returns the inputted data.
    """
    
    # Get the features that only contain numeric values
    numeric_data = data.select_dtypes(include=[np.number])
    
    # Notify the user if non-numeric columns were removed
    if len(numeric_data.columns) < len(data.columns):
        print "Non-numeric columns were removed from the data set"
        
    return numeric_data
        
def sort_data(data, sort_by=[], clusters=None):
    """
    Sorts the rows of a DataFrame by the values of listed columns.
    Args:
        data (DataFrame): Unprocessed data. Contains examples on 
            rows and features of interest on columns, where each 
            entry in the DataFrame is the feature value for a 
            particular example. Only works on numerical feature 
            values.
        sort_by (list): Contains the feature names that the 
            individuals should be sorted by prior to generating
            the heat map. Sorting will occur in the order that 
            these values are listed.
        clusters (DataFrame): Clustering result with individuals 
            on rows and cluster ID's on columns. Must be in the 
            form of "crisp" results, where each entry is a 0 or 1 
            indicating membership in the cluster. An individual 
            may only belong to a single cluster. If provided, 
            will sort individuals in the feature set by cluster 
            membership before features in the list sort_by.
    Returns:
        sorted_data (DataFrame): The sorted input data. If sorting 
            by cluster, then adds column of cluster membership as 
            first column in data.
    Note: If sort_by is empty and no clusters are provided, then 
        this function just returns the inputted data.
    """
    
    # Catches input errors
    for col_name in sort_by:
        if col_name not in data.columns:
            raise ValueError(col_name + ' is not in the data set.')
    if isinstance(clusters, pd.DataFrame) \
    and not single_cluster_membership(clusters):
        raise ValueError('Individuals must belong to one cluster.')
    
    # Add the cluster_id as a column to the dataset
    if isinstance(clusters, pd.DataFrame):
        clusters.columns = range(len(clusters.columns))
        cluster_id = clusters.idxmax(axis=1).to_frame(name='cluster_id')
        data = pd.concat([cluster_id, data], axis=1)
        sort_by.insert(0, 'cluster_id')
    
    # If there is anything to sort by, then sort the data
    if sort_by:
        data = data.sort_values(by=sort_by)
        
    return data

def scale_data(data):
    """
    Scales the values in each column of data set to be in [0,1].
    Args:
        data (DataFrame): Unprocessed data. Contains examples on 
            rows and features of interest on columns, where each 
            entry in the DataFrame is the feature value for a 
            particular example. Only works on numerical feature 
            values.
    Returns:
        scaled_data (DataFrame): The scaled data with non-numeric 
            columns removed. 
    """
    
    # Catch invalid input
    data = pd.DataFrame(numeric_data(data), dtype=np.float64)
    
    for column in data:
        
        # If all values in the column are the same, then prevent 
        # column from being turned into NaN's. Assign an integer 
        # value of 0 because we care about visual differences 
        # between values and a lighter color will draw less 
        # attention to a value that is the same for all individuals
        if data[column].max() == data[column].min():
            data[column] = 0
        
        # Otherwise scale the data to be in [0,1]
        else:
            data[column] = ((data[column] - data[column].min()) / \
                            (data[column].max() - data[column].min()))
    
    return data

def display_heatmap(data, size=(15,9)):
    """
    Displays heat map for a data set.
    Args:
        data (DataFrame): Unprocessed data. Contains examples on 
            rows and features of interest on columns, where each 
            entry in the DataFrame is the feature value for a 
            particular example. Only works on numerical feature 
            values.
        size (tuple): The figure size for the heat map.
    Returns:
        None
    """
    
    # Catch invalid input
    data = pd.DataFrame(numeric_data(data), dtype=np.float64)
    
    # Plot the heat map
    sns.set()
    plt.figure(figsize=size)
    sns.heatmap(data, xticklabels=False, yticklabels=False)
    plt.show()