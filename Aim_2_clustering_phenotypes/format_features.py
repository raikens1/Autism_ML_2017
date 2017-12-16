import pandas as pd

def format_features(raw_data, feature_names, individuals):
    """
    Gets only the features that were used to generate the 
        cluster from the raw data file.
    Args:
        raw_data (DataFrame): the original raw data for all 
            samples
        feature_names (list): the features names from the 
            centroid files used to generate the clusters
        individuals (list): the indices from the cluster 
            files, contains only the individuals that were 
            clustered
    Returns:
        all_data (DataFrame): Features that were used for
            generating the clusters with names formatted to 
            be compatible with feature names of other data 
            structures in analysis. Only includes
            individuals that were used in clustering.
    """
    
    raw_data = raw_data.loc[individuals, :]
    raw_data = raw_data.loc[:, get_compatible(feature_names)]

    return raw_data

def get_compatible(feature_names):
    """
    Changes column names from all_data.csv file to equivalent
        form in clustering files.
    Args:
        feature_names (list): column names from centroid file
    Returns:
        (list): column names that are compatible with those in 
            the all_data.csv file
    """
    return [convert(name) for name in feature_names]

def convert(name):
    """
    Converts the name from all_data.csv file version to 
        clustering file version.
    Args:
        name (str): name from all_data.csv
    Returns:
        new_name (str): name from clustering files
    """
    
    # For ADIR questions
    if name[0:4] == 'ADIR':
        if len(name) == 8:
            new_name = name[0:4] + ':' + name[5:]
        if len(name) == 10:
             new_name = name[0:4] + ':' + name[5:8] + '.' + name[9]
    
    # For ADOS questions
    if name[0:4] == 'ADOS':
        new_name = name[0:4] + ':' + name[5:]
        
    return new_name