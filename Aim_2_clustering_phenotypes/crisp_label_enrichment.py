import pandas as pd
import numpy as np
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests
from utilities import *

def crisp_label_enrichment(labels, clusters, alpha=0.05):
    """
    Performs multiple hypergeometric tests with Benjamini-Hochberg 
        FDR correction to test for significant enrichment of each
        unique value for each label between clusters in a single 
        clustering result.
    Args:
        labels (DataFrame): The diagnostic, demographic, ADOS 
            phenotype, and ADIR phenotype labels on the columns 
            with individuals on the rows. Each entry is the label
            value for a particular individual.
        clusters (DataFrame): Clustering result with individuals on 
            rows and cluster ID's on columns. Must be "crisp" results,
            where each entry is a 0 or 1 indicating membership in the
            cluster. Can be used with multiple cluster membership.
        alpha (float): The significance level that should be used to 
            evaluate the labels
    Returns:
        tests (DataFrame): Each row represents a label value for a 
            particular cluster. The columns contain the adjusted 
            p_value and an indicator for whether the label was 
            significantly enriched for the cluster.
    Note: Calculations are based on the formulation provided in 
        scipy.stats.hypergeom, with variables:
            N = population size
            n = sample (cluster) size
            K = # in population with label value
            k = # in sample (cluster) with label value
        You might need to install statsmodels. To do this, you can use
            "pip install --user statsmodels"
    """
    
    # Calculate N, the total number of individuals being clustered
    N = len(labels.index)
    
    # Get a dictionary with key label and value a DataFrame with 
    # cluster ID on rows and unique label values on columns. Each
    # entry in the DataFrame is a count.
    label_counts = {label: counts_for(labels[label], clusters) \
                    for label in labels.columns}
    
    # Calculate K, the total number of individuals with the label value
    label_totals = {label: label_counts[label].sum() \
                    for label in label_counts}
    
    # Create structure for storing label, label_value, cluster, and
    # the values for calculating label enrichment: k, N, K, and n
    tests = []
    
    # Iterate over every unique label value for every cluster
    for label in label_counts:
        for label_value in label_counts[label].columns:
            for cluster in label_counts[label].index:
                
                # Calculate K, n, and k
                K = label_totals[label].loc[label_value]
                n = label_counts[label].loc[cluster, :].sum()
                k = label_counts[label].loc[cluster, label_value]
                
                # Add information to the data structure
                tests.append([label, label_value, cluster, k, N, K, n])

    # Convert data list into a DataFrame
    col_names = ['label', 'label_value', 'cluster', 'k', 'N', 'K', 'n']
    tests = pd.DataFrame(tests,  index=None, columns=col_names)

    # Calculate all of the p-values
    tests['p_value'] = hypergeom.sf(tests.k, tests.N, tests.K, tests.n)

    # Adjust the p-values using FDR Benjamini-Hochberg correction
    (x, adjusted_p_values, y, z) = multipletests( \
                                                  tests.p_value.tolist(), \
                                                  alpha=alpha, \
                                                  method='fdr_bh', \
                                                  is_sorted=False, \
                                                  returnsorted=False \
                                                 )
    
    # Modify DataFrame to only include desired information
    tests['adjusted_p_value'] = adjusted_p_values
    tests['significant'] = np.where(tests.adjusted_p_value<alpha, 1, 0)
    tests = tests.drop(['k', 'N', 'K', 'n', 'p_value'], axis=1)
    
    return tests