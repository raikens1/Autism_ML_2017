import pandas as pd

def soft_to_crisp_clusters(soft_clusters, rule='single'):
    """
    Converts soft clustering results to crisp clustering 
        results based on rule for cluster assignment from 
        partial membership values.
    Args:
        soft_clusters (DataFrame): Clustering result with 
            individuals on rows and cluster ID's on columns.
            For "soft" results, each entry is in [0,1] 
            indicating partial membership to cluster and
            rows sum to 1.
        rule (string): 'single' specifies that individuals 
            should be assigned to the cluster they have the
            most partial membership in. 'multiple' 
            specifies that the individuals should be 
            assigned to all of the clusters they have 
            partial membership in.
    Returns:
        crisp_clusters (DataFrame): Clustering result with 
            individuals on rows and cluster ID's on columns, 
            where each entry is a 0 or 1 indicating membership 
            to cluster. If type is 'single', then each 
            individual is only assigned to a single cluster. 
            If there are ties, the individual is assigned to 
            the cluster that comes first in the DataFrame 
            columns. If type is 'multiple', then each 
            individual is assigned to all clusters that they 
            have partial membership in.
    """
    
    if rule not in ['single', 'multiple']:
        raise ValueError('Rule for cluster assignment must be \
        "single" or "multiple"')
    
    if rule == 'single':
        assignment = soft_clusters.idxmax(axis=1)
        crisp_clusters = pd.get_dummies(assignment)
        return crisp_clusters
    
    elif rule == 'multiple':
        crisp_clusters = soft_clusters.astype(bool).astype(int)
        return crisp_clusters