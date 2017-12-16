import pandas as pd

def format_labels(raw_data, individuals):
    """
    Returns the desired labels for clustered individuals with 
        missing values filled with 'missing'.
    Args:
        raw_data (DataFrame): the original raw data for all 
            samples
        individuals (list): the indices from the cluster files, 
            contains only the individuals that were clustered
    Returns:
        labels (DataFrame): Contains clustered individuals on 
            rows and desired labels on columns with missing 
            values assigned the value 'missing'
    """

    raw_data = raw_data.fillna('missing')
    raw_data = raw_data.loc[individuals, :]
    
    # Identify labels of interest
    dataset = raw_data[ \
                             'dataset'\
                            ]
    diagnostic = raw_data[[ \
                                  'cpea_adjusted_diagnosis', \
                                  'cpea_diagnosis', \
                                  'diagnosis' \
                                 ]]
    demographic = raw_data[[ \
                                   'ethnicity', \
                                   'gender', \
                                   'race' \
                                  ]]
    adir = raw_data[[ \
                     'ADIR:diagnosis', \
                     'ADIR:diagnosis_num_nulls', \
                     'ADIR:communication', \
                     'ADIR:restricted_repetitive_behavior', \
                     'ADIR:social_interaction', \
                     'ADIR:abnormality_evident_before_3_years'\
                    ]]
    ados = raw_data[[ \
                     'ADOS:diagnosis', \
                     'ADOS:diagnosis_num_nulls', \
                     'ADOS:communication', \
                     'ADOS:restricted_repetitive_behavior', \
                     'ADOS:social_interaction', \
                     'ADOS:module' \
                    ]]
    medical = raw_data[['Medical History:Anxiety Disorder', \
                        'Medical History:Autoimmune/Allergic', \
                        'Medical History:Behavior', \
                        'Medical History:ID', \
                        'Medical History:Mood Disorder', \
                        'Medical History:Psychotic Disorder', \
                        'Medical History:Seizures', \
                        'Medical History:Tourette or Tic Disorder' \
                       ]]

    # Subset labels from raw data
    labels = pd.concat([dataset, diagnostic, demographic, \
                        adir, ados, medical], axis=1)
    
    return labels