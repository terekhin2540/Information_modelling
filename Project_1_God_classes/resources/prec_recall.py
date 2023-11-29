import pandas as pd
import os
from itertools import combinations


directory = '.'
paths_csv_keywords = []
paths_csv_Kmeans = []
paths_csv_Agglo = []
substring_KMeans = "_KMeans"
substring_Agglo = "_Agglo"
substring_keyword = '_keywords'
substring_test = '_test_test_test'
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f) and f.endswith('.csv'):
        if substring_KMeans in f:
            paths_csv_Kmeans.append(f)

        elif substring_Agglo in f:
            paths_csv_Agglo.append(f)

        elif substring_keyword in f:
            paths_csv_keywords.append(f)

        else:
            pass


paths_csv_Kmeans.sort()
paths_csv_Agglo.sort()
paths_csv_keywords.sort()



# Load the clustering and ground truth data from CSV files
for i in range(len(paths_csv_Kmeans)):
    name_of_file = paths_csv_Kmeans[i].split('.')[1].split('/')[1].split('_')[0]

    Kmeans_clustering_data = pd.read_csv(paths_csv_Kmeans[i])
    Kmeans_clustering_data = Kmeans_clustering_data.drop_duplicates(subset=['cluster_id', 'method_name'])
    Kmeans_clustering_data = Kmeans_clustering_data.sort_values(by=['method_name'])
    Kmeans_clustering_data = Kmeans_clustering_data[['cluster_id', 'method_name']]
    # print(f'Kmeans_clustering_data.shape: {Kmeans_clustering_data.shape}')

    Agglo_clustering_data = pd.read_csv(paths_csv_Agglo[i])
    Agglo_clustering_data = Agglo_clustering_data.drop_duplicates(subset=['cluster_id', 'method_name'])
    Agglo_clustering_data = Agglo_clustering_data.sort_values(by=['method_name'])
    Agglo_clustering_data = Agglo_clustering_data[['cluster_id', 'method_name']]
    # print(f'Agglo_clustering_data.shape: {Agglo_clustering_data.shape}')

    ground_truth_data = pd.read_csv(paths_csv_keywords[i])
    ground_truth_data = ground_truth_data.drop_duplicates(subset=['cluster_id', 'method_name'])
    ground_truth_data = ground_truth_data.sort_values(by=['method_name'])
    ground_truth_data = ground_truth_data[['cluster_id', 'method_name']]
    # print(f'ground_truth_data.shape: {ground_truth_data.shape}')

    # Group the clustering data by cluster_id
    Kmeans_clustering_groups = Kmeans_clustering_data.groupby('cluster_id')['method_name'].apply(list)

    print(f'Kmeans_clustering_groups: \n{Kmeans_clustering_groups}')

    Agglo_clustering_groups = Agglo_clustering_data.groupby('cluster_id')['method_name'].apply(list)
    # Group the ground truth data by cluster_id
    ground_truth_groups = ground_truth_data.groupby('cluster_id')['method_name'].apply(list)

    k_means_list_intra_pairs = []
    for i in Kmeans_clustering_groups:
        # print(f'Cluster: \n{i}')
        combinations_set = combinations(i, 2)
        for j in combinations_set:
            k_means_list_intra_pairs.append(j)

    k_means_set_intra_pairs = set(k_means_list_intra_pairs)



    agglo_list_intra_pairs = []
    for i in Agglo_clustering_groups:

        combinations_set = combinations(i, 2)
        for j in combinations_set:
            agglo_list_intra_pairs.append(j)

    agglo_set_intra_pairs = set(agglo_list_intra_pairs)



    ground_list_intra_pairs = []
    for i in ground_truth_groups:

        combinations_set = combinations(i, 2)
        for j in combinations_set:
            ground_list_intra_pairs.append(j)

    ground_set_intra_pairs = set(ground_list_intra_pairs)

    precision_kmeans = len((k_means_set_intra_pairs.intersection(ground_set_intra_pairs))) / len(k_means_set_intra_pairs)
    precision_agglo = len((agglo_set_intra_pairs.intersection(ground_set_intra_pairs))) / len(agglo_set_intra_pairs)

    recall_kmeans = len((k_means_set_intra_pairs.intersection(ground_set_intra_pairs))) / len(ground_set_intra_pairs)
    recall_agglo = len((agglo_set_intra_pairs.intersection(ground_set_intra_pairs))) / len(ground_set_intra_pairs)

    f_score_kmeans = (2 * precision_kmeans * recall_kmeans) / (precision_kmeans + recall_kmeans)

    f_score_agglo = (2 * precision_agglo * recall_agglo) / (precision_agglo + recall_agglo)

    print(f'File {name_of_file}')
    print(f'Precision Kmeans:  {round(precision_kmeans, 4)},   Precision Agglo : {round(precision_agglo, 4)}')
    print(f'Recall Kmeans: {round(recall_kmeans,4)},   Recall Agglo : {round(recall_agglo, 4)}')
    print(f'F-score Kmeans: {round(f_score_kmeans, 4)},   F-score Agglo: {round(f_score_agglo, 4)}')
    print()


