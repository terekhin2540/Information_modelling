from sklearn.metrics import silhouette_score
import pandas as pd
from k_means import paths_csv
import os
from sklearn.cluster import KMeans, AgglomerativeClustering


print(paths_csv)
def find_cluster_files(path, Kmeans=False):
    paths_csv = []
    if Kmeans:
        substring = "_KMeans"
    else:
        substring = "_Agglo"
    for file_name in os.listdir(path):
        f = os.path.join(path, file_name)
        # checking if it is a file
        if os.path.isfile(f) and f.endswith('.csv'):
            if substring in f:
                paths_csv.append(f)
    return paths_csv


def prepare_csv_to_labels(path):
    df = pd.read_csv(path)
    labels = df['cluster_id'].values
    return labels


def prepare_csv_to_array(path):
    df = pd.read_csv(path)
    df.columns = df.iloc[0]
    df = df.iloc[1:]
    df.columns = df.columns.fillna('method_name')
    # df = df.drop_duplicates(subset=['method_name'])
    df_methods = df['method_name'].values # (148)
    df.drop(df.columns[[0]], axis=1, inplace=True)
    df = df.astype('int')
    # print(f'df_methods - {len(df_methods)}')
    df_array = df.values #(124, 148)
    # print(f'df_array - {df_array.shape}')
    return df_array, df_methods





directory = '.'


def silhuette_score(directory, cluster_file=True):
    if cluster_file:
        agglo_files = find_cluster_files(directory)
        kmeans_files = find_cluster_files(directory, Kmeans=True)

        paths_csv.sort()
        agglo_files.sort()
        kmeans_files.sort()

        Kmean_scores = []
        Agglo_scores = []
        name_of_files = []
        final_df = pd.DataFrame()

        for i in range(len(paths_csv)):
            name_of_file = paths_csv[i].split('.')[1].split('/')[1]
            result, methods = prepare_csv_to_array(paths_csv[i])
            Kmeans_labels = prepare_csv_to_labels(kmeans_files[i])
            Agglo_labels = prepare_csv_to_labels(agglo_files[i])

            Kmean_score = silhouette_score(result, Kmeans_labels, metric='euclidean')
            Agglo_score = silhouette_score(result, Agglo_labels, metric='euclidean')

            Kmean_scores.append(Kmean_score)
            Agglo_scores.append(Agglo_score)
            name_of_files.append(name_of_file)

        final_df['Kmeans_score'] = Kmean_scores
        final_df['Agglo_score'] = Agglo_scores
        final_df['File_name'] = name_of_files
        print(final_df)

    else:

        paths_csv.sort()
        # print(paths_csv)
        for i in range(len(paths_csv)):
            name_of_file = paths_csv[i].split('.')[1].split('/')[1]
            result, methods = prepare_csv_to_array(paths_csv[i])
            k_list = []
            kmean_scores = []
            agglo_scores = []

            for j in range(2, 60):
                n_clusters = j
                k_list.append(j)

            # implementation for Kmeans
                kmeans = KMeans(n_clusters=n_clusters, random_state=0)
                kmeans.fit_predict(result)
                df_final_kmeans = pd.DataFrame()
                df_final_kmeans['cluster_id'] = kmeans.labels_
                df_final_kmeans['method_name'] = methods
                labels_kmeans = df_final_kmeans['cluster_id'].values
                kmean_scores.append(silhouette_score(result, labels_kmeans, metric='euclidean'))


                # Implementation for Agglo
                cluster = AgglomerativeClustering(n_clusters=n_clusters)
                cluster.fit_predict(result)
                df_final_agglo = pd.DataFrame()
                df_final_agglo['cluster_id'] = cluster.labels_
                df_final_agglo['method_name'] = methods
                labels_agglo = df_final_agglo['cluster_id'].values
                agglo_scores.append(silhouette_score(result, labels_agglo, metric='euclidean'))

            final_df = pd.DataFrame()
            final_df['K'] = k_list
            final_df['K_mean'] = kmean_scores
            final_df['Agglo'] = agglo_scores
            print(f'Relusts of Kmeans and Agglo for {name_of_file} file : ')
            print(final_df)
            print()


silhuette_score(directory)
silhuette_score(directory, cluster_file=False)
print(paths_csv)










