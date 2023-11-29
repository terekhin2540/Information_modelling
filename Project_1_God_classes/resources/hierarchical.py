from sklearn.cluster import AgglomerativeClustering
from k_means import prepare_csv_to_array, paths_csv
import pandas as pd

n_clusters = 5


for i in range(len(paths_csv)):
    name_of_file = paths_csv[i].split('.')[1].split('/')[1]
    result, methods = prepare_csv_to_array(paths_csv[i])
    cluster = AgglomerativeClustering(n_clusters=n_clusters)
    cluster.fit_predict(result)
    df_final = pd.DataFrame()
    df_final['cluster_id'] = cluster.labels_
    df_final['method_name'] = methods
    # df_final = df_final.drop_duplicates(subset=['cluster_id', 'method_name'])
    # df_final = df_final.sort_values(by=['method_name'])
    df_final.to_csv(name_of_file + '_Agglo' + '.csv')
    print(f'{name_of_file} AgglomerativeClustering created')
