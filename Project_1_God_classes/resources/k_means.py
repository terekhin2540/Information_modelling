from sklearn.cluster import KMeans
import pandas as pd
import os

directory = '.'
paths_csv = []
substring_KMeans = "_KMeans"
substring_Agglo = "_Agglo"
substring_keyword = '_keywords'
substring_test = '_test_test_test'
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f) and f.endswith('.csv'):
        if substring_KMeans in f or substring_Agglo in f or substring_keyword in f or substring_test in f:
            pass
        else:
            paths_csv.append(f)


# print(paths_csv)
def prepare_csv_to_array(path):
    df = pd.read_csv(path)
    df.columns = df.iloc[0]
    df = df.iloc[1:]
    df.columns = df.columns.fillna('method_name')
    df_methods = df['method_name'].values # (148)
    df.drop(df.columns[[0]], axis=1, inplace=True)
    df = df.astype('int')
    # print(f'df_methods - {len(df_methods)}')
    df_array = df.values #(124, 148)
    # print(f'df_array - {df_array.shape}')
    return df_array, df_methods



n_clusters = 5

for i in range(len(paths_csv)):
    name_of_file = paths_csv[i].split('.')[1].split('/')[1]
    result, methods = prepare_csv_to_array(paths_csv[i])
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit_predict(result)
    df_final = pd.DataFrame()
    df_final['cluster_id'] = kmeans.labels_
    df_final['method_name'] = methods
    # df_final = df_final.drop_duplicates(subset=['cluster_id', 'method_name'])
    # df_final = df_final.sort_values(by=['method_name'])
    df_final.to_csv(name_of_file +'_KMeans' +'.csv')
    print(f'{name_of_file} KMeans created')

