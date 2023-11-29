import os
import pandas as pd
from k_means import paths_csv

# direction = '.'
with open("./keyword_list.txt", "r") as f:
    keywords = f.readlines()

keywords = [x.replace('\n', '') for x in keywords]
print(keywords)


def compute_ground_truth(path):
    df = pd.read_csv(path)
    df.columns = df.iloc[0]
    df = df.iloc[1:]
    df.columns = df.columns.fillna('method_name')
    df_methods = df['method_name'].values  # (148)
    # if keyword == 'create'
    return df_methods



def create_cluster_id(temp_index, keyword):
    if temp_index == 100:
        return 15
    if keyword == 'create':
        return 0
    elif keyword == 'object':
        return 1
    elif keyword == 'cache':
        return 3
    elif keyword == 'uri':
        return 4
    elif keyword == 'standalone':
        return 5
    elif keyword == 'encoding':
        return 6
    elif keyword == 'identifier':
        return 7
    elif keyword == 'user':
        return 8
    elif keyword == 'error':
        return 9
    elif keyword == 'content':
        return 10
    elif keyword == 'parameter':
        return 11
    elif keyword == 'subset':
        return 12
    elif keyword == 'global':
        return 13
    elif keyword == 'component':
        return 14


for i in range(len(paths_csv)):
    methods = compute_ground_truth(paths_csv[i])
    name_of_file = paths_csv[i].split('.')[1].split('/')[1]
    # print(methods)
    our_class_df = pd.DataFrame()
    method_list = []
    index_list = []
    keywords_list = []
    for method in methods:
        origin_method = method
        for keyword in keywords:
            method_list.append(origin_method)
            method = method.lower()
            index = method.find(keyword)
            if index == -1:
                index = 100
            index_list.append(index)
            keywords_list.append(keyword)

    our_class_df['cluster_id_temp'] = index_list
    our_class_df['method_name'] = method_list
    our_class_df['keyword_name'] = keywords_list

    our_class_df = our_class_df.drop_duplicates(subset=['cluster_id_temp', 'method_name'])
    idx = our_class_df.groupby('method_name')['cluster_id_temp'].idxmin()
    result = our_class_df.loc[idx]

    cluster_id_list = []
    result['cluster_id'] = result.apply(lambda x: create_cluster_id(x.cluster_id_temp, x.keyword_name), axis=1)
    result['cluster_id'] = result['cluster_id'].rank(method='dense', ascending=True).astype(int) - 1
    result = result[['cluster_id', 'method_name']]
    result.to_csv(name_of_file + '_keywords' + '.csv')









