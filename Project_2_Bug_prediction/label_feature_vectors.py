import os
import pandas as pd
import re

def find_class_names(directory):
    class_names = set()
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.src'):
                file_path = os.path.join('resources/modified_classes/', file)
                with open(file_path, 'r') as f:
                    content = f.read()
                    class_names |= {line.strip().split(".")[-1] for line in content.split("\n")}
    class_names.discard('')
    return class_names

directory = 'resources/modified_classes/'
class_name = find_class_names(directory)

# print(class_name)


class_names = find_class_names('resources/modified_classes/')
feature_vector_file = 'feature_vector_file.csv'
df = pd.read_csv(feature_vector_file)

feature_set = set(df.iloc[:, 0])

df['buggy'] = df.iloc[:, 0].isin(class_names).astype(int)

new_df = df.copy()

print(new_df[df['buggy'] == 1]['buggy'] .count())
print(new_df[df['buggy'] == 0]['buggy'] .count())
print(new_df.shape)
print(new_df.describe().transpose()[['min', 'max', 'mean']])

new_csv_file = 'feature_vector_file_target.csv'
new_df.to_csv(new_csv_file, index=False)
