import os
import javalang
import pandas as pd

def extract_data(input_path):

    class_names = []
    method_nums = []
    file_path = []
    for path, dirs, files in os.walk(input_path):
        for file in files:
            if file.endswith('.java'):  # change this to the appropriate file extension
                with open(os.path.join(path, file), 'r') as f:
                    content = f.read()
                    tree = javalang.parse.parse(content)
                    for _, node in tree:
                        if isinstance(node, javalang.tree.ClassDeclaration):
                            class_names.append(node.name)
                            method_nums.append(len(node.methods))
                            file_path.append(os.path.join(path, file))

    df = pd.DataFrame({'class_name': class_names, 'method_num': method_nums, 'file_path': file_path})
    mean_value = df['method_num'].mean()
    std = df['method_num'].std()
    df = df[df['method_num'] > mean_value + 6 * std]
    return df

# path = '.'
# print(extract_data(path))
# df = extract_data(path)
# print(df['file_path'].iloc[0])
# print(df['file_path'].iloc[1])
# print(df['file_path'].iloc[2])
# print(df['file_path'].iloc[3])

