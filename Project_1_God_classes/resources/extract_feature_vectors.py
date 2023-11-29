import pandas as pd

from find_god_classes import *
import os
import javalang



def get_fields(java_class):
    tree = javalang.parse.parse(java_class)
    class_names = []
    # Extract the fields
    # fields = []
    fields = set()
    for member in tree.types:
        class_names.append(member.name)
        for body in member.body:
            if isinstance(body, javalang.tree.FieldDeclaration):
                for declarator in body.declarators:
                    # fields.append(declarator)
                    fields.add(declarator)
    return fields, class_names



def get_methods(java_class):
    tree = javalang.parse.parse(java_class)
    # java_class = tree.types[0] #??????
    # Extract the fields
    # methods = []
    methods = set()
    for member in tree.types:
        for body in member.body:
            if isinstance(body, javalang.tree.MethodDeclaration):
                # methods.append(body)
                methods.add(body)
        return methods


def get_fields_accessed_by_method(method):
    # fields_accessed = []
    fields_accessed = set()

    for _, node in method.filter(javalang.tree.MemberReference):

        if node.qualifier == "":
            fields_accessed.add(node.member)
        else:
            fields_accessed.add(node.qualifier)

    return fields_accessed




def get_methods_accessed_by_method(method):
    # methods_accessed = []
    methods_accessed = set()
    if len(method.body) == 0:
        return methods_accessed

    for _, node in method.filter(javalang.tree.MethodInvocation):
        methods_accessed.add(node.member)

    return methods_accessed


file_names = []
number_of_methods = []
attributes = []
path = '.'
df = extract_data(path)
for i in df['file_path']:
    with open(f"{i}", 'r') as f:
        content = f.read()
        # print(content)
        fields_list = []
        methods_list = []
        results = []

        methods = get_methods(content)
        for i in methods:
            methods_list.append(i.name)

        fields, class_names = get_fields(content)
        for i in fields:
            fields_list.append(i.name)
        fields_accessed_by_method = {}
        methods_accessed_by_method = {}

        for method in methods:
            accessed_fields = get_fields_accessed_by_method(method)
            # print(accessed_fields)
            if accessed_fields:
                fields_accessed_by_method[method.name] = accessed_fields

            accessed_methods = get_methods_accessed_by_method(method)
            if accessed_methods:
                methods_accessed_by_method[method.name] = accessed_methods

        list_fields = []
        for i in range(len(methods_list)):
            list_fields.append([0] * len(fields_list))
            method = methods_list[i]
            field_new = fields_accessed_by_method.get(method, [])
            for j in range(len(fields_list)):
                field = fields_list[j]
                if field in field_new:
                    list_fields[i][j] = 1

        list_methods = []
        for i in range(len(methods_list)):
            list_methods.append([0] * len(methods_list))
            method = methods_list[i]
            method_new = methods_accessed_by_method.get(method, [])
            for j in range(len(methods_list)):
                if methods_list[j] in method_new:
                    list_methods[i][j] = 1

        fields_cols = pd.MultiIndex.from_product([['Fields'], fields_list])
        methods_cols = pd.MultiIndex.from_product([['Methods'], methods_list])
        columns = fields_cols.union(methods_cols)

        df = pd.DataFrame(columns=columns, index=methods_list)

        for i in range(len(methods_list)):
            fields_row = []
            methods_row = []
            for j in range(len(fields_list)):
                fields_row.append(list_fields[i][j])
            for j in range(len(methods_list)):
                methods_row.append(list_methods[i][j])
            df.loc[methods_list[i], ('Fields', slice(None))] = fields_row
            df.loc[methods_list[i], ('Methods', slice(None))] = methods_row

        df = df.fillna(0)
        df.to_csv(class_names[0] + '.csv')
        print(f'file {class_names[0]} created')
        num_cols_with_ones = sum(df.sum(axis=0) >= 1)
        attributes.append(num_cols_with_ones)
        file_names.append(class_names[0])
        number_of_methods.append(df.shape[1])


df_res = pd.DataFrame()
df_res['God_class'] = file_names
df_res['number_of_methods'] = number_of_methods
df_res['Attributes'] = attributes
print(df_res)
