import os
import pandas as pd
import javalang
import re


def get_class_metrics(tree):
    metrics_list = []

    for _, node in tree:
        if isinstance(node, javalang.tree.ClassDeclaration):
            class_metrics = {'class': node.name, 'MTH': 0, 'FLD': 0, 'RFC': 0, 'INT': 0,
                              'SZ': 0, 'CPX': 0, 'EX': 0, 'RET': 0, 'BCM': 0, 'NML': 0,
                              'WRD': 0, 'DCM': 0}
            max_method_metrics = {'SZ': 0, 'CPX': 0, 'EX': 0, 'RET': 0}
            nlp_metrics = get_nlp_metrics(node)

            for member in node.body:
                if isinstance(member, javalang.tree.MethodDeclaration):
                    method_metrics = get_method_metrics(member)

                    class_metrics['MTH'] += 1
                    method_invocations = [n for n in member.filter(javalang.tree.MethodInvocation)]
                    class_metrics['RFC'] = len(method_invocations)

                    if 'public' in member.modifiers:
                        class_metrics['RFC'] += 1

                    max_method_metrics['SZ'] = max(max_method_metrics['SZ'], method_metrics['SZ'])
                    max_method_metrics['CPX'] = max(max_method_metrics['CPX'], method_metrics['CPX'])
                    max_method_metrics['EX'] = max(max_method_metrics['EX'], method_metrics['EX'])
                    max_method_metrics['RET'] = max(max_method_metrics['RET'], method_metrics['RET'])

                elif isinstance(member, javalang.tree.FieldDeclaration):
                    class_metrics['FLD'] += 1

            if node.implements:
                class_metrics['INT'] += 1

            class_metrics.update(max_method_metrics)
            class_metrics.update(nlp_metrics)
            metrics_list.append(class_metrics)

    return metrics_list


def get_method_metrics(node):
    metrics = {'SZ': 0, 'CPX': 0, 'EX': 0, 'RET': 0}
    # print(node.body)
    if node.body is not None:
        for statement in node.body:
            if isinstance(statement, javalang.tree.StatementExpression):
                metrics['SZ'] += 1

            elif isinstance(statement, (javalang.tree.IfStatement, javalang.tree.ForStatement,
                                        javalang.tree.WhileStatement, javalang.tree.DoStatement,
                                        javalang.tree.SwitchStatement)):
                metrics['CPX'] += 1

            elif isinstance(statement, javalang.tree.TryStatement):
                if statement.catches is not None:
                    metrics['EX'] += 1

            elif isinstance(statement, javalang.tree.MethodDeclaration):
                if statement.throws is not None:
                    metrics['EX'] += len(statement.throws)

            elif isinstance(statement, javalang.tree.ReturnStatement):
                metrics['RET'] += 1

    return metrics


def get_nlp_metrics(node):
    nlp_metrics = {'BCM': 0, 'NML': 0, 'WRD': 0, 'DCM': 0}

    if node.body is not None:
        if node.documentation:
            nlp_metrics['BCM'] = 1

            method_name_lengths = []
            for member in node.body:
                if isinstance(member, javalang.tree.MethodDeclaration):
                    method_name_lengths.append(len(member.name))

            if len(method_name_lengths) > 0:
                nlp_metrics['NML'] = sum(method_name_lengths) / len(method_name_lengths)

            comment_words = re.findall('\w+', node.documentation)
            if len(comment_words) > 0:
                nlp_metrics['WRD'] = max(len(word) for word in comment_words)
                num_statements = len(node.body)
                if num_statements > 0:
                    nlp_metrics['DCM'] = len(comment_words) / num_statements
                else:
                    nlp_metrics['DCM'] = 0

    return nlp_metrics


def devide_file(file_path):
    with open(file_path) as file:
        source_code = file.read()
    tree = javalang.parse.parse(source_code)
    class_met = get_class_metrics(tree)
    #print(class_metrics)
    return class_met


list_of_rows = []
for path, dirs, files in os.walk("."):
    for file in files:
        # we are only going to analyze the java files
        if file.endswith('.java'):
            java_file = os.path.join(path, file)
            #print(java_file)
            new_list = devide_file(java_file)
            for i in new_list:
                list_of_rows.append(i)


# print(len(list_of_rows))
# print(list_of_rows)
df = pd.DataFrame(list_of_rows)
df.to_csv('feature_vector_file.csv', index=False)
