def dataset_split(dataset, ratio, label):
    size = len(dataset[list(dataset.keys())[0]])
    test_size = int(size * ratio)
    X_test = {k: [v for v in dataset[k][:test_size]] for k in dataset if k != label}
    X_train = {k: [v for v in dataset[k][test_size:size]] for k in dataset if k != label}
    y_test = {k: [v for v in dataset[k][:test_size]] for k in dataset if k == label}
    y_train = {k: [v for v in dataset[k][test_size:size]] for k in dataset if k == label}
    return (X_train, y_train), (X_test, y_test)
    
def label_encode_column(dataset, column):
    categories = sorted(list(set(dataset[column])))
    for data_index, data in enumerate(dataset[column]):
        for index, label in enumerate(categories):
            if data == label:
                dataset[column][data_index] = index

def convert_to_row(dataset):
    size = len(dataset[list(dataset.keys())[0]])
    for index in range(size):
        yield [dataset[k][index] for k in dataset]

def one_hot_encoder(dataset, column):
    categories = list(set(dataset[column]))
    new_dataset = {k: [] for k in dataset if k != column}
    for category in categories:
        new_column = f'{column}__{category}'
        new_dataset[new_column] = []
    for index, data in enumerate(dataset[column]):
        for category in categories:
            col_name = f'{column}__{category}'
            if data == category:
                for key in dataset:
                    if key == column:
                        continue
                    new_dataset[key].append(dataset[key][index])
                new_dataset[col_name].append(1)
            else:
                new_dataset[col_name].append(0)
    return new_dataset

def cast_column_to_int(dataset, column):
    dataset[column] = [int(i) for i in dataset[column]]