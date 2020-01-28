from dataset_utils import label_encode_column, one_hot_encoder, cast_column_to_int

def preprocess(X_train, X_test, y_train, y_test):
    label_encode_column(X_train, 'age')
    label_encode_column(X_test, 'age')

    label_encode_column(X_train, 'tumor-size')
    label_encode_column(X_test, 'tumor-size')

    label_encode_column(X_train, 'inv-nodes')
    label_encode_column(X_test, 'inv-nodes')

    X_train = one_hot_encoder(X_train, 'menopause')
    X_test = one_hot_encoder(X_test, 'menopause')

    X_train = one_hot_encoder(X_train, 'node-caps')
    X_test = one_hot_encoder(X_test, 'node-caps')

    X_train = one_hot_encoder(X_train, 'irradiant')
    X_test = one_hot_encoder(X_test, 'irradiant')

    X_train = one_hot_encoder(X_train, 'breast')
    X_test = one_hot_encoder(X_test, 'breast')

    X_train = one_hot_encoder(X_train, 'breast-quad')
    X_test = one_hot_encoder(X_test, 'breast-quad')

    y_train = one_hot_encoder(y_train, 'class')
    y_test = one_hot_encoder(y_test, 'class')

    cast_column_to_int(X_train, 'deg-malig')
    cast_column_to_int(X_test, 'deg-malig')

    for key in X_train:
        size = len(X_test[list(X_test.keys())[0]])
        if key not in X_test:
            X_test[key] = [0 for i in range(size)]


    return (X_train, y_train), (X_test, y_test)