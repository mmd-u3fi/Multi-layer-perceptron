from dataset_parser import parse_dataset
from multi_layer_perceptron import MLP
from dataset_utils import dataset_split
from preprocessor import preprocess

dataset = parse_dataset()
(X_train, y_train), (X_test, y_test) = dataset_split(dataset, 0.2, 'class')
(X_train, y_train), (X_test, y_test) = preprocess(X_train, X_test, y_train, y_test)
mlp = MLP(input_dim=len(X_train), hidden_dim=20, output_dim=2)
mlp.train(X_train, y_train, 100)
print('Training complete\t\t\t\t')
accuracy = mlp.test(X_test, y_test)
print(accuracy)