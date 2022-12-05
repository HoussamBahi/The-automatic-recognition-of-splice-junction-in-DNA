# X: 0=A, T=1, G=2, C=3
# Y: EI=0, IE=1, N=2

## fixing random seed for comparaison and improvement
fix_seed = False
if fix_seed:
    from numpy.random import seed

    seed(1)
    from tensorflow import set_random_seed

    set_random_seed(2)

## importing libraries
import tensorflow as tf
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
import time

## loading dataset
file_path = "path\\splice-tr-woD.csv"

delimiter = ";"


def load_dataset(file_path, delimiter):
    # assumes last column contains labels and orther columns contain data
    dataset = numpy.loadtxt(file_path, delimiter=delimiter)
    N, labels_col = dataset.shape
    inputSize = labels_col - 1
    X = dataset[:, 0:inputSize]
    # converts int labels to neural network responses example: 1->[1,0,0] 2->[0,1,0] 3->[0,0,1]
    maxlabel = int(max(dataset[:, inputSize:labels_col]))
    Y = numpy.zeros((len(dataset), maxlabel + 1))
    for i, df in enumerate(dataset[:, inputSize:labels_col]):
        d = int(df)
        Y[i][d] = 1
    outputSize = Y.shape[1]
    return X, Y, N, inputSize, outputSize


X, Y, N, inputSize, outputSize = load_dataset(file_path, delimiter)

## Neural network setup
squeeze_func = 'relu'
if fix_seed:
    squeeze_func = 'linear'


def new_model():
    model = Sequential()
    model.add(Dense(10, input_dim=inputSize, activation=squeeze_func))
    model.add(Dense(10, activation=squeeze_func))
    model.add(Dense(outputSize, activation='sigmoid'))
    modified_sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.90, nesterov=True)  # optimizer customization
    model.compile(loss='binary_crossentropy', optimizer=modified_sgd, metrics=['accuracy'])
    return model


print("Using squeeze function = '" + squeeze_func + "' for internal layers")

## Holdout: Single Validation Fit
p = 0.25  # proportion of data saved for evaluation
epochs = 1500
n1 = int((1 - p) * N)
model1 = new_model()
batch_size = int(0.1 * n1)
verbose = 1


def SingleValidationFit(model0, ki, kf, epochs, batch_size):
    Xtrain = numpy.concatenate((X[0:ki, :], X[kf:N, :]))
    Ytrain = numpy.concatenate((Y[0:ki, :], Y[kf:N, :]))
    Xtest = X[ki:kf, :]
    Ytest = Y[ki:kf, :]
    model0.fit(Xtrain, Ytrain, epochs=epochs, batch_size=batch_size, verbose=verbose)
    scores = model0.evaluate(Xtest, Ytest)  # evaluate the model
    return (model0, scores)


do_single_validation = True

if do_single_validation:
    print("\n----------------------------")
    print("Computing holdout single validation fitting...")
    trainedModel, scores = SingleValidationFit(model1, n1, N, epochs, batch_size)
    print("\nSingle validation fit saving %.2f%% of the data for evaluation:" % (p * 100))
    print("%s: %.2f%% \n" % (trainedModel.metrics_names[1], scores[1] * 100))

    ## calculate predictions
    NN_predictions = trainedModel.predict(X[n1:N, :])
    label_predictions = []
    for p in NN_predictions:
        label_predictions.append(numpy.argmax(p))
    # print(label_predictions)

## cross-validation Fit
epochs = 1500
nfold = 10  # nfold-fold cross validation
batch_size = int(0.01 * N)


def CrossValidationFit(nfold, epochs, batch_size):
    dN = int(N / nfold)
    Scores = []
    Models = []
    for k in range(nfold):
        print("\n----------------------------")
        print("Computing %d/%d fitting for %d-fold-cross validation..." % (k, nfold, nfold))
        t1 = time.time()
        modelCV = new_model()
        ki, kf = k * dN, (k + 1) * dN
        model, score = SingleValidationFit(modelCV, ki, kf, epochs, batch_size)
        print("Accuracy of %d/%d fitting: %.2f%%" % (k, nfold, score[1] * 100))
        Scores.append(score)
        Models.append(model)
        t2 = time.time()
        print("Computation time of %d/%d fitting = %.2f s" % (k, nfold, t2 - t1))
    return numpy.array(Models), numpy.array(Scores)


do_cross_validation = True
if do_cross_validation:
    t1 = time.time()
    Models, Scores = CrossValidationFit(nfold, epochs, batch_size)
    cv_acc = numpy.average(Scores[:, 1]) * 100
    print("\n----------------------------")
    print("\n%d-fold cross validation fit:" % (nfold))
    print("avg acc: %.2f%% \n" % (cv_acc))
    t2 = time.time()
    print("Computation time = %.2f s" % (t2 - t1))

