import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # import can be necessary for older versions of matplotlib
from itertools import combinations
import inspect
from PostProcessing.Postprocessing import *
from PreProcessing.Preprocessing import *


# Define the functions for logistic regression

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def h(theta):
    def h_theta(x):
        return sigmoid(np.dot(x, theta))
    return h_theta


def cost_function(theta, X, y):
    return - (1 / (X.shape[0])) * np.sum(y * np.log(h(theta)(X)) + (1 - y) * np.log(1 - h(theta)(X)))


def gradient_descent(theta, x, y, learning_rate=1.):
    one_over_mean = (1 / x.shape[0])
    return theta - learning_rate * one_over_mean * np.dot(h(theta)(x) - y, x)


def train_model(theta, X, y, n_iter, learning_rate=1., cost_return=False, save_file_path=None, saving_params=False):
    if saving_params:
        func_args = inspect.getargvalues(inspect.currentframe())
    else:
        func_args = None
    if cost_return:
        all_costs = []
        for i in range(n_iter):
            all_costs.append(cost_function(theta, X, y))
            theta = gradient_descent(theta, X, y, learning_rate)
        if save_file_path:
            save_model(theta, save_file_path, params_saved=func_args)
        return theta, all_costs
    else:
        for i in range(n_iter):
            theta = gradient_descent(theta, X, y, learning_rate)  # 0.001 à mettre qd j'appelle ici
        if save_file_path:
            save_model(theta, save_file_path, params_saved=func_args)
        return theta


if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS)

    # Load the dataset

    FracturesDataset = pd.read_csv("..\\Datasets\\Fractures\\fractures.csv")

    # Drop the id column

    FracturesDataset.drop('id', axis=1, inplace=True)

    # Check min and max values for each feature column with continuous values

    for feature, feature_label in [('age', 'age'),
                                   ('height', 'height_cm'),
                                   ('weight', 'weight_kg'),
                                   ('waiting time', 'waiting_time'),
                                   ('bmd', 'bmd')]:

        print("Max {} \n".format(feature))
        print(FracturesDataset[feature_label].max(), "\n")
        print("Min {} \n".format(feature))
        print(FracturesDataset[feature_label].min(), "\n\n")

    # Print part of the data and data information

    print("Head of the data\n")
    print(FracturesDataset.head(), "\n")

    print("\nData info\n")
    print(FracturesDataset.info(), "\n")

    print("\nDescribe the data\n")
    print(FracturesDataset.describe(), "\n\n")

    # Create normalized data then scaling them in the range of the height feature

    scaled_age = (FracturesDataset['age'] / FracturesDataset['age'].max()) * (177 - 142) + 142
    scaled_weight = (FracturesDataset['weight_kg'] / FracturesDataset['weight_kg'].max()) * (177 - 142) + 142
    scaled_waiting_time = (FracturesDataset['waiting_time'] / FracturesDataset['waiting_time'].max()) * \
                          (177 - 142) + 142

    # For scaling the bmd, multiply by 100 to have pretty much the same range as the other variables, and also being
    # able to plot correctly this variable

    scaled_bmd = (FracturesDataset['bmd'] / FracturesDataset['bmd'].max()) * (177 - 142) * 100 + 142

    # Encoding target column with 0 (no fracture) and 1 (fracture), then scaling it in the range of the height feature

    scaled_target = np.array(FracturesDataset['fracture'] == 'fracture') * 177

    # 3D-plot of the target variable with respect to each combination of 2 continuous variables among the 4

    SCALED_FEATURES = {'age': scaled_age,
                       'weight_kg': scaled_weight,
                       'waiting_time': scaled_waiting_time,
                       'bmd': scaled_bmd}

    NOT_SCALED_FEATURES = {'age': FracturesDataset['age'],
                           'weight_kg': FracturesDataset['weight_kg'],
                           'waiting_time': FracturesDataset['waiting_time'],
                           'bmd': FracturesDataset['bmd']}

    for x, y in combinations(['age', 'weight_kg', 'waiting_time', 'bmd'], 2):

        plt.rcParams['figure.figsize'] = (8, 6)
        ax = plt.axes(projection='3d')

        floor_scaled_xmin = int(np.floor(SCALED_FEATURES[x].min()))
        floor_scaled_xmax = int(np.floor(SCALED_FEATURES[x].max()))
        scaled_xrange = int(floor_scaled_xmax - floor_scaled_xmin + 1)
        scaled_xstep = scaled_xrange // 10

        floor_xmin = int(np.floor(NOT_SCALED_FEATURES[x].min()))
        floor_xmax = int(np.floor(NOT_SCALED_FEATURES[x].max()))
        xrange = int(floor_xmax - floor_xmin + 1)
        xstep = xrange // 10

        xticks = [floor_scaled_xmin + i for i in range(0, scaled_xrange, scaled_xstep)]
        ax.set_xticks(xticks)
        if len(xticks) > len([i for i in range(0, xrange, xstep)]):
            xrange += xstep
        ax.set_xticklabels([floor_xmin + i for i in range(0, xrange, xstep)])

        floor_scaled_ymin = int(np.floor(SCALED_FEATURES[y].min()))
        floor_scaled_ymax = int(np.floor(SCALED_FEATURES[y].max()))
        scaled_yrange = int(floor_scaled_ymax - floor_scaled_ymin + 1)
        scaled_ystep = scaled_yrange // 10

        if y == 'bmd':

            floor_ymin = int(np.floor(NOT_SCALED_FEATURES[y].min() * 100))
            floor_ymax = int(np.floor(NOT_SCALED_FEATURES[y].max() * 100))
            yrange = int(floor_ymax - floor_ymin + 1)
            ystep = yrange // 10
            yticks = [floor_scaled_ymin + i for i in range(0, scaled_yrange, scaled_ystep)]
            ax.set_yticks(yticks)
            if len(yticks) > len([i for i in range(0, yrange, ystep)]):
                yrange += ystep
            ax.set_yticklabels([round(floor_ymin / 100 + i/100, 1) for i in range(0, yrange, ystep)])

        else:

            floor_ymin = int(np.floor(NOT_SCALED_FEATURES[y].min()))
            floor_ymax = int(np.floor(NOT_SCALED_FEATURES[y].max()))
            yrange = int(floor_ymax - floor_ymin + 1)
            ystep = yrange // 10

            yticks = [floor_scaled_ymin + i for i in range(0, scaled_yrange, scaled_ystep)]
            ax.set_yticks(yticks)
            if len(yticks) > len([i for i in range(0, yrange, ystep)]):
                yrange += ystep
            ax.set_yticklabels([floor_ymin + i for i in range(0, yrange, ystep)])

        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_zlabel("fracture")

        ax.scatter3D(SCALED_FEATURES[x], SCALED_FEATURES[y], scaled_target)
        plt.show()

    # Encode categorical features as numerical ones

    FracturesDataset['fracture'] = np.array(FracturesDataset['fracture'] == 'fracture', dtype='float64')

    _ = one_hot_encoding(FracturesDataset, FracturesDataset.select_dtypes(include='object').columns.to_list(),
                         replace=True)

    # Shuffle the dataset

    FracturesDataset = FracturesDataset.sample(frac=1).reset_index(drop=True)

    # Create a theta vector/array with the first value being the intercept equal to 1, and randomly initialized
    # elsewhere

    columns_nb = len(FracturesDataset.columns)
    rng = np.random.default_rng()

    theta = np.concatenate((np.array([1], dtype='float64'),
                            np.array([rng.random() for i in range(columns_nb - 1)], dtype='float64')))
    # Note that we do columns_nb - 1 because we exclude the target column

    # Add ones to FracturesDataset to deal with the intercept term in theta

    FracturesDataset = add_ones(FracturesDataset)

    # Split the data into a train set and a test set

    train_set, test_set, train_target, test_target = split_data(FracturesDataset, 80, 2)

    # Train the model

    n_iter = 1000

    theta, all_costs = train_model(theta, train_set, train_target, n_iter,
                                   learning_rate=0.0001,
                                   cost_return=True,
                                   save_file_path="model_saved\\Fractures\\saved_model.txt",
                                   saving_params=True)

    # Print the size of both the training and testing set

    print("\nSize of the training set : {}\n".format(len(train_set)))
    print("\nSize of the testing set : {}\n".format(len(test_set)))

    # Print the number of fractures in both of the two sets

    print("\nNumber of fractures in the training set : {}\n".format(np.sum(train_target)))
    print("Number of fractures in the testing set : {}\n".format(np.sum(test_target)))

    # Make predictions on the training set with the new theta coming from the trained model

    predictions = h(theta)(train_set)

    # Fixing the threshold for predictions to 0.6, i.e if a prediction value is >= 0.6, we consider the model has
    # predicted a fracture

    predictions[predictions >= 0.6] = 1
    predictions[predictions < 0.6] = 0

    # Print the number of fractures estimated by the model on the training set

    print("\nNumber of fractures estimated by the model on the training set : {}\n".format(np.sum(predictions)))

    # Print the accuracy of the model on the training set

    print("Train accuracy : {}\n".format(1 - np.abs(np.mean(predictions - train_target))))

    # Make predictions on the testing set with the new theta coming from the trained model

    predictions = h(theta)(test_set)

    predictions[predictions >= 0.6] = 1
    predictions[predictions < 0.6] = 0

    # Print the number of fractures estimated by the model on the testing set

    print("\nNumber of fractures estimated by the model on the testing set : {}\n".format(np.sum(predictions)))

    # Print the accuracy of the model on the testing set

    print("Test accuracy : {}\n".format(1 - np.abs(np.mean(predictions - test_target))))

    # Plot the cost function

    iterations = [i for i in range(n_iter)]
    plt.plot(iterations, all_costs, label="Cost function")
    plt.suptitle("Cost function against the number of iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Cost function")

    plt.show()

    # Making a grid of parameters for the logistic regression

    parameters_grid = {'n_iter': [1_000, 10_000, 30_000], 'learning_rate': [0.01, 0.001, 0.0001]}

    # Searching best parameters through a grid search

    grid_model = GridSearch(None, cost_function, train_model, parameters_grid)

    grid_model.fit(theta, train_set, train_target, best=True)
    best_model = grid_model.gridsearch_best()

    # Print the parameters of the best model

    print("\nBEST :", best_model)

    # Use the best model to make the predictions

    best_theta = best_model['best_theta']
    predictions = h(best_theta)(test_set)
    predictions[predictions >= 0.6] = 1
    predictions[predictions < 0.6] = 0

    # Print the number of fractures estimated by the best model on the testing set

    print("\nNumber of fractures estimated by the best model on the testing set : {}\n".format(np.sum(predictions)))

    # Print the accuracy of the best model on the testing set

    print("Test accuracy of the best model : {}\n".format(1 - np.abs(np.mean(predictions - test_target))))

    print('\nDummy Classifier (predicting only the absence of a fracture)\n')
    print(np.sum(test_target == np.array([0 for i in range(len(test_target))])) / len(test_target))
    print(np.sum(train_target == np.array([0 for i in range(len(train_target))])) / len(train_target))

