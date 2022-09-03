import matplotlib.pyplot as plt
import inspect
from PostProcessing.Postprocessing import*
from PreProcessing.Preprocessing import*


# Define the functions for linear regression

def h(theta):
    def h_theta(x):
        return np.dot(x, theta)
    return h_theta


def cost_function(theta, X, y):
    return (1 / (2 * X.shape[0])) * np.sum(np.square(h(theta)(X) - y))


def gradient_descent(theta, x, y, learning_rate=1.):
    one_over_mean = (1 / x.shape[0])
    return theta - learning_rate * one_over_mean * np.dot(h(theta)(x) - y, x)


def normal_equations(X, y):
    return (np.linalg.inv(X.T @ X) @ X.T) @ y


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

    BodyMeasuresDataset = pd.read_csv("..\\Datasets\\SOCR-HeightWeight\\SOCR-HeightWeight.csv")

    # Check min and max values

    print("Max height / max weight\n")
    print(BodyMeasuresDataset['Height(Inches)'].max(), BodyMeasuresDataset['Weight(Pounds)'].max(), "\n")
    print("\nMin height / min weight\n")
    print(BodyMeasuresDataset['Height(Inches)'].min(), BodyMeasuresDataset['Weight(Pounds)'].min(), "\n\n")

    # Plot the weight as a function of the height for 100 rows chosen between 0 and 1000 with a step of 10

    index_plots = [i for i in range(0, 1000, 10)]
    heights = BodyMeasuresDataset.iloc[index_plots, [1]]
    weights = BodyMeasuresDataset.iloc[index_plots, [2]]

    plt.scatter(heights, weights)
    plt.suptitle("Linear regression\n Weight depending on Height")
    plt.axis([60, 76, 78, 171])
    plt.xlabel('Heights (Inches)')
    plt.ylabel('Weights (Pounds)')
    plt.xticks([i for i in range(60, 76, 1)])
    plt.yticks([i for i in range(75, 171, 10)])
    plt.show()

    # Drop Index column (we get rid of the Index column)

    BodyMeasuresDataset.drop('Index', axis=1, inplace=True)

    # Print part of the data and data informations

    print("Head of the data\n")
    print(BodyMeasuresDataset.head(), "\n")

    print("\nData info\n")
    print(BodyMeasuresDataset.info(), "\n")

    print("\nDescribe the data\n")
    print(BodyMeasuresDataset.describe(), "\n\n")

    # Shuffle the dataset

    BodyMeasuresDataset = BodyMeasuresDataset.sample(frac=1).reset_index(drop=True)

    # Create a theta vector/array with the first value being the intercept equal to 1, and randomly initialized
    # elsewhere

    columns_nb = len(BodyMeasuresDataset.columns)
    rng = np.random.default_rng()

    theta = np.concatenate((np.array([1], dtype='float64'),
                            np.array([rng.random() for i in range(columns_nb - 1)], dtype='float64')))
    # Note that we do columns_nb - 1 because we exclude the target column

    # Add ones to BodyMeasuresDataset to deal with the intercept term in theta

    BodyMeasuresDataset = add_ones(BodyMeasuresDataset)

    # Split the data into a train set and a test set

    train_set, test_set, train_target, test_target = split_data(BodyMeasuresDataset, 80, 2)

    # Mean normalization and feature scaling

    train_target_mean = train_target.mean()
    train_target_std = train_target.std()
    train_set_mean = train_set['Height(Inches)'].mean()
    train_set_std = train_set['Height(Inches)'].std()
    test_target_mean = test_target.mean()
    test_target_std = test_target.std()
    test_set_mean = test_set['Height(Inches)'].mean()
    test_set_std = test_set['Height(Inches)'].std()

    train_target = (train_target - train_target.mean()) / train_target.std()
    train_set['Height(Inches)'] = (train_set['Height(Inches)'] - train_set['Height(Inches)'].mean()) / \
                                  train_set['Height(Inches)'].std()
    test_target = (test_target - test_target.mean()) / test_target.std()
    test_set['Height(Inches)'] = (test_set['Height(Inches)'] - test_set['Height(Inches)'].mean()) / \
                                 test_set['Height(Inches)'].std()

    # Train the model

    theta, all_costs = train_model(theta, train_set, train_target, 3000,
                                   learning_rate=0.001,
                                   cost_return=True,
                                   save_file_path="model_saved\\SOCR\\saved_model.txt",
                                   saving_params=True)

    # Make predictions on the test set with the new theta coming from the trained model

    predictions = h(theta)(test_set)  # be careful not to rescale the data before making predictions

    # Rescaling predictions

    predictions = predictions * test_target_std + test_target_mean

    # Plot the predictions and some data points of the test target set

    plt.scatter(test_set['Height(Inches)'] * test_set_std + test_set_mean,
                predictions,
                label="Predictions")
    plt.scatter(test_set['Height(Inches)'][0:5000:100] * test_set_std + test_set_mean,
                test_target[0:5000:100] * test_target_std + test_target_mean,
                label="Sample of the test set")

    # Plot the line of the linear regression

    horizontal_axis = np.linspace(60, 76, 200)

    scaled_x_values = (horizontal_axis - test_set_mean) / test_set_std  # Be careful to scale the data
    scaled_x_values = add_ones(scaled_x_values).T  # Add the intercept 1 and transpose the result

    plt.plot(horizontal_axis,
             (h(theta)(scaled_x_values)) * test_target_std + test_target_mean,
             label="Regression line")  # Be careful to rescale
    # the values returned by h_theta with regard to the scaling of the test target set for consistency

    plt.suptitle("Linear regression\n Weight depending on Height")
    plt.legend(prop={"size": 7.7}, loc='upper left')
    plt.xlabel('Heights (Inches)')
    plt.ylabel('Weights (Pounds)')

    plt.show()

    # Plot the cost function

    iterations = [i for i in range(3000)]
    plt.plot(iterations, all_costs, label="Cost function")
    plt.suptitle("Cost function against the number of iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Cost function")

    plt.show()
