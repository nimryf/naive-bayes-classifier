import numpy as np
import os

# Example use of the classifier available at the bottom.
# If this script is run as given, it should print classification accuracy
#  against both training and testing datasets.


def test_classifier(training_spam):
    """
    Tests the classifier against training data. More information of individual functions available below.

    Input
        training_spam: dataset to train and test the classifier with.

    Output
        training_set_accuracy: accuracy of classifications of seen data.
    """
    log_class_priors = estimate_log_class_priors(training_spam)
    log_class_conditional_likelihoods = estimate_log_class_conditional_likelihoods(
        training_spam, alpha=1.0)

    class_predictions = predict(
        training_spam[:, 1:], log_class_priors, log_class_conditional_likelihoods)

    # Check data type(s)
    assert(isinstance(class_predictions, np.ndarray))

    # Check shape of numpy array
    assert(class_predictions.shape == (1000,))

    # Check data type of array elements
    assert(np.all(np.logical_or(class_predictions == 0, class_predictions == 1)))

    # Check accuracy function
    true_classes = training_spam[:, 0]
    training_set_accuracy = accuracy(class_predictions, true_classes)
    assert(isinstance(training_set_accuracy, float))
    assert(0 <= training_set_accuracy <= 1)

    return training_set_accuracy


def test_classifier_2(training_spam, testing_spam):
    """
    Tests the classifier against testing data. More information of individual functions available below.

    Input
        training_spam: dataset to train the classifier with.
        testing_spam: dataset to test the classifier with.

    Output
        testing_set_accuracy: accuracy of classsifications of unseen data.
    """
    log_class_priors = estimate_log_class_priors(training_spam)
    log_class_conditional_likelihoods = estimate_log_class_conditional_likelihoods(
        training_spam, alpha=1.0)

    class_predictions = predict(
        testing_spam[:, 1:], log_class_priors, log_class_conditional_likelihoods)
    testing_set_accuracy = accuracy(class_predictions, testing_spam[:, 0])

    return testing_set_accuracy


def estimate_log_class_priors(data):
    """
    Given a data set with binary response variable (0s and 1s) in the
    left-most column, calculate the logarithm of the empirical class priors,
    that is, the logarithm of the proportions of 0s and 1s:
    log(P(C=0)) and log(P(C=1))

    Input
        data: a two-dimensional numpy-array with shape = [n_samples, 1 + n_features]
                 the first column contains the binary response (coded as 0s and 1s).

    Output
        log_class_priors: a numpy array of length two
    """
    ones, zeroes, responses = 0, 0, [x[0] for x in data]

    for j in responses:
        if j == 1.0:
            ones += 1
        elif j == 0.0:
            zeroes += 1
        else:
            print('Error: Data entry not a one or zero.')

    ones_prior = ones / (ones + zeroes)
    zeroes_prior = zeroes / (ones + zeroes)

    log_class_priors = np.array([np.log(zeroes_prior), np.log(ones_prior)])

    return log_class_priors


def estimate_log_class_conditional_likelihoods(data, alpha=1.0):
    """
    Given a data set with binary response variable (0s and 1s) in the
    left-most column and binary features, calculate the empirical
    class-conditional likelihoods, that is,
    log(P(w_i | c_j)) for all features i and both classes (j in {0, 1}).

    Assumes a multinomial feature distribution and uses Laplace smoothing
    if alpha > 0.

    Input
        data: a two-dimensional numpy-array with shape = [n_samples, n_features]

    Output
        theta: a numpy array of shape = [2, n_features]. theta[j, i] corresponds to the
            logarithm of the probability of feature i appearing in a sample belonging 
            to class j.
    """
    index, num_features = 0, len(data[0]) - 1

    class0_features = np.asarray([x[1:] for x in data if (x[0] == 0)])
    class1_features = np.asarray([x[1:] for x in data if (x[0] == 1)])

    class0_feature_sums = class0_features.sum(axis=0)
    class1_feature_sums = class1_features.sum(axis=0)

    class0_sum = class0_feature_sums.sum(axis=0)
    class1_sum = class1_feature_sums.sum(axis=0)

    theta = np.empty([2, num_features])

    for each_sum in class0_feature_sums:
        theta[0, index] = np.log(
            (each_sum + alpha) / (class0_sum + (alpha * num_features)))
        index += 1

    index = 0
    for each_sum in class1_feature_sums:
        theta[1, index] = np.log(
            (each_sum + alpha) / (class1_sum + (alpha * num_features)))
        index += 1

    return theta


def predict(new_data, log_class_priors, log_class_conditional_likelihoods):
    """
    Given a new data set with binary features, predict the corresponding
    response for each instance (row) of the new_data set.

    Input
        new_data: a two-dimensional numpy-array with shape = [n_test_samples, n_features].
        log_class_priors: a numpy array of length 2.
        log_class_conditional_likelihoods: a numpy array of shape = [2, n_features].
            theta[j, i] corresponds to the logarithm of the probability of feature i appearing
            in a sample belonging to class j.
    Output
        class_predictions: a numpy array containing the class predictions for each row
        of new_data.
    """
    num_messages, num_features = len(new_data), len(new_data[0])
    class_predictions = np.zeros(num_messages)

    for i in range(num_messages):
        ham, spam = log_class_priors[0], log_class_priors[1]

        for j in range(num_features):
            ham += new_data[i, j] * log_class_conditional_likelihoods[0, j]
            spam += new_data[i, j] * log_class_conditional_likelihoods[1, j]

        if spam > ham:
            class_predictions[i] = 1

    return class_predictions


def accuracy(y_predictions, y_true):
    """
    Calculate the accuracy.

    Input
        y_predictions: a one-dimensional numpy array of predicted classes (0s and 1s).
        y_true: a one-dimensional numpy array of true classes (0s and 1s).

    Output
        acc: accuracy, a float between 0 and 1 
    """
    t, c = len(y_predictions), 0
    for i in range(t):
        if (y_predictions[i] == y_true[i]):
            c = c + 1
    acc = c / t
    return acc


def test_training_spam():
    # This is a test function, outputs true or false.
    # Use to check whether data is available.
    try:
        training_spam = np.loadtxt(
            open("data/training_spam.csv"), delimiter=",")
        print("Shape of the spam training data set:", training_spam.shape)
        print(training_spam)
        return True
    except:
        print("Set not available or invalid!")
        return False


def test_priors(training_spam):
    # This is a test function, outputs true or false.
    # You can use this cell to check whether the returned objects of your function are of the right data type.
    try:
        training_spam = np.loadtxt(
            open("data/training_spam.csv"), delimiter=",")
        log_class_priors = estimate_log_class_priors(training_spam)
        print("result", log_class_priors)

        # Check length
        assert(len(log_class_priors) == 2)

        # Check whether the returned object is a numpy.ndarray
        assert(isinstance(log_class_priors, np.ndarray))

        # Check wehther the values of this numpy.array are floats.
        assert(log_class_priors.dtype == float)

        return True

    except:
        print("Something went wrong!")
        return False


def test_likelihoods(training_spam):
    # This is a test function, outputs true or false.
    # You can use this cell to check whether the returned objects of your function are of the right data type.

    try:
        log_class_conditional_likelihoods = estimate_log_class_conditional_likelihoods(
            training_spam, alpha=1.0)
        print("result", log_class_conditional_likelihoods)

        # Check data type(s)
        assert(isinstance(log_class_conditional_likelihoods, np.ndarray))

        # Check shape of numpy array
        assert(log_class_conditional_likelihoods.shape == (2, 54))

        # Check data type of array elements
        assert(log_class_conditional_likelihoods.dtype == float)

        print("Test successful")
    except:
        print("Something went wrong!")
        return False


# Example use of classifier.
here = os.path.dirname(os.path.abspath(__file__))

training_spam = os.path.join(here, "data\\training_spam.csv")
testing_spam = os.path.join(here, "data\\testing_spam.csv")

train_file = np.loadtxt(open(training_spam, "r"), delimiter=",")
test_file = np.loadtxt(open(testing_spam, "r"), delimiter=",")

print("Accuracy against training set: ", 100 *
      test_classifier(train_file), "%\n")

print("Accuracy against testing set: ", 100 *
      test_classifier_2(train_file, test_file), "%\n")
