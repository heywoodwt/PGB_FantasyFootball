# LOAD THESE PACKAGES
from datascience import *
import numpy as np
import matplotlib
%matplotlib inline
import matplotlib.pyplot as plots
plots.style.use('fivethirtyeight')

# LOAD THE CSV FILE (FILE.csv IS FILE NAME)
csv_file_data = Table.read_table('FILE.csv')
csv_file_data  # CHANGE TO FIT THE .csv FILE NAME


# CONVERT ARRAY TO STANDARD UNITS (X IS ARRAY)
def standard_units(any_numbers):
    "Convert any array of numbers to standard units."
    return (any_numbers - np.mean(any_numbers)) / np.std(any_numbers)


# CORRELATION (T IS TABLE, X IS LABEL OF X COLUMN, Y IS LABEL OF Y COLUMN)
def correlation(t, x, y):
    return np.mean(standard_units(t.column(x)) * standard_units(t.column(y)))


# SLOPE OF REGRESSION LINE (T IS TABLE, LABEL_X IS LABEL OF X COLUMN, LABEL_Y IS LABEL OF Y COLUMN)
def slope(t, label_x, label_y):
    r = correlation(t, label_x, label_y)
    return r * np.std(t.column(label_y)) / np.std(t.column(label_x))


# INTERCEPT OF REGRESSION LINE (T IS TABLE, LABEL_X IS LABEL OF X COLUMN, LABEL_Y IS LABEL OF Y COLUMN)
def intercept(t, label_x, label_y):
    return np.mean(t.column(label_y)) - slope(t, label_x, label_y) * np.mean(t.column(label_x))


# LINE OF BEST FIT (TABLE IS TABLE, X IS LABEL OF X COLUMN, Y IS LABEL OF Y COLUMN) *DOES NOT GRAPH THE LINE*
def fit(table, x, y):
    """Return the height of the regression line at each x value."""
    a = slope(table, x, y)
    b = intercept(table, x, y)
    return a * table.column(x) + b


# RESIDUALS (TABLE IS TABLE, X IS LABEL OF X COLUMN, Y IS LABEL OF Y COLUMN)
def residual(table, x, y):
    return table.column(y) - fit(table, x, y)


# PLOT OF RESIDUALS (TABLE IS TABLE, X IS LABEL OF X COLUMN, Y IS LABEL OF Y COLUMN)
def residual_plot(table, x, y):
    x_array = table.column(x)
    t = Table().with_columns(
        x, x_array,
        'residuals', residual(table, x, y)
    )
    t.scatter(x, 'residuals', color='r')
    xlims = make_array(min(x_array), max(x_array))
    plots.plot(xlims, make_array(0, 0), color='darkblue', lw=4)
    plots.title('Residual Plot')


# BOOTSTRAP SLOPE (TABLE IS TABLE, X IS LABEL OF X COLUMN, Y IS LABEL OF Y COLUMN, REPETITIONS IS NUMBER OF REPETITIONS)
# PLOT THE DISTRIBUTION OF THE BOOTSTRAP SLOPE
def bootstrap_slope(table, x, y, repetitions):
    # For each repetition:
    # Bootstrap the scatter, get the slope of the regression line,
    # augment the list of generated slopes
    slopes = make_array()
    for i in np.arange(repetitions):
        bootstrap_sample = table.sample()
        bootstrap_slope = slope(bootstrap_sample, x, y)
        slopes = np.append(slopes, bootstrap_slope)

    # Find the endpoints of the 95% confidence interval for the true slope
    left = percentile(2.5, slopes)
    right = percentile(97.5, slopes)

    # Slope of the regression line from the original sample
    observed_slope = slope(table, x, y)

    # Display results
    Table().with_column('Bootstrap Slopes', slopes).hist(bins=20)
    plots.plot(make_array(left, right), make_array(0, 0), color='yellow', lw=8);
    print('Slope of regression line:', observed_slope)
    print('Approximate 95%-confidence interval for the true slope:')
    print(left, right)


# FITTED VALUE COMPUTES THE PREDICTION OF THE REGRESSION LINE AT A GIVEN X VALUE (TABLE IS TABLE, X IS LABEL OF X COLUMN, Y IS LABEL OF Y COLUMN, GIVEN_X IS X VALUE)
def fitted_value(table, x, y, given_x):
    a = slope(table, x, y)
    b = intercept(table, x, y)
    return a * given_x + b


# BOOTSTRAP PREDICTION (TABLE IS TABLE, X IS LABEL OF X COLUMN, Y IS LABEL OF Y COLUMN, NEW_X IS X VALUE, REPETITIONS IS NUMBER OF REPETITIONS)
def bootstrap_prediction(table, x, y, new_x, repetitions):
    # For each repetition:
    # Bootstrap the scatter;
    # get the regression prediction at new_x;
    # augment the predictions list
    predictions = make_array()
    for i in np.arange(repetitions):
        bootstrap_sample = table.sample()
        bootstrap_prediction = fitted_value(bootstrap_sample, x, y, new_x)
        predictions = np.append(predictions, bootstrap_prediction)

    # Find the ends of the approximate 95% prediction interval
    left = percentile(2.5, predictions)
    right = percentile(97.5, predictions)

    # Prediction based on original sample
    original = fitted_value(table, x, y, new_x)

    # Display results
    Table().with_column('Prediction', predictions).hist(bins=20)
    plots.xlabel('predictions at x=' + str(new_x))
    plots.plot(make_array(left, right), make_array(0, 0), color='yellow', lw=8);
    print('Height of regression line at x=' + str(new_x) + ':', original)
    print('Approximate 95%-confidence interval:')
    print(left, right)


# DISTANCE BETWEEN POINT 1 AND 2 (POINT1 IS AN ARRAY OF X AND Y COORDINATES, POINT2 IS AN ARRAY OF X AND Y COORDINATES)
def distance(point1, point2):
    """Returns the distance between point1 and point2
    where each argument is an array
    consisting of the coordinates of the point"""
    return np.sqrt(np.sum((point1 - point2) ** 2))


# DISTANCES BETWEEN EACH POINT IN THE TRAINING SET AND A GIVEN POINT (TRAINING IS AN , NEW_POINT)
def all_distances(training, new_point):
    """Returns an array of distances
    between each point in the training set
    and the new point (which is a row of attributes)"""
    attributes = training.drop('Class')

    def distance_from_point(row):
        return distance(np.array(list(new_point)), np.array(list(row)))

    return attributes.apply(distance_from_point)


# ADDS COLUMN WITH THE DISTANCE BETWEEN EACH POINT IN THE TRAINING SET AND A GIVEN POINT (TRAINING, NEW_POINT) *USES ALL DISTANCES FUNCTION*
def table_with_distances(training, new_point):
    """Augments the training table
    with a column of distances from new_point"""
    return training.with_column('Distance', all_distances(training, new_point))


# FIND THE CLOSEST K POINTS (TRAINING IS , NEW_POINT, K IS THE NUMBER OF POINTS TO COMPARE TO (USE AN ODD NUMBER))
def closest(training, new_point, k):
    """Returns a table of the k rows of the augmented table
    corresponding to the k smallest distances"""
    with_dists = table_with_distances(training, new_point)
    sorted_by_distance = with_dists.sort('Distance')
    topk = sorted_by_distance.take(np.arange(k))
    return topk


# FIND THE MOST COMMON CLASS AMONG THE K CLOSEST POINT (TRAINING IS , NEW_POINT, K IS THE NUMBER OF POINTS TO COMPARE TO (USE AN ODD NUMBER))
def majority(topkclasses):
    ones = topkclasses.where('Class', are.equal_to(1)).num_rows
    zeros = topkclasses.where('Class', are.equal_to(0)).num_rows
    if ones > zeros:
        return 1
    else:
        return 0


# CLASSIFY A NEW POINT (TRAINING IS A SET OF POINTS TO COMPARE AGAINST, NEW_POINT IS THE , K IS THE NUMBER OF NEAREST NEIGHBORS(USE AN ODD NUMBER))
def classify(training, new_point, k):
    closestk = closest(training, new_point, k)
    topkclasses = closestk.select('Class')
    return majority(topkclasses)


# FIND THE PROPORTION OF CORRECT PREDICTIONS (TRAINING IS THE TRAINING SET, TEST IS THE TEST SET, K IS NUMBER OF NEAREST NEIGHBORS TO TEST (USE AN ODD NUMBER))
def evaluate_accuracy(training, test, k):
    """Return the proportion of correctly classified examples
    in the test set"""
    test_attributes = test.drop('Class')
    num_correct = 0
    for i in np.arange(test.num_rows):
        c = classify(training, test_attributes.row(i), k)
        num_correct = num_correct + (c == test.column('Class').item(i))
    return num_correct / test.num_rows


# CONVERT LIST TO AN ARRAY (ALL THE ELEMENTS IN THE LIST MUST BE THE SAME TYPE)
np.array(list)
