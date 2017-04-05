import numpy as np
import math
import random


def content_reader(input_file):
    """ this function reads the input file and return a list of all athletes records in one line per athlete format """
    f = open(input_file, "r")  # open the file for read
    data_list = f.readlines()  # this reads all the lines and returns a list of the strings
    f.close()
    data_list = data_list[1:]  # removes the first line that contains legends
    data_set = data_list[:]  # create a copy of the data_set

    # create the data set in form of a list of lists
    data_set_list = []
    for line in data_set:
        temp_list = line.split(',')
        data_set_list.append(temp_list)

    # provide a single record per athlete in the return list
    new_lst = [data_set_list[0]]
    recent_element = data_set_list[0]
    for element in data_set_list[1:]:
        if element[0] != recent_element[0]:
            new_lst.append(element)
            recent_element = element
    return new_lst


def content_extractor(lst):
    """this function extracts and returns desired features from list file"""
    # creating empty lists for output files
    rec_2003 = []
    rec_2004 = []
    rec_2005 = []
    rec_2006 = []
    rec_2007 = []
    rec_2008 = []
    rec_2009 = []
    rec_2010 = []
    rec_2011 = []
    rec_2012 = []
    rec_2013 = []
    rec_2014 = []
    rec_2015 = []
    rec_2016 = []
    total_number = []
    sex_m = []
    sex_f = []

    for v in lst:
        rec_2003.append(int(v[11]))
        rec_2004.append(int(v[12]))
        rec_2005.append(int(v[13]))
        rec_2006.append(int(v[14]))
        rec_2007.append(int(v[15]))
        rec_2008.append(int(v[16]))
        rec_2009.append(int(v[17]))
        rec_2010.append(int(v[18]))
        rec_2011.append(int(v[19]))
        rec_2012.append(int(v[20]))
        rec_2013.append(int(v[21]))
        rec_2014.append(int(v[22]))
        rec_2015.append(int(v[23]))
        rec_2016.append(int(v[24]))
        total_number.append(int(v[10]))
        if v[3] == "M":  # if athlete is male
            sex_m.append(1)
            sex_f.append(0)
        else:
            sex_m.append(0)
            sex_f.append(1)

    return rec_2003, rec_2004, rec_2005, rec_2006, rec_2007, rec_2008, rec_2009, rec_2010, rec_2011, rec_2012, \
           rec_2013, rec_2014, rec_2015, rec_2016, sex_m, sex_f, normalization(total_number)


def product(lst1, lst2):
    result = []
    for v in range(len(lst1)):
        result.append(lst1[v] * lst2[v])
    return result


def normalization(x):
    """this function normalized the data using the min-max scaling approach"""
    """ X_normalized = (x_original - min_value) / (max_value - min_value)"""
    max_value = np.amax(x)
    min_value = np.amin(x)
    denominator = max_value - min_value
    normalized_set = []

    for i in range(len(x)):
        numerator = x[i] - min_value
        x_normalized = numerator / denominator
        normalized_set.append(x_normalized)

    return normalized_set


def gradient_descent(x_in, y):
    """this function calculate logistic regression for given X and Y and returns weight vector W"""
    # crating initial weight vector
    w0 = []
    for i in range(len(x_in) + 1):
        w0.append(1)

    x = x_in[:]
    all_one = []
    for i in range(len(x[0])):
        all_one.append(1)
    x.append(all_one)
    x = np.array(x)
    x = np.transpose(x)
    k = 0
    w = w0
    while True:
        k += 1
        alpha_k = 0.0001
        summation = 0
        for i in range(len(x)):
            a = np.dot(np.transpose(w), x[i])
            sigmoid = 1 / (1 + math.exp(-a))
            summation += np.dot(x[i], y[i] - sigmoid)
        new_w = w + alpha_k * summation
        if np.allclose(new_w, w, rtol=1e-04, atol=1e-05, equal_nan=False):
            return new_w
        print(w)
        w = new_w


def predictor(x, w):
    y = []
    # create an all 1 vector
    all_one = []
    for i in range(len(x[0])):
        all_one.append(1)

    x_temp = x[:]
    x_temp.append(all_one)
    x_temp = np.array(x_temp)
    # print(np.shape(x_temp))

    temp = []
    for i in range(len(x_temp[0])):
        for j in range(len(x_temp)):
            temp.append(x_temp[j][i])

        # compute the predicted value using the weights
        y_predicted = np.dot(temp, w)
        temp = []
        if y_predicted > 0:
            y_predicted = 1
        else:
            y_predicted = 0
        # the first one is the true mean, second one is the predicted
        y.append(y_predicted)

    return y


def file_maker(name, lst):
    import csv
    with open(name, 'w', newline='') as csv_file:
        file_writer = csv.writer(csv_file, delimiter=',', quotechar=',')
        file_writer.writerow(["ID", "Y2"])
        for i in range(len(lst)):
            file_writer.writerow([str(i+1), str(lst[i])])

all_athlete_records = content_reader("modified_file_4.csv")
t_2003, t_2004, t_2005, t_2006, t_2007, t_2008, t_2009, t_2010, t_2011, t_2012, t_2013, t_2014, t_2015, \
t_2016, t_sex_m, t_sex_f, t_total_number = content_extractor(all_athlete_records)
training_features_list = [t_2014, t_2013, t_2012, t_2011, product(t_2014, t_2013), product(t_2014, t_2012),
                  product(t_2012, t_2013), product(t_2011, t_2013), product(t_2011, t_2014)]
training_output = t_2016
# w = gradient_descent(training_features_list, training_output)
w = [-0.16657548, -1.01372817, -1.38412009,  2.34516041,  2.66771123 , 2.51064349, -2.60883599, -1.97542305]
input_for_prediction = [t_2016, t_2015, t_2014, product(t_2016, t_2015), product(t_2016, t_2014),
                        product(t_2015, t_2014), product(t_2016, product(t_2014, t_2015))]
y = predictor(input_for_prediction, w)
file_maker("logistic_regression_output2.csv", y)
