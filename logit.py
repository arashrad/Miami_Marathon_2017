import numpy as np
import math


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
# extracting data from file
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
    """this function calculate the interception of two vectors"""
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
        alpha_k = 0.001
        if k > 100:
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


def calculate_predicted_y(x, w, y):
    """this function calculates the predicted value of y"""
    # create a list storing the the predicted value and the true value
    true_vs_pred = []
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
        true_vs_pred.append([y[i], y_predicted])

    return true_vs_pred


def file_maker(name, lst):
    """this function takes a list and name and creates a csv file"""
    import csv
    with open(name, 'w', newline='') as csv_file:
        file_writer = csv.writer(csv_file, delimiter=',', quotechar=',')
        file_writer.writerow(["Model Number", "TP", "FP", "TN", "FN", "Error_rate"])
        for i in range(len(lst)):
            temp_list = [str(i+1)]
            temp_list.append(str(lst[i][0]))
            temp_list.append(str(lst[i][1]))
            temp_list.append(str(lst[i][2]))
            temp_list.append(str(lst[i][3]))
            temp_list.append(str(lst[i][4]))
            file_writer.writerow(temp_list)


# the  main function
def main():
    all_athlete_records = content_reader("modified_file_4.csv")

    t_2003, t_2004, t_2005, t_2006, t_2007, t_2008, t_2009, t_2010, t_2011, t_2012, t_2013, t_2014, t_2015, \
    t_2016, t_sex_m, t_sex_f, t_total_number = content_extractor(all_athlete_records)
    # v_2003, v_2004, v_2005, v_2006, v_2007, v_2008, v_2009, v_2010, v_2011, v_2012, v_2013, v_2014, v_2015, \
    # v_2016, v_sex_m, v_sex_f, v_total_number = content_extractor(validation_set)

    y_t = t_2011
    y_v = t_2012

    input_1_t = [t_2014]
    input_2_t = [t_2014, t_2013, t_2012]
    input_3_t = [t_2014, t_2012, t_2011]
    input_4_t = [t_2014, t_2012, t_2011, t_2010]
    input_5_t = [t_2014, t_2012, t_2011, t_2010, t_2009]
    input_6_t = [t_2014, t_2013, t_2012, t_2011, t_2010]
    input_7_t = [t_2014, t_2012, t_2011, t_2010, t_2009, t_sex_f, t_sex_m]
    input_8_t = [t_2014, t_2012, t_2011, t_2010, t_2009, t_2008, t_2007]
    input_9_t = [t_2014, t_2012, product(t_2014, t_2012)]
    input_10_t = [t_2014, t_2013, t_2012, product(t_2014, t_2013), product(t_2014, t_2012), product(t_2012, t_2013)]
    input_11_t = [t_2014, t_2013, t_2012, t_2011, product(t_2014, t_2013), product(t_2014, t_2012),
                  product(t_2012, t_2013)]
    input_12_t = [t_2014, t_2013, t_2012, t_2011, product(t_2014, t_2013), product(t_2014, t_2012),
                  product(t_2012, t_2013), product(t_2011, t_2013), product(t_2011, t_2014)]
    input_13_t = [t_2014, t_2013, t_2011, t_2010, t_2009, t_2008,  product(t_2014, t_2012), product(t_2014, t_2011),
                  product(t_2014, t_2010), product(t_2011, t_2013), product(t_2010, t_2013), product(t_2013, t_2009)]
    input_14_t = [t_2014, t_2012, t_2011, t_2010, t_2009, product(t_2014, t_2012)]
    input_15_t = [t_2014, t_2012, t_2011, t_2010, product(t_2014, t_2012), product(t_2014, t_2011),
                  product(t_2012, t_2010), product(t_2011, t_2012)]
    input_16_t = [t_2014, t_2013, t_2012, product(t_2014, t_2013), product(t_2014, t_2012), product(t_2013, t_2012)]
    input_17_t = [t_2010, t_2009, t_2008, product(t_2010, t_2009), product(t_2010, t_2008), product(t_2009, t_2008)]
    input_18_t = [t_2011, t_2010, t_2009, product(t_2010, t_2011)]
    input_19_t = [t_2010, t_2009, t_2008, product(t_2010, t_2009), product(t_2010, t_2008), product(t_2009, t_2008),
                  product(t_2010, product(t_2008, t_2009))]


    input_list_t = [input_19_t]

    input_1_v = [t_2015]
    input_2_v = [t_2015, t_2014, t_2012]
    input_3_v = [t_2015, t_2014, t_2013]
    input_4_v = [t_2015, t_2014, t_2013, t_2012]
    input_5_v = [t_2015, t_2014, t_2013, t_2012, t_2011]
    input_6_v = [t_2015, t_2014, t_2012, t_2011, t_2010]
    input_7_v = [t_2015, t_2014, t_2013, t_2012, t_2011, t_2010, t_sex_m]
    input_8_v = [t_2015, t_2014, t_2013, t_2012, t_2011, t_2010, t_2009]
    input_9_v = [t_2015, t_2014, product(t_2015, t_2014)]
    input_10_v = [t_2015, t_2014, t_2013, product(t_2015, t_2014), product(t_2015, t_2013), product(t_2013, t_2014)]
    input_11_v = [t_2015, t_2014, t_2013, t_2012, product(t_2015, t_2014), product(t_2015, t_2013),
                  product(t_2013, t_2014)]
    input_12_v = [t_2015, t_2014, t_2013, t_2012, product(t_2015, t_2014), product(t_2015, t_2013),
                  product(t_2013, t_2014), product(t_2012, t_2014), product(t_2012, t_2015)]
    input_13_v = [t_2015, t_2014, t_2012, t_2011, t_2010, t_2009, product(t_2015, t_2014), product(t_2015, t_2012),
                  product(t_2015, t_2011), product(t_2012, t_2014), product(t_2011, t_2014), product(t_2014, t_2010)]
    input_14_v = [t_2015, t_2014, t_2012, t_2011, t_2010, product(t_2014, t_2012)]
    input_15_v = [t_2015, t_2014, t_2012, t_2011, product(t_2015, t_2012), product(t_2015, t_2011),
                  product(t_2015, t_2010), product(t_2011, t_2012)]
    input_16_v = [t_2015, t_2014, t_2013, product(t_2015, t_2014), product(t_2015, t_2013), product(t_2014, t_2013)]
    input_17_v = [t_2011, t_2010, t_2009, product(t_2011, t_2010), product(t_2011, t_2009), product(t_2010, t_2009)]
    input_18_v = [t_2011, t_2010, t_2009, product(t_2009, t_2010)]
    input_19_v = [t_2011, t_2010, t_2009, product(t_2011, t_2010), product(t_2011, t_2009), product(t_2010, t_2009),
                  product(t_2011, product(t_2009, t_2010))]

    input_list_v = [input_19_v]

    weight_list = []
    t_predict_result_list = []
    v_predict_result_list = []
    for item in input_list_t:
        w = gradient_descent(item, y_t)
        weight_list.append(w)
# finding training error and write it to a file
    for i in range(len(weight_list)):
        prediction = calculate_predicted_y(input_list_t[i], weight_list[i], y_t)
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for v in prediction:
            if v[0] == 1 and v[1] == 1:
                tp += 1
            elif v[0] == 1 and v[1] == 0:
                fn += 1
            elif v[0] == 0 and v[1] == 0:
                tn += 1
            else:
                fp += 1
        error_rate = (fp + fn) / (fp + fn + tn + tp)
        t_predict_result_list.append([tp, fp, tn, fn, error_rate])
        print(tp, fp, tn, fn, error_rate)
    file_maker("logistic_training_result_19.csv", t_predict_result_list)

    # finding validation error and write it to a file
    for i in range(len(weight_list)):
        prediction = calculate_predicted_y(input_list_v[i], weight_list[i], y_v)
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for v in prediction:
            if v[0] == 1 and v[1] == 1:
                tp += 1
            elif v[0] == 1 and v[1] == 0:
                fn += 1
            elif v[0] == 0 and v[1] == 0:
                tn += 1
            else:
                fp += 1
        error_rate = (fp + fn)/(fp + fn + tn + tp)
        v_predict_result_list.append([tp, fp, tn, fn, error_rate])
        print(tp, fp, tn, fn, error_rate)
        file_maker("logistic_valid_result_19.csv", v_predict_result_list)

# executing the main function
main()
