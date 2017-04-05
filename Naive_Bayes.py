import numpy as np
import numpy.linalg 
import matplotlib.pyplot as plt
import math
import process
import random
import csv

# extract data from csv file
def content_Reader(filename):
    """this function reads the csv file then output the data"""
    f = open(filename, "r") # open file to read it
    data_list = f.readlines() # read all the lines 
    f.close()
    data_list = data_list[1:] # remove first row which are the features name

    data_set = []
    previous = 0

    for i in data_list:
        temp_list = i.split(',')
        if temp_list[0] != previous:
            data_set.append(i)
            previous = temp_list[0]

    return data_set

def content_extractor(data_list): 
    # creating empty lists for output files
    athlete_id = []
    athlete_name = []
    athlete_age = []
    sex_m = []
    sex_f = []
    sex_u = []
    match_year = []
    athlete_rank = []
    athlete_time_in_hours = []
    athlete_average = []
    athlete_participation = []
    athlete_2003 = []
    athlete_2004 = []
    athlete_2005 = []
    athlete_2006 = []
    athlete_2007 = []
    athlete_2008 = []
    athlete_2009 = []
    athlete_2010 = []
    athlete_2011 = []
    athlete_2012 = []
    athlete_2013 = []
    athlete_2014 = []
    athlete_2015 = []
    athlete_2016 = []

    for v in data_list:
        temp_list = v.split(',')  # split the line based on ',' delimiter
        athlete_id.append(int(temp_list[0]))
        athlete_age.append(int(temp_list[2]))
        athlete_participation.append(int(temp_list[10]))
        athlete_2003.append(int(temp_list[11]))
        athlete_2004.append(int(temp_list[12]))
        athlete_2005.append(int(temp_list[13]))
        athlete_2006.append(int(temp_list[14]))
        athlete_2007.append(int(temp_list[15]))
        athlete_2008.append(int(temp_list[16]))
        athlete_2009.append(int(temp_list[17]))
        athlete_2010.append(int(temp_list[18]))
        athlete_2011.append(int(temp_list[19]))
        athlete_2012.append(int(temp_list[20]))
        athlete_2013.append(int(temp_list[21]))
        athlete_2014.append(int(temp_list[22]))
        athlete_2015.append(int(temp_list[23]))
        athlete_2016.append(int(temp_list[24]))

        # encoding sex into integer values
        if temp_list[3] == "M":  # if athlete is male
            sex_m.append(1)
            sex_f.append(0)
            sex_u.append(0)
        elif temp_list[3] == "F":
            sex_m.append(0)
            sex_f.append(1)
            sex_u.append(0)
        else:
            sex_m.append(0)
            sex_f.append(0)
            sex_u.append(1)

    result = (athlete_id, athlete_age, sex_m, sex_f, sex_u,athlete_participation,athlete_2003,athlete_2004,athlete_2005,
              athlete_2006,athlete_2007,athlete_2008,athlete_2009,athlete_2010,athlete_2011,athlete_2012,athlete_2013,
              athlete_2014,athlete_2015,athlete_2016)
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
        x_normalized = numerator/denominator
        normalized_set.append(x_normalized)

    return normalized_set

def p_y(y):
    # this function computes Pr(Y = 1) and Pr(Y = 0)
    # n is the number of example in the given set
    n = len(y)
    summation_1 = 0
    summation_0 = 0
    for i in range(len(y)):
        if y[i] == 1:
            summation_1 += 1
        else:
            summation_0 += 1

    prob_y1 = summation_1 / n # Pr(Y = 1)
    prob_y0 = summation_0 / n # Pr(Y = 0)

    return (prob_y0, prob_y1)

def p_x_y(x,y,value_x, value_y):
    # this function computes Pr(X=value_x|y=value_y)
    summation_x = 0
    summation_y = 0
    for i in range(len(x)):
        if y[i] == value_y:
            summation_y += 1
            if x[i] == value_x:
                summation_x += 1

    prob = summation_x / summation_y
    return prob

def w_j (x,y):
    # this function returns a list of weights for one class
    x_min = np.amin(x)
    x_max = np.amax(x)

    w_j = []

    for i in range(x_min,x_max+1):
        denominator = p_x_y(x,y,i,0)
        numerator = p_x_y(x,y,i,1)
        w_j_i = np.log(numerator/denominator)
        w_j.append(w_j_i)
    return w_j


def naive_Bayes(p_y_0, p_y_1, w_j_0, w_j_1, x,y):
    # this function computes the accuracy and recall of prediction

    summation_w_0 = np.sum(w_j_0)
    m11 = 0 #example of class 1, predicted as class 1
    m01 = 0 #example of class 0, predicted as class 1
    m10 = 0 #example of class 1, predicted as class 0
    m00 = 0 #example of class 0, predicted as class 0

    difference_term = p_y_1/p_y_0
    log_term = math.log(difference_term)
    weight = np.subtract(w_j_1, w_j_0)
    decisionBd = []

    for i in range(len(x[0])):
        temp = np.dot(weight,np.array(x)[:,i])
        log_odd = log_term/10 + difference_term + temp
        #log_odd = log_term + summation_w_0 + temp
        if log_odd > 0:
            log_odd = 1
            decisionBd.append(log_odd)
        elif log_odd < 0:
            log_odd = 0
            decisionBd.append(log_odd)
        else:
            log_odd = random.choice(['0', '1'])
            decisionBd.append(log_odd)

        if y[i] == 1: # example of class 1
            if log_odd == 1:
                m11 += 1 # predicted as class 1
            else:
                m10 += 1 # predicted as class 0
        else:
            if log_odd == 0:
                m00 += 1
            else:
                m01 += 1

    accuracy = (m00 + m11)/(m00 + m11 + m01 + m10)
    accuracy_1 = m11 / (m11+m10)

    return (decisionBd,accuracy,accuracy_1)

def predict_2017(p_y_0, p_y_1, w_j_0, w_j_1, x, id):
    # this function directly outputs the prediction solution

    summation_w_0 = np.sum(w_j_0)

    difference_term = p_y_1/p_y_0
    log_term = math.log(difference_term)
    weight = np.subtract(w_j_1, w_j_0)
    decisionBd = []

    for i in range(len(x[0])):
        temp = np.dot(weight,np.array(x)[:,i])
        log_odd = log_term/10 + difference_term + temp
        #log_odd = log_term + summation_w_0 + temp
        if log_odd > 0:
            log_odd = 1
            decisionBd.append([id[i],log_odd])
        elif log_odd < 0:
            log_odd = 0
            decisionBd.append([id[i],log_odd])
        else:
            log_odd = random.choice(['0', '1'])
            decisionBd.append([id[i],log_odd])

    return decisionBd


#------------------MAIN----------------#
# extracting data
data_set = content_Reader('modified_file_3.csv')

# separating data according to the feauture name
(t_id_number, t_age, t_male, t_female, t_unknown, t_participation,t_2003,t_2004,t_2005,t_2006,
    t_2007,t_2008,t_2009,t_2010,t_2011,t_2012,t_2013,t_2014,t_2015,t_2016) = content_extractor(data_set)

'''
The following are all the tested model 

# model 1:
training_set_x = [t_2014,t_2013, t_2012,t_2011, t_female, t_male]
training_set_y = t_2015

validation_set_x = [t_2012, t_2013, t_2014,t_2015, t_female, t_male]
validation_set_y = t_2016

# model 2:
interaction1 = np.multiply(t_2010,t_2009)
training_set_x = [ t_2010 , t_2011 ,t_2012,t_2013,t_2014,interaction1]
training_set_y = t_2015

interaction1_valid = np.multiply(t_2010,t_2009)
validation_set_x = [t_2011 ,t_2012, t_2013, t_2014,t_2015, interaction1_valid]
validation_set_y = t_2016

# model 3:
interaction1 = np.multiply(t_2008,t_2009)
training_set_x = [ t_2010, t_2011 ,t_2012,t_2013,t_2014,t_female, t_male,interaction1]
training_set_y = t_2015

interaction1_valid = np.multiply(t_2012,t_2013)
validation_set_x = [t_2011 ,t_2012,t_2013, t_2014,t_2015, t_female,t_male, interaction1_valid]
validation_set_y = t_2016

# model 4:
training_set_x = [ t_2008,t_2009 ,t_2010,t_2011,t_2012, t_2013,t_2014,t_female, t_male, t_unknown]
training_set_y = t_2015

validation_set_x = [ t_2009, t_2010,t_2011, t_2012, t_2013, t_2014,t_2015, t_female,t_male, t_unknown]
validation_set_y = t_2016

# model 5:
interaction1 = np.multiply(t_2003,t_2004)
interaction2 = np.multiply(t_2005,t_2006)
interaction3 = np.multiply(t_2012,t_2014)
training_set_x = [t_2011, t_2012 ,t_2013,t_2014,t_female, t_male,interaction1,interaction2,interaction3]
training_set_y = t_2015

interaction1_v = np.multiply(t_2004,t_2005)
interaction2_v = np.multiply(t_2006,t_2007)
interaction3_v = np.multiply(t_2013,t_2015)
validation_set_x = [t_2012 ,t_2013,t_2014,t_2015, t_female,t_male,interaction1_v,interaction2_v,interaction3_v]
validation_set_y = t_2016

# model 6:
interaction1 = np.multiply(t_2013,t_2014)
interaction2 = np.multiply(t_2012,t_2014)
interaction3 = np.multiply(t_2011,t_2013)
training_set_x = [ t_2008,t_2009,t_2010,t_2011, t_2012, t_2013,t_2014,interaction1,interaction2,interaction3]
training_set_y = t_2015

interaction1_v = np.multiply(t_2013,t_2014)
interaction2_v = np.multiply(t_2012,t_2014)
interaction3_v = np.multiply(t_2011,t_2013)
validation_set_x = [t_2009,t_2010,t_2011, t_2012, t_2013,t_2014,t_2015, interaction1_v,interaction2_v,interaction3_v]
validation_set_y = t_2016

# model 7:
interaction1 = np.multiply(t_2012,t_2014)
interaction2 = np.multiply(t_2011,t_2014)
training_set_x = [ t_2007,t_2008,t_2009,t_2010,t_2011, t_2012, t_2013,t_2014,interaction1,interaction2]
training_set_y = t_2015

interaction1_v = np.multiply(t_2013,t_2015)
interaction2_v = np.multiply(t_2012,t_2015)
validation_set_x = [t_2008,t_2009,t_2010,t_2011, t_2012, t_2013,t_2014,t_2015, interaction1_v,interaction2_v]
validation_set_y = t_2016

# model 8:
interaction1 = np.multiply(t_2012,t_2014)
interaction2 = np.multiply(t_2011,t_2014)
training_set_x = [ t_2007,t_2008,t_2009,t_2010,t_2011, t_2012, t_2013,t_2014,interaction1,interaction2,t_female,t_male]
training_set_y = t_2015

interaction1_v = np.multiply(t_2013,t_2015)
interaction2_v = np.multiply(t_2012,t_2015)
validation_set_x = [t_2008,t_2009,t_2010,t_2011, t_2012, t_2013,t_2014,t_2015, interaction1_v,interaction2_v,t_female,t_male]
validation_set_y = t_2016

# model 9:
interaction1 = np.multiply(t_2012,t_2014)
interaction2 = np.multiply(t_2011,t_2014)
interaction3 = np.multiply(t_2013,t_2014)
training_set_x = [t_2005,t_2006,t_2007,t_2008,t_2009,t_2010,t_2011,t_2012,t_2013,t_2014,interaction1,interaction2, interaction3,t_female,t_male]
training_set_y = t_2015

interaction1_v = np.multiply(t_2013,t_2015)
interaction2_v = np.multiply(t_2012,t_2015)
interaction3_v = np.multiply(t_2014,t_2015)
validation_set_x =[t_2006,t_2007,t_2008,t_2009,t_2010,t_2011,t_2012,t_2013,t_2014,t_2015, interaction1_v,interaction2_v,interaction3_v,t_female,t_male]
validation_set_y = t_2016

# model 10:
interaction1 = np.multiply(t_2012,t_2014)
interaction2 = np.multiply(t_2011,t_2014)
interaction3 = np.multiply(t_2013,t_2014)
interaction4 = np.multiply(t_2010,t_2014)
training_set_x = [t_2004,t_2005,t_2006,t_2007,t_2008,t_2009,t_2010,t_2011,t_2012,t_2013,t_2014,interaction1,interaction2, interaction3, interaction4, t_female,t_male]
training_set_y = t_2015

interaction1_v = np.multiply(t_2013,t_2015)
interaction2_v = np.multiply(t_2012,t_2015)
interaction3_v = np.multiply(t_2014,t_2015)
interaction4_v = np.multiply(t_2011,t_2015)
validation_set_x =[t_2005 ,t_2006,t_2007,t_2008,t_2009,t_2010,t_2011,t_2012,t_2013,t_2014,t_2015, interaction1_v,interaction2_v,interaction3_v,interaction4_v, t_female,t_male]
validation_set_y = t_2016

# model 11:
interaction1 = np.multiply(t_2012,t_2014)
interaction2 = np.multiply(t_2011,t_2014)
interaction3 = np.multiply(t_2013,t_2014)
interaction4 = np.multiply(t_2010,t_2014)
interaction5 = np.multiply(t_2012,t_2013)
training_set_x = [t_2006,t_2007,t_2008,t_2009,t_2010,t_2011,t_2012,t_2013,t_2014,interaction1,interaction2, interaction3, interaction4,interaction5, t_female,t_male]
training_set_y = t_2015

interaction1_v = np.multiply(t_2013,t_2015)
interaction2_v = np.multiply(t_2012,t_2015)
interaction3_v = np.multiply(t_2014,t_2015)
interaction4_v = np.multiply(t_2011,t_2015)
interaction5_v = np.multiply(t_2013,t_2014)
validation_set_x =[t_2007,t_2008,t_2009,t_2010,t_2011,t_2012,t_2013,t_2014,t_2015, interaction1_v,interaction2_v,interaction3_v,interaction4_v,interaction5_v, t_female,t_male]
validation_set_y = t_2016

# model 12:
interaction1 = np.multiply(t_2012,t_2014)
interaction2 = np.multiply(t_2011,t_2014)
interaction3 = np.multiply(t_2013,t_2014)
interaction4 = np.multiply(t_2010,t_2013)
interaction5 = np.multiply(t_2012,t_2013)
interaction6 = np.multiply(t_2011,t_2013)
training_set_x = [t_2009,t_2010,t_2011,t_2012,t_2013,t_2014,interaction1,interaction2, interaction3, interaction4,interaction5, interaction6, t_female,t_male]
training_set_y = t_2015

interaction1_v = np.multiply(t_2013,t_2015)
interaction2_v = np.multiply(t_2012,t_2015)
interaction3_v = np.multiply(t_2014,t_2015)
interaction4_v = np.multiply(t_2011,t_2014)
interaction5_v = np.multiply(t_2013,t_2014)
interaction6_v = np.multiply(t_2012,t_2014)
validation_set_x =[t_2010,t_2011,t_2012,t_2013,t_2014,t_2015, interaction1_v,interaction2_v,interaction3_v,interaction4_v,interaction5_v, interaction6_v, t_female,t_male]
validation_set_y = t_2016


# model 13:
interaction1 = np.multiply(t_2012,t_2014)
interaction2 = np.multiply(t_2011,t_2014)
interaction3 = np.multiply(t_2013,t_2014)
interaction4 = np.multiply(t_2010,t_2013)
interaction5 = np.multiply(t_2012,t_2013)
interaction6 = np.multiply(t_2011,t_2013)
interaction7 = np.multiply(interaction3,t_2012)
training_set_x = [t_2009,t_2010,t_2011,t_2012,t_2013,t_2014,interaction1,interaction2, interaction3, interaction4,interaction5, interaction6, interaction7]
training_set_y = t_2015

interaction1_v = np.multiply(t_2013,t_2015)
interaction2_v = np.multiply(t_2012,t_2015)
interaction3_v = np.multiply(t_2014,t_2015)
interaction4_v = np.multiply(t_2011,t_2014)
interaction5_v = np.multiply(t_2013,t_2014)
interaction6_v = np.multiply(t_2012,t_2014)
interaction7_v = np.multiply(interaction3_v,t_2013)
validation_set_x =[t_2010,t_2011,t_2012,t_2013,t_2014,t_2015, interaction1_v,interaction2_v,interaction3_v,interaction4_v,interaction5_v, interaction6_v,interaction7_v]
validation_set_y = t_2016

# model 14:

training_set_x = [t_2008,t_2009,t_2010,t_2011,t_2012,t_2013,t_2014]
training_set_y = t_2015

validation_set_x =[t_2009,t_2010,t_2011,t_2012,t_2013,t_2014,t_2015]
validation_set_y = t_2016

# model 15:
interaction1 = np.multiply(t_2012,t_2014)
interaction2 = np.multiply(t_2011,t_2014)
interaction3 = np.multiply(t_2013,t_2014)
interaction4 = np.multiply(t_2010,t_2013)
interaction5 = np.multiply(t_2012,t_2013)
interaction6 = np.multiply(t_2011,t_2013)
interaction7 = np.multiply(t_female,t_2014)
training_set_x = [t_2009,t_2010,t_2011,t_2012,t_2013,t_2014,interaction1,interaction2, interaction3, interaction4,interaction5, interaction6, interaction7]
training_set_y = t_2015

interaction1_v = np.multiply(t_2013,t_2015)
interaction2_v = np.multiply(t_2012,t_2015)
interaction3_v = np.multiply(t_2014,t_2015)
interaction4_v = np.multiply(t_2011,t_2014)
interaction5_v = np.multiply(t_2013,t_2014)
interaction6_v = np.multiply(t_2012,t_2014)
interaction7_v = np.multiply(t_female,t_2015)
validation_set_x =[t_2010,t_2011,t_2012,t_2013,t_2014,t_2015, interaction1_v,interaction2_v,interaction3_v,interaction4_v,interaction5_v, interaction6_v,interaction7_v]
validation_set_y = t_2016

# model 16:
interaction1 = np.multiply(t_2012,t_2014)
interaction2 = np.multiply(t_2011,t_2014)
interaction3 = np.multiply(t_2013,t_2014)
interaction4 = np.multiply(t_2010,t_2013)
interaction5 = np.multiply(t_2012,t_2013)
interaction6 = np.multiply(t_2011,t_2013)
interaction7 = np.multiply(t_2011,t_2012)
training_set_x = [t_2008,t_2009,t_2010,t_2011,t_2012,t_2013,t_2014,interaction1,interaction2, interaction3, interaction4,interaction5, interaction6, interaction7]
training_set_y = t_2015

#model 17:
interaction1 = np.multiply(t_2012,t_2014)
interaction2 = np.multiply(t_2011,t_2014)
interaction3 = np.multiply(t_2013,t_2014)
interaction4 = np.multiply(t_2010,t_2013)
interaction5 = np.multiply(t_2012,t_2013)
interaction6 = np.multiply(t_2011,t_2013)
training_set_x = [t_2009,t_2010,t_2011,t_2012,t_2013,t_2014,interaction1,interaction2, interaction3, interaction4,interaction5, interaction6]
training_set_y = t_2015

interaction1_v = np.multiply(t_2013,t_2015)
interaction2_v = np.multiply(t_2012,t_2015)
interaction3_v = np.multiply(t_2014,t_2015)
interaction4_v = np.multiply(t_2011,t_2014)
interaction5_v = np.multiply(t_2013,t_2014)
interaction6_v = np.multiply(t_2012,t_2014)
validation_set_x =[t_2010,t_2011,t_2012,t_2013,t_2014,t_2015, interaction1_v,interaction2_v,interaction3_v,interaction4_v,interaction5_v, interaction6_v]
validation_set_y = t_2016

#----------------------------------------------------------------------------#
#The following part are used to compute training error and validation error

(p_y_0,p_y_1) = p_y(training_set_y)
weights = []
for i in range(len(training_set_x)):
    temp = w_j(training_set_x[i],training_set_y)
    weights.append(temp)

print("running")

(decision, accuracy,accuracy1) = naive_Bayes(p_y_0,p_y_1,np.array(weights)[:,0],np.array(weights)[:,1],training_set_x, training_set_y)

print("accuracy is: ")
print(accuracy)
print("accuracy1 is: ")
print(accuracy1)

print("still running...")

(decision_test, accuracy_test, accuracy1_test) = naive_Bayes(p_y_0,p_y_1,np.array(weights)[:,0],np.array(weights)[:,1],validation_set_x, validation_set_y)
print("accuracy_test is: ")
print(accuracy_test)
print("accuracy1_test is: ")
print(accuracy1_test)
print(decision_test)

'''

# The following code is to compute the final prediction result
# final model () 17 <-- the chosen best one

# features and interaction terms for training set
interaction1 = np.multiply(t_2013,t_2015)
interaction2 = np.multiply(t_2012,t_2015)
interaction3 = np.multiply(t_2014,t_2015)
interaction4 = np.multiply(t_2011,t_2014)
interaction5 = np.multiply(t_2013,t_2014)
interaction6 = np.multiply(t_2012,t_2014)
training_set_x = [t_2010,t_2011,t_2012,t_2013,t_2014,t_2015,interaction1,interaction2, interaction3, interaction4,interaction5, interaction6]
training_set_y = t_2016

# features and interaction terms for test set
interaction1_v = np.multiply(t_2014,t_2016)
interaction2_v = np.multiply(t_2013,t_2016)
interaction3_v = np.multiply(t_2015,t_2016)
interaction4_v = np.multiply(t_2012,t_2015)
interaction5_v = np.multiply(t_2014,t_2015)
interaction6_v = np.multiply(t_2013,t_2015)
test_set_x =[t_2011,t_2012,t_2013,t_2014,t_2015,t_2016, interaction1_v,interaction2_v,interaction3_v,interaction4_v,interaction5_v, interaction6_v]

# compute the probability of Y = 0 and Y = 1
(p_y_0,p_y_1) = p_y(training_set_y)

# compute the weights for each Xj_0 and Xj_1
weights = []
for i in range(len(training_set_x)):
    temp = w_j(training_set_x[i],training_set_y)
    weights.append(temp)

print("running")
# using the training weights to predict 2017 result
predicted_2017 = predict_2017(p_y_0,p_y_1,np.array(weights)[:,0],np.array(weights)[:,1],test_set_x,t_id_number)
#print(predicted_2017)

#creating csv file which contain the final prediction
myfile = open('final_result.csv', 'w', newline='')
wr = csv.writer(myfile)
for i in range(len(predicted_2017)):
    wr.writerow(predicted_2017[i])
myfile.close()
