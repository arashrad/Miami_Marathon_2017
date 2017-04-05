def index_finder(data_set, year):
    """it takes data_set and year as inputs and returns an index list of all corresponding records for given year"""
    result = []
    i = 0
    for i, v in enumerate(data_set):
        temp_list = v.split(',')
        if int(temp_list[7]) == year:
            result.append(i)
    return result


def time_average(data_set, year):
    """gives the average match time for given year"""
    result = 0
    index_list = index_finder(data_set, year)
    for v in index_list:
        temp_list = data_set[v].split(',')
        result += int(temp_list[8])
    return result/len(index_list)


def time_modifier_factor(data_set):
    total_time = 0
    for year in range(2003, 2017, 1):
        total_time += time_average(data_set, year)
    total_time -= time_average(data_set, 2013)
    total_average = total_time / 13
    factor = total_average / time_average(data_set, 2013)
    return factor


def match_generator(t_list, v_list):
    validation_id = []
    training_id = []
    new_validation = []
    for v in t_list:
        temp_list = v.split(',')
        training_id.append(temp_list[0])
    for v in v_list:
        temp_list = v.split(',')
        validation_id.append(temp_list[0])
    for (i, v) in enumerate(validation_id):
        if v in training_id:
            new_validation.append(v_list[i])
    return new_validation









