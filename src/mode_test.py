import statistics


def find_max_mode(list1):
    list_table = statistics._counts(list1)
    len_table = len(list_table)

    if len_table == 1:
        max_mode = statistics.mode(list1)
    else:
        new_list = []
        for i in range(len_table):
            new_list.append(list_table[i][0])
        max_mode = max(new_list)  # use the max value here
    return max_mode


if __name__ == '__main__':
    a = [1, 1, 2, 2, 3]
    print(find_max_mode(a))  # print 2
