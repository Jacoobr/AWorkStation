#coding=utf-8
def leven(str1, str2):
    len_str1 = len(str1)
    len_str2 = len(str2)
    #construct  matrix (len_str1)*(len_str1)
    #matrix = []
    matrix = [0 for n in range(len_str1 * len_str2)]
    len_matrix = len(matrix)
    # x axis
    for x_axis in range(len_str1):
        matrix = x_axis
    #y axis
    # the range should be len_str1 to len(matrix)
    for y_axis in range(0, len_matrix, len_str1):
        if y_axis % len_str1 == 0:
            matrix[y_axis] = y_axis // len_str1

    for x_axis in range(1, len_str1):
        for y_axis in range(1, len_str2):
            if str1[x_axis-1] == str2[y_axis-1]:
                cost = 0
            else:
                const = 1
            matrix[y_axis*len_str1+x_axis] = min(matrix[(y_axis-1)*len_str1+x_axis]+1,
                                                 matrix[y_axis*len_str1+(x_axis-1)]+1,
                                                 matrix[(j-1)*len_str1+(x_axis-1)]+const)
    return matrix[-1]


if __name__ == '__main__':
    leven("beauty","batyu")