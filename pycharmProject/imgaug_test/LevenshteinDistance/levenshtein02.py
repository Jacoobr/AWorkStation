#coding=utf-8
from __future__ import print_function


def levenshtein(str1, str2):
    matrix = [[i + j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]
    #xrange() should be better than range()
    for i in xrange(1, len(str1)+1):
        for j in xrange(1, len(str2)+1):
            if str1[i-1] == str2[j-1]:
                d = 0
            else:
                d = 1
            matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + d)
    print('the value of matrix is: ')
    for i in range(len(str1)):
        for j in range(len(str2)):
            print(matrix[i][j], end='')  #cut the '\n' charactor
            print(' ', end='')

        print('\n')
    print('the distance of "'+ str1 + '" and "' +str2+'" is: ')
    return matrix[len(str1)][len(str2)]

str1 = raw_input('please input the first string: ')
str2 = raw_input('please input the first string: ')
result = levenshtein(str1, str2)
print(result)