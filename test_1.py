
# coding: utf-8

# 螺旋数组求任意数到1的曼哈顿距离


def getSteps(num):
    if num == 1:
        return 0
    elif num < 1:
        print('{} is not an positive integer'.format(num))
        return None
    else:
        i = 1

        while (2*i - 1) **2 < num:
            i += 1

        i -= 1

        length = 2*i - 1

        min_one = length**2

        each_side = length + 1

        loss = num - min_one

        a = loss//each_side
        b = loss%each_side

        idx_1 = [int((length + 1)/2),int((length + 1)/2)]

        if a == 0:
            idx_input = [each_side - b,length + 1]
        elif a == 1:
            idx_input = [0,length + 1 - b]
        elif a == 2:
            idx_input = [b,0]
        elif a == 3:
            idx_input = [length + 1,b]
        else:
            idx_input = [length + 1, length + 1]

        steps = abs(idx_1[0] - idx_input[0]) + abs(idx_1[1] - idx_input[1])
        
        return steps

