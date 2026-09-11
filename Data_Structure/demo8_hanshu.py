# 函数
# # 定义一个函数
# def greet():
#     print("hello")
#
# # 调用这个函数
# greet()

# # 如何定义一个带参数的函数呢
# def greet_person(name):
#     print(f"hello, {name}")
#
# # 如何调用一个带参数的函数
# greet_person("张三")
# greet_person("李四")

# 如何定义带返回值的函数
# def add(a,b):
#     return a + b
#
# # 如何调用 / 如何接收这个返回值
# result = add(5,10)
# print(result)

# 函数参数类型
# #１.位置参数  按照位置顺序的去给我们的函数参数去赋值
# def func1(a, b, c):
#     print(f"a的值:{a}, b的值:{b}, c的值:{c}")


#
# func1(3,2,1)
#
# # 2. 默认参数
# def func2(a,b=10,c=10):
#     print(f"a的值:{a}, b的值:{b}, c的值:{c}")
# func2(3)

# 3. 关键字参数  # 如果使用了关键字参数, 最好都用关键字参数传参, 关键字传参可以单独的针对最后一个要传的参数
# 如果在定义函数的时候, 出现了带默认值的参数, 那么后面的所有参数定义都要带默认值, 不然就定义函数报错
# def func1(a, b, c = 10, d = 100):
#     print(f"a的值:{a}, b的值:{b}, c的值: , d的值:{d}")
# func1(10,5, d = 100)


# #４.　可变参数　*args  # 可变，指的是参数的数量可变
# def sum_all(*args):
#     return sum(args)
#
# print(sum_all(1,2,3.5,4,5.5))
#
#
# # 5. 可变关键字参数 **kwargs  # 可变, 指的是 参数的数量可变  另外 变量名也可以变
# def print_info(**kwargs):
#     for key, value in kwargs.items():
#         print(f"{key}, {value}")
#
# print_info(name = "张三", age = 24, gender = "male")


# # 函数文档字符串
# def calculate_area(radius):
#     """
#     这个方法用于计算一个给定半径的圆的面积
#     :param radius (float) :  圆的半径
#     :return: area (float) :  圆的面积
#     """
#     import math
#     area = math.pi * (radius ** 2)
#     return area
#
# # 查看这个函数的文档字符串
# print(calculate_area.__doc__)
# print(calculate_area(5))

# 匿名函数  lambda 表达式  应用范围, 一般用于比较简单的, 不需要大量复用的代码块
# 基本语法
# add = lambda x, y: x + y
# print(add(2, 3))

# 配合 map 使用
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x : x ** 2, numbers))  # map(arg1,arg2)  arg1 往往是一个(匿名)函数, 或者一种运算, arg2 传前面这个函数或者运算的执行对象
print(numbers)
print(squared)

# 配合 filter 使用
even = list(filter(lambda x : x % 2 == 0, numbers))
print(even)


