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
def func1(a, b, c = 10, d = 100):
    print(f"a的值:{a}, b的值:{b}, c的值: , d的值:{d}")
func1(10,5, d = 100)
