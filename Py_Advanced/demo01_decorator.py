# 基本装饰器
# 装饰器的作用 : 是包装一个函数, 给这个函数增强一些相关逻辑或能力

# # 定义装饰器
# def my_decorator(func):
#     def wrapper():
#         print("原函数调用前")
#         func()
#         print("原函数调用后")
#     return wrapper
#
# # # 定义一个原函数
# # def say_hello():
# #     print("Hello!")
#
# # 使用我们的装饰器
# @my_decorator
# def say_hello():
#     print("Hello!")
#
#
# say_hello()  # 1
#
# say_hello = my_decorator(say_hello)  # 2


# # 带参数的装饰器
# # 定义
# def repeat(times):
#     def decorator(func):
#         def wrapper(*args, **kwargs):
#             for _ in range(times):
#                 result = func(*args, **kwargs)
#             return result
#         return wrapper
#     return decorator
#
#
#
# # 原函数定义出来
# @repeat(3)
# def greet(name):
#     print(f"Hello, {name}!")
#
# # 调用
# greet("张三")


# 类装饰器
class CountCalls:
    def __init__(self, func):
        self.func = func
        self.count = 0

    def __call__(self, *args, **kwargs):
        self.count += 1
        print(f"{self.func.__name__} 被调用了 {self.count} 次")
        return self.func(*args, **kwargs)

@CountCalls
def greet(name):
    print(f"Hello! {name}")

@CountCalls
def greet2():
    print("Hello! World!")

# 这里在调用的时候, 先初始化了一个类装饰器的实例, 然后当我们的函数被调用的时候, 就会触发我们的__call__这个魔法方法, 相当于可以在这个魔法方法里面对我们的函数进行功能的增强
greet("张飞")
greet("刘备")
greet("张飞")
greet2()
greet("刘备")
greet2()
