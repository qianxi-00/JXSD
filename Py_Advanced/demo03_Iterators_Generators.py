# 自定义迭代器
# class Countdown:
#     def __init__(self, start):
#         self.start = start
#
#     def __iter__(self):
#         return self
#
#     def __next__(self):
#         if self.start > 0 :
#             self.start -= 1
#             return self.start + 1
#         else :
#             raise StopIteration
#
# for num in Countdown(5):
#     print(num)

# 生成器
# 首先 : 生成器 是属于 我们的迭代器的
# yield 方法
# def countdown(n):
#     print("开始倒计时!")
#     while n > 0 :
#         yield n
#         n -= 1
#     print("倒计时完成!")
#
# for i in countdown(5):
#     print(i)


# 开始倒计时
# 5
# 4
# 3
# 2
# 1
# 倒计时完成





# 生成器表达式
gen = ((x + 2)for x in range(5))
print(next(gen))
print(next(gen))
print(next(gen))
print(next(gen))
print(next(gen))
print(next(gen))
print(next(gen))