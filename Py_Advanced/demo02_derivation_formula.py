# 推导式 : 是一种简洁的创建列表、字典、集(一种迭代器)合的方法

# 1. 列表推导式

# 基本语法
# # 列表变量 = [ 表达式 for 变量 in 迭代器 条件语句]
# squares = [(x + 2) for x in range(10)]
# print(squares)  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# 带条件语句的列表推导式
# evens = [(x + 1) for x in range(20) if x % 2 != 0]
# print(evens)



# 生成器的推导式表达式
# 生成器变量 = (表达式 for 变量 in 迭代器 条件语句)
gen = (x**2 for x in range(10))   # 圆括号! 不是我们的元组推导式



# # 字典推导式
# # 字典 : key : value 对 键值对
# # 字典变量 = {x : y for x , y in 迭代器里面 条件语句}
# # 举一个只用一个变量完成我们字典推导式的例子
# squares = { x: x + 2 for x in range(10) if x % 2 == 0}
# print(squares)
#
# # 如果用两个变量
# dic = {x+2:y-2 for x , y in squares.items()}
# print(dic)
#
# # 合并两个列表, 使其成为一个字典
# keys = ["name", "age", "gender"]
# values = ["张三", " 24", "男"]
# person = {k : v for k , v in zip(keys, values)}
# print(person)

# 集合推导式
# 集合变量 = {表达式 for 变量 in 迭代器 if条件语句}
# squares = {x**2 for x in range(10)}
# print(type(squares))
# print(squares)

# 集合用来去重
# words = ["apple", "banana", "apple", "orange", "blue"]
# unique = {word[0] for word in words}
# print(unique)



