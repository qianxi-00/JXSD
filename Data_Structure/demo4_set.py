# 集合的两个特性 : 1. 无序   2. 不重复
# # 集合的创建
# set1 = {1, 2, 3, 4, 5, 'a', 'b', 'c', 1, 1, 1}
# # 创建一个空集合
# set2 = set()
# # 第三种方法
# set3 = set((1,2,3,4,5))
# print(set3)
# print(type(set3))
# print(set2)
# print(type(set2))
# print(set1)
# print(type(set1))


# 集合操作
# a = {1, 2, 3, 4, 5, 6}
# b = {4, 5, 6, 7, 8, 9}
# print(a)
# print(b)

# # 集合的并集
# print(a | b)
# print(a.union(b))

# # 集合的交集
# print(a & b)
# print(a.intersection(b))

# # 集合的差集
# print(a - b)
# print(a.difference(b))
# print(b - a)
# print(b.difference(a))

# # 集合的对称差集
# print(a ^ b)
# print(a.symmetric_difference(b))

# 集合的常用方法
# fruits = {"苹果", "香蕉", "桃子", "西瓜"}
# print(fruits)

# 增加和删除集合里面的元素
# 增加
# fruits.add("橙子")
# print(fruits)
# fruits.add("苹果")
# print(fruits)

# 删除
# fruits.remove("橙子")  # 如果移除的元素不在我们的集合里面, 则会报错
# fruits.discard("橙子")  # 如果移除的元素不在集合里面, 则不会报错
# print(fruits)

# 其他的方法
# 1. 统计我们集合里面的元素个数
# print(len(fruits))
# print("橙子" in fruits)

# 2. 清空我们的集合
# fruits.clear()
# print(fruits)
# print(type(fruits))