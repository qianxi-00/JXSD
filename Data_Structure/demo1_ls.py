# 创建列表
# 1. 第一种方法
# fruits = ["苹果", "桃子", "西瓜"]
# number = [1, 2, 3]
# mixed = [1, 2.5, "西瓜", True]
# empty = []
# ls = list()
#
# print(fruits)
# print(number)
# print(mixed)
# print(empty)
# print(ls)


# 2. 我们可以使用 range 创建
# nums = list(range(5))
# print(nums)

# 3. 用列表推导式来创建一个列表
# ls2 = [(x+1) for x in range(5)]  # [0, 1, 2, 3, 4] ---> [0, 1, 4, 9, 16]
#
# print(ls2)

# 列表的索引和切片
# 索引操作
# fruits = ['苹果', '香蕉', '橙子', '葡萄', '西瓜']
# # 取橙子这个字符串
# print(fruits[2])
# print(fruits[0])
# print(fruits[-1])
#
# # 切片操作
# ls3 = fruits[:3]
# ls4 = fruits[2:]
# ls5 = fruits[::-1]
#
# print(ls3)
# print(ls4)
# print(ls5)

# 列表的常用方法
# fruits = ['苹果', '香蕉', '橙子', '葡萄', '西瓜', '香蕉']

# 列表的增加元素
# fruits.append("桃子")  # 在列表的末尾增加一个指定元素
# fruits.insert(-1, "桃子")  # 在列表指定的索引位置增加一个元素
# fruits.extend(["桃子", "番茄"])  # 在列表末尾增加多个元素
# print(fruits)

# 列表的删除元素
# fruits.remove('香蕉')  # 删除第一次出现的指定元素
# fruits.pop(3)   # 删除指定位置索引地方的元素
# del fruits[-1]  # 删除指定元素
# print(fruits)

# 列表的修改元素
# fruits[0] = "番茄"
# print(fruits)

# 列表的查找
# print(len(fruits))  # len()  查看我们列表中的元素个数
# print(fruits.count('香蕉'))  # count() 查找指定元素在原列表中出现的次数
# print(fruits.index('葡萄'))  # index() 返回指定元素在列表中出现的第一次的位置索引

# 列表的排序
# number = [2, 15, 30, 17, 44, 8, 12]
# print(number)
# number.sort() # [2, 8, 12, 15, 17, 30, 44]  # 默认从小到大进行排序
# print(number)
# number.sort(reverse=True)  # 从大到小进行排序
# num = number[::-1]
# print(number)
# print(num)
# num = sorted(number,reverse=True)  # sorted() 返回一个排完序的新列表
# print(num)

# 列表的遍历
# 遍历 : 从我们列表中, 一个个的取出元素
# fruits = ['苹果', '香蕉', '橙子', '葡萄', '西瓜', '香蕉']

# 借助循环来实现遍历
# for i in fruits:
#     print(i)

# 利用索引取遍历列表
# for i in range(len(fruits)):
#     print(fruits[i])

# 一次性拿到元素和对应的索引
# for index,iteam in enumerate(fruits):
#     print(f"第{index}位置的元素是: {iteam}")


# 列表的复制
# ls1 = [1, 2, 3]
# ls2 = ls1.copy()
# ls2 = ls1[::]
# print(ls1)
# print(ls2)

# 浅拷贝和 深拷贝
# 浅拷贝
# ls1 = [1, 2, 3]
# ls2 = ls1.copy()
# # ls1[0] = 4
# ls2[1] = 5
# print(ls1)
# print(ls2)

# 浅拷贝 它 只 复制我们外层的内容
# ls3 = [[1,2],[3,4]]
# ls4 = ls3.copy()
# print(ls3)
# print(ls4)
# ls3[0]= [5,6]
# print(ls3)
# print(ls4)
#
# print("---------------")

# 深拷贝
# import copy
# ls5 = [[1,2],[3,4]]
# ls6 = copy.deepcopy(ls5)
# print(ls5)
# print(ls6)
# ls6[0][0] = 5
# print(ls5)
# print(ls6)




# 高维列表的索引
# ls3 = [[[1,2],[3,4],[5,6]],[[1,2],[3,4],[5,6]]]
# print(ls3[1][1][1])





