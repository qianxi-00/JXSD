# 条件语句

# 1. if 语句

# 判断一个人的成绩优劣
# score = 45
# # /t   按一下 tab  四个空格
# if score >= 90 :
# #     print("优秀")
# # elif score >= 80 :
# #     print("良好")
# # elif score >= 70 :
# #     print("中等")
# # elif score >= 60 :
# #     print("及格")
# else :
#     print("不及格")

# 嵌套多层 if 语句
# x = -10
# y = 20
#
# if x > 0:
#     if y > 0:
#         print("x和y都大于0")
#
# if x > 0 and y > 0:
#     print("x和y都大于0")

# 2. 条件语句 match case

# status = 503
#
#
# match status :
#     case 200 :
#         print("请求成功")
#     case 404 :
#         print("未找到")
#     case 500 :
#         print("服务器错误")
#     case _:
#         print("未知错误")

# code = 402
# match code :
#     case 400 | 401 | 403 :
#         print("客户端错误")
#     case 500 | 502 | 503 :
#         print("服务端错误")
#     case _ :
#         print(code)

# 用 match case 实现条件匹配
# score = 68
# match score:
#     case 100 | 99 | 98 | 97 | 96 | 95 | 94 | 93 | 92 | 91 | 90:
#         print("优秀")
#     case 89 | 88 | 87 | 86 | 85 | 84 | 83 | 82 | 81 | 80:
#         print("良好")

# match case 还可以用来遍历我们的字典
# person1 = {
#     'name': '张三',
#     'age': 24,
#     'city': '北京',
#     'phone': '15600000'
# }
# print(person1.keys())
# print(type(person1.keys()))
# print(person1)
# match person1:
#     case {'name' : n , 'age' : a , 'city' : c, 'phone' : p} :
#         print(f"姓名: {n}, 年龄: {a}, 城市: {c} , 电话{p}")

# all 和 any
# all  全部
# numbers = [1,2,3,4,5]
# print(all(i > 3 for i in numbers))
# print(all([True, False, True]))
# print(all([True, True, True]))
#
# # any 任一
# print(any(i > 1 for i in numbers))
# print(any([True, True, True]))
# print(any([False, False, True]))
# print(any([False, False, False]))


# all 和 any 的常见使用场景
# 1. 我们可以用他们来判断一个密码是否符合长度要求
passwords = ['1234qwe', '1qaz2wsx', '6900x', '1']  # 我们要求一个密码必须长度大于等于 6
print(all(len(p) >= 6 for p in passwords))

# 2. 检查某用户里面是否有管理员
users = [{'name': 'A', 'role': 'user'}, {'name': 'A', 'role': 'user'}, {'name': 'A', 'role': 'user'}]
print(any(u["role"] == 'admin' for u in users))

# 3. 我们还可以用来检查一个列表里面是否包含有效数据
ls = ["data1", "data2", "data3", ""]
print(any(ls))  # true 这个ls 里面有 有效数据
ls2 = ["","",""]
print(any(ls2))  # 无有效数据
