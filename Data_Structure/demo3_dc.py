# 字典创建
# 第一种方式
# person1 = {
#     'name': '张三',
#     'age': 24,
#     'city': '北京',
#     'phone': '15600000'
# }
#
# print(person1)
# # 第二种方式
# person2 = dict(name='张三', age=24)
#
# # 创建空字典
# person3 = dict()
# person4 = {}
# print(person3)
# print(type(person3))
# print(person4)
# print(type(person4))


# 字典相关操作
# person1 = {
#     'name': '张三',
#     'age': 24,
#     'city': '北京',
#     'phone': '15600000'
# }
# 访问字典里面的某一个元素
# print(person1['name'])
# print(person1['city'])
# print(person1['gender'])  # 如果传入一个原字典内不存在的 key 则会报错
# print(person1.get('phone'))
# print(person1.get('gender'))  # get方法 如果访问的 key 不在原字典里面, 如果没有填default参数则返回 None , 有则返回default值
# print(person1.get('age','30'))

# 如何 增加 字典元素 和 修改 字典元素
# 增加一个元素, 一个元素  -- >  一个 键值对
# person1['gender'] = '男'
# print(person1)
# print(person1['gender'])

# 修改一个元素
# person1['name'] = '李四'
# print(person1)
# print(person1['name'])

# 删除字典里面的元素   删除 -- > 删除一组键值对
# del person1['age']
# print(person1)
# value = person1.pop('gender','男')
# print(value)
# print(person1)
# 清空字典
# person1.clear()
# print(person1)

# 查看所有的 键 和 值 以及 键值对一起查看
# print(person1.keys())
# print(person1.values())
# print(person1.items())

# 字典的遍历
# person1 = {
#     'name': '张三',
#     'age': 24,
#     'city': '北京',
#     'phone': '15600000'
# }

# 遍历键
# for key in person1.keys():
#     print(key)

# 遍历值
# for value in person1.values():
#     print(value)

# 遍历 键值对
# for key, value in person1.items():
#     print(key, value)

# 字典常用方法
person1 = {
    'name': '张三',
    'age': 24,
    'city': '北京',
    'phone': '15600000'
}

# # 我们想检查某一个键是否存在于我们的字典里面
# # 第一种方法, 用 get
# print(person1.get('gender', '没找到'))
# # 第二种方法, 用 in
# print('gender' in person1)
#
# # 合并字典
# gender = {
#     'gender' : '男'
# }
#
# person1.update(gender)   # 合并 update
# print(person1)

# 怎么获取一个字典的长度   # 字典的长度 : 键值对的个数
print(len(person1))

# 创建新字典
keys = ['name', 'age', 'city', 'phone', ]
values = ['张三', '24', '北京', '15600000',]
dic = dict(zip(keys, values))
print(dic)
