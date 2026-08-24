# 循环语句
# 1. 遍历列表
# ls = ["a", "b", "c"]
# for char in ls :
#     print(char)
#
# # 2. 遍历range
# for i in range(5):
#     print(i)

# 3. 遍历字典
# person = {
#     'name' : '张三',
#     'age' : 22,
#     'gender' : 'male'
# }
# for key,value in person.items():
#     print(f'{key},{value}')

# while 循环
# count = 0
# while count < 10:
#     print(count)
#     count += 1
#
# print('循环1结束')
#
# count = 0
# while count < 10:
#     count += 1
#     print(count)

# while True :
#     value = input("请输入 q 退出循环!\n")
#     if value == "q" :
#         break


# 控制循环
# 1. break 用于结束一个循环
# while 循环中
# while True :
#     value = input("请输入 q 退出循环!\n")
#     if value == "q" :
#         break

# for 循环中
# for i in range(10):
#     if i == 5 :
#         break
#     print(i)


# 2. continue  # 结束本轮循环, 在循环体内 continue 往后的代码都不会执行 直接跳转到下次循环
# i = 0
# # 输出所有小于 10 的奇数
# while i < 10 :
#     i += 1
#     if i % 2 == 0 :
#         continue
#     print(i)
#
# print("hello")

# for 循环中
# for i in range(10):
#     if i % 2 ==0 :
#         continue
#     print(i)

# for i in range(5):
#     print(i)
#
# else:
#     print("循环结束")

# break 时 else 下面的内容是不会执行的, 但是不会影响非else下的代码内容
# for i in range(5):
#     if i == 3:
#         break
#     print(i)
# else :
#     print("循环结束")
#
# print("hello")



