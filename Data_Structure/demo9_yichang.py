# 异常处理
# 处理代码中可能出错的代码块

# 1. 基本的 try-except 使用
# try:
#     # try 下面 编写可能会出错的代码
#     result = 10 / 0  # 肯定会报错, 因为 0 不能作为除数
# except ZeroDivisionError :
#     # 针对不同的报错类型, 去输出不同的日志信息
#     print(" 0 不能作为除数 !!")
# except :
#     # 处理其他所有的异常情况
#     print("程序发生了错误")

# 多个异常的情况
# try:
#     num = int(input("请输入一个数字 :  "))
#     result = 10 / num
#     print(f"输出结果: {result}")
#
# except ZeroDivisionError :
#     print(" 0 不能作为除数 !!")
#
# except ValueError :
#     print("请输入一个有效的数字 !!")
#
# except Exception as e :
#     print(f"未知错误类型: {e}")

# try - except - else - finally
# try:
#     f = open('file.txt', 'r')
#     content = f.read()
# except FileNotFoundError:
#     print("文件不存在, 没找到该文件")
# except Exception as e:
#     print(f"错误 : {e}")
# else:
#     # 没有发生异常的时候会执行
#     print("文件读取成功")
# finally:
#     # 无论是否有异常, 都会执行finally 下面的代码内容
#     print("操作完毕")
#     print(content)
#     if 'f' in locals():
#         f.close()


# 自定义异常
# def validate_age(age):
#     if age < 0 :
#         raise ValueError('年龄不能为负数, 必须大于等于0')
#     if age > 150 :
#         raise ValueError("正常人的年龄不可能超过150")
#     return True
#
# try :
#     validate_age(45)
# except ValueError as e :
#     print(e)
# else :
#     print("没有异常")







