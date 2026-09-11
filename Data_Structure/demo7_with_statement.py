# 文件操作
# 打开读取一个文本文件
f = open("F://ProGram//Python_Base//Data_Structure//file.txt", 'a+',encoding='utf-8')
f.write("Hello World2222")
f.close()

# 如果用 with 语句  # 自动的帮我们处理资源的关闭
# 追加写文件
with open("F://ProGram//Python_Base//Data_Structure//file.txt", 'a+',encoding='utf-8') as f:
    f.write("\nHello World3333")

# 读文件
# with open("F://ProGram//Python_Base//Data_Structure//file.txt", 'r',encoding='utf-8') as f:
#     print(f.read())

# 文件的常用操作
# 如何一行一行的去读呢
with open("F://ProGram//Python_Base//Data_Structure//file.txt", 'r', encoding='utf-8') as f:
    for line in f.readlines():
        print(line.strip())

# 多个上下文
# 我们可以同时打开多个文件
with open("F://ProGram//Python_Base//Data_Structure//file.txt", 'r', encoding='utf-8') as f1 ,open("F://ProGram//Python_Base//Data_Structure//file2.txt", 'w', encoding='utf-8') as f2:
    content = f1.read()
    f2.write(content)

