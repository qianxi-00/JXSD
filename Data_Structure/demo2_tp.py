# 1. 元组的创建
# color = ('红', '绿', '蓝')
# number = (1,2,3)
# mixed = (1,2,2.5,'白',True)
# empty = ()
tp = tuple()
# print(color)
# print(type(color))
# print(type(number))
# print(number)
# print(mixed)
# print(type(mixed))
# print(type(empty))
# print(type(tp))

# # 注意一点, 如果我们要创建的元组里面 只存放一个元素, 那么我们要在这个元素后面加一个 ,
# single = (2.5,)
# print(single)
# print(type(single))
#
# # 特殊的, 其实不加 () 括起来 也是可以创建一个元组的
# t = 1, 2, 3
# print(t)
# print(type(t))

# 元组的操作

# 索引
# color = ('红', '绿', '蓝', '黄', '白', '黑')
# print(color[2])
# print(color[-1])
# print(color[len(color)-1])
# tp = (1,2,(3,4,(5,6)))
# print(tp[2][1])

# # 切片
# color = ('红', '绿', '蓝', '黄', '白', '黑')
# print(color[0:3])
# print(color[:6])
#
# tp = (1, 2, (3, 4, (5, 6)))
# print(tp[1:])
# print((tp[1], (tp[2][0],tp[2][1],(tp[2][2][0],))))

# 常用方法
# color = ('红', '绿', '蓝', '黄', '白', '黑', '蓝')
# print(len(color))
# print(color.count('蓝'))
# print(color.index('蓝'))

# 元组解包
# a,b,c= (1,2,3)
# print(a,b,c)

# 交换变量
# x, y = 1, 2
# x, y = y, x
# print(x, y)
