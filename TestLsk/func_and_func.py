# def func1():    
#     print("func1")
#     def func2():
#         print("func2")
# func1()
# print("func2 in func1")
# func1().apply(func2)#apply 是 PyTorch 中 nn.Module 类的方法（用于递归应用函数到所有子模块），普通 Python 函数没有这个方法，不能直接调用。
# 调用func1，获取返回的func2

def func1():
    print("func1")
    
    # 定义内部函数func2（带参数）
    def func2(m):
        print(f"func2 处理: {m}")
    
    # 返回内部函数，使其可以被外部访问
    return func2

# 调用func1，获取返回的func2
inner_func2 = func1()

# 模拟遍历调用（类似PyTorch的apply逻辑）
layers = ["layer1", "layer2", "layer3"]
for layer in layers:
    inner_func2(layer)  # 手动对每个"层"调用func2
