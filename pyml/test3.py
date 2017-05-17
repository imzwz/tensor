# _*_ coding:utf-8 _*_
def knapsack(t, w):
    """
    :param t: 背包总容量
    :param w: 物品重量列表
    :return:
    """
    n = len(w)  # 可选的物品数量
    stack = []  # 创建一个栈
    k = 0  # 当前所选择的物品游标
    while stack or k < n:  # 栈不为空或者k<n
        while t > 0 and k < n:  # 还有剩余空间并且有物品可装
            if t >= w[k]:  # 剩余空间大于等于当前物品重量
                stack.append(k)  # 把物品装备背包
                t -= w[k]  # 背包空间减少
            k += 1  # 继续向后找
        if t == 0:  # 找到了解
            print(stack)
        # 回退过程
        k = stack.pop()  # 把最后一个物品拿出来
        t += w[k]  # 背包总容量加上w[k]
        k += 1  # 装入下一个物品
knapsack(10, [1, 8, 4, 3, 5, 2])