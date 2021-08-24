import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号

if __name__ == '__main__':
    #   plt.figure(figsize=(19.2, 9))

    # 第一个图：折线图
    plt.subplot(2, 4, 1)
    X = range(0, 100)
    Y = [(x-50)**2 for x in X]
    plt.title("折线图")
    plt.plot(X, Y, c="r", label="图例一")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()

    # 第二个图：柱状图
    plt.subplot(2, 4, 2)
    X=["苹果", "雪梨", "红浪"]
    Y = [100,200,150]
    plt.title("柱状图")
    plt.bar(X, Y, facecolor='#9999ff', edgecolor='white')
    plt.ylabel("Y")

    # 第三个图：条形图
    plt.subplot(2, 4, 3)
    plt.title("条形图")
    plt.barh(X, Y, facecolor='#9900cc', edgecolor='white')

    # 第五个图：饼图
    plt.subplot(2, 4, 5)
    labels=["香蕉", "拔辣", "西柚", "柠檬茶", "王炸"]
    sizes=[100,150,30,75,68]
    explode = (0, 0.1, 0, 0, 0)
    plt.title("饼图")
    plt.pie(sizes, explode=explode, labels=labels, autopct='%.1f%%',shadow=False,startangle=150)

    # 第六个图：散点图
    plt.subplot(2, 4, 6)
    X = range(0, 100)
    Y1 = np.random.randint(0, 20, 100)
    Y2 = np.random.randint(0, 20, 100)
    plt.title("散点图")
    plt.plot(X, Y1, marker=".", c="#9966ff", label="Y1")
    plt.plot(X, Y2, marker="*", c="#6699ff", label="Y2")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(loc="upper right")

    # 第七个图：雷达图
    plt.subplot(2, 4, 7, polar=True)
    plt.title("雷达图", pad=20)
    labels = np.array(["生命值", "灵敏度", "攻击力", "护甲", "守护光环", "威慑力", "成长"])
    dataLength = 7
    data1 = np.random.randint(5, 15, 7)
    data2 = np.random.randint(4, 15, 7)
    angles = np.linspace(0, 2 * np.pi, dataLength, endpoint=False)  # 分割圆周长
    data1 = np.concatenate((data1, [data1[0]]))  # 闭合
    data2 = np.concatenate((data2, [data2[0]]))  # 闭合
    angles = np.concatenate((angles, [angles[0]]))  # 闭合
    plt.polar(angles, data1, '.-', linewidth=1)  # 做极坐标系
    plt.fill(angles, data1, alpha=0.25)  # 填充
    plt.polar(angles, data2, '.-', linewidth=1)  # 做极坐标系
    plt.fill(angles, data2, alpha=0.25)  # 填充
    # plt.thetagrids(angles * 180 / np.pi, labels)  # 设置网格、标签

    # 第把个图：箱线图
    plt.subplot(2, 4, 8)
    A = np.random.randint(0, 20, 100)
    B = np.random.randint(5, 20, 100)
    plt.title("箱线图")
    plt.boxplot((A,B),labels=["A","B"])

    plt.tight_layout(pad=1.08)
    plt.show()