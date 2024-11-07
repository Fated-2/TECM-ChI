import os
import matplotlib.pyplot as plt

# 获取当前目录
current_directory = os.getcwd()

# 查找所有以".txt"为扩展名的文件
txt_files = [file for file in os.listdir(current_directory) if file.endswith(".txt")]

# 创建一个图形对象
plt.figure()

# 遍历每个txt文件
for file in txt_files:
    # 提取文件名作为图例标签
    legend_label = os.path.splitext(file)[0]

    # 读取txt文件中的ROC数据
    with open(file, 'r') as f:
        roc_data = f.readlines()    # 存成string类型的list数组

    # 提取真阳性率和假阳性率
    true_positive_rate = []
    false_positive_rate = []
    for line in roc_data:
        line = line.strip().split('\t')
        true_positive_rate.append(float(line[1]))
        false_positive_rate.append(float(line[0]))

    # 绘制ROC曲线
    plt.plot(false_positive_rate, true_positive_rate, label=legend_label)

# 添加图例
plt.legend()
# 添加图名
plt.title('ROC Curves')
# 添加坐标轴标签
plt.xlabel('FPR')
plt.ylabel('TPR')

# 显示图像
plt.show()
