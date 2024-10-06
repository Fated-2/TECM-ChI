import numpy as np


def save_to_file(x_sequence, y_sequence, cell_line):
    # 把序列的正样本和负样本变成1:1的平衡状态 , 并存到文件中
    label = np.loadtxt('data/' + cell_line + '/label.txt')
    index_pos = np.where(label == 1)
    index_neg = np.where(label == 0)
    num = len(index_pos[0])
    print(cell_line, '中正样本数量 : ', num)
    index_neg_sel = np.random.choice(index_neg[0], num, replace=False)  # 随机选取负样本
    x_pos = x_sequence[index_pos]
    x_neg = x_sequence[index_neg_sel]
    y_pos = y_sequence[index_pos]
    y_neg = y_sequence[index_neg_sel]
    data_pos = np.concatenate((x_pos, y_pos), axis=0)
    data_neg = np.concatenate((x_neg, y_neg), axis=0)
    genomics = np.loadtxt('data/' + cell_line + '/genomics.csv', delimiter=',')  # shape(9427, 1148)
    genomics_pos = genomics[index_pos]
    genomics_neg = genomics[index_neg_sel]
    genomics = np.concatenate((genomics_pos, genomics_pos, genomics_neg, genomics_neg), axis=0)
    # genomics = np.concatenate((genomics_pos, genomics_neg), axis=0)
    # 将正负样本数据存储到2个文件中
    with open('data/' + cell_line + '/positive', 'w') as f:
        for sample in data_pos:
            f.write(sample + '\n')
    with open('data/' + cell_line + '/negative', 'w') as f:
        for sample in data_neg:
            f.write(sample + '\n')
    return genomics


def load_pos_neg(x_sequence, cell_line):
    # 把序列的正样本和负样本变成1:1的平衡状态 , 并存到文件中
    label = np.loadtxt('data/' + cell_line + '/label.txt')
    index_pos = np.where(label == 1)
    index_neg = np.where(label == 0)
    num = len(index_pos[0])
    print(cell_line, '中正样本数量 : ', num)
    index_neg_sel = np.random.choice(index_neg[0], num, replace=False)  # 随机选取负样本
    x_pos = x_sequence[index_pos]
    x_neg = x_sequence[index_neg_sel]
    # y_pos = y_sequence[index_pos]
    # y_neg = y_sequence[index_neg_sel]
    # data_pos = np.concatenate((x_pos, y_pos), axis=0)
    # data_neg = np.concatenate((x_neg, y_neg), axis=0)
    genomics = np.loadtxt('data/' + cell_line + '/genomics.csv', delimiter=',')  # shape(9427, 1148)
    genomics_pos = genomics[index_pos]
    genomics_neg = genomics[index_neg_sel]
    genomics = np.concatenate((genomics_pos, genomics_neg), axis=0)
    return x_pos, x_neg, genomics


def load(cell_line):
    x_region = 'data/' + cell_line + '/' + 'x.fasta'
    y_region = 'data/' + cell_line + '/' + 'y.fasta'
    x_sequence = []
    y_sequence = []

    f = open(x_region, 'r')
    for i in f.readlines():
        if i[0] != ' ':
            if i[0] != '>':
                x_sequence.append(i.strip().upper())
    f.close()

    f = open(y_region, 'r')
    for i in f.readlines():
        if i[0] != ' ':
            if i[0] != '>':
                y_sequence.append(i.strip().upper())
    f.close()
    return x_sequence, y_sequence

