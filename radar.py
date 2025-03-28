import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

font = {'family': 'Hiragino Sans GB', 'size': 12}
processed = pd.read_excel("radars.xlsx")
sns.set(font_scale=1.2)
plt.rc('font', family='Hiragino Sans GB')
plt.style.use('ggplot')  # 使用ggplot的绘图风格

# 创建存放雷达图的文件夹
output_dir = "radars"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 提取数据
for _, row in processed.iterrows():
    brand = row['旗下品牌名']

    # 获取数据并进行归一化处理
    value = [
        row['空间'] / 5,  # 将空间的值归一化到 [0, 1] 范围
        row['智能化'] / 5,  # 将智能化的值归一化到 [0, 1] 范围
        row['性价比'] / 5,  # 将性价比的值归一化到 [0, 1] 范围
        row['内饰'] / 5,  # 将内饰的值归一化到 [0, 1] 范围
        row['外观'] / 5,  # 将外观的值归一化到 [0, 1] 范围
        row['续航'] / 5,  # 将续航的值归一化到 [0, 1] 范围
        row['最低价>20']  # 该项数据已是0-1范围，无需归一化
    ]

    feature = ["空间", "智能化", "性价比", "内饰", "外观", "续航", "最低价>20"]

    # 设置每个数据点的显示位置，在雷达图上用角度表示
    angles = np.linspace(0, 2 * np.pi, len(feature), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))
    feature = np.concatenate((feature, [feature[0]]))

    # 绘图
    fig = plt.figure(figsize=(8, 8))
    # 设置为极坐标格式
    ax = fig.add_subplot(111, polar=True)

    # 拼接数据首尾，使图形中线条封闭
    value = np.concatenate((value, [value[0]]))

    # 绘制折线图
    ax.plot(angles, value, 'o-', linewidth=2, label=brand)

    # 填充颜色
    ax.fill(angles, value, alpha=0.25)

    # 设置图标上的角度划分刻度，为每个数据点处添加标签
    ax.set_thetagrids(angles * 180 / np.pi, feature, fontsize=14, style='italic')

    # 设置雷达图的范围
    ax.set_ylim(0, 1)  # 将所有数据的范围设置为 [0, 1]

    # 设置雷达图的0度起始位置
    ax.set_theta_zero_location('N')

    # 设置雷达图的坐标值显示角度，相对于起始角度的偏移量
    ax.set_rlabel_position(270)

    # 添加标题
    plt.title(f'{brand} 雷达图', fontsize=14)

    # 添加网格线
    plt.grid(True)

    # 保存图像到 radars 文件夹
    save_path = os.path.join(output_dir, f'{brand}_雷达图.png')
    plt.savefig(save_path, dpi=300)
    plt.close()  # 关闭当前图表以防止显示过多图像