import pandas as pd


def process_excel_data(input_file, output_file):
    """
    从原始Excel文件中提取特定列数据，按car_id递增排序并保存到新的Excel文件

    参数:
        input_file (str): 输入Excel文件路径
        output_file (str): 输出Excel文件路径
    """
    print(f"正在读取文件: {input_file}")

    # 读取Excel文件，跳过第一行（标题行），从第二行开始
    df = pd.read_excel(input_file, skiprows=0)

    # 我们需要的列
    required_columns = [
        'car_id', 'car_model', 'most_sat', 'least_sat', 'space_desc',
        'drive_exp_desc', 'range_desc', 'exterior_desc', 'interior_desc',
        'cost_perf_desc', 'allocation_desc'
    ]

    # 检查所需列是否都存在
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"警告: 以下列在原始数据中不存在: {', '.join(missing_columns)}")
        # 过滤掉不存在的列
        required_columns = [col for col in required_columns if col in df.columns]

    # 只保留我们需要的列
    df_filtered = df[required_columns]

    # 按car_id递增排序
    print("正在按car_id递增排序数据...")
    df_sorted = df_filtered.sort_values(by='car_id')

    # 保存到新的Excel文件
    print(f"正在保存到: {output_file}，共 {len(df_sorted)} 行数据")
    df_sorted.to_excel(output_file, index=False)

    print("处理完成!")
    return output_file


if __name__ == "__main__":
    # 示例使用
    input_file = "vehiclereputationAll_20250307.xlsx"
    output_file = "vehiclenew.xlsx"

    process_excel_data(input_file, output_file)