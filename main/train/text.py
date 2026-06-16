import pandas as pd
import numpy as np
import os
import random


def select_flights_with_min_records(input_csv, output_csv, num_flights=2000, min_records=20, random_seed=42):
    # 设置随机种子
    random.seed(random_seed)
    np.random.seed(random_seed)

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"输入文件不存在: {input_csv}")

    print(f"读取原始数据: {input_csv}")

    # 分块读取以节省内存（如果文件很大）
    chunk_size = 100000
    chunks = []
    for chunk in pd.read_csv(input_csv, chunksize=chunk_size, low_memory=False):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    print(f"原始数据总行数: {len(df)}")

    # 确保有 FL_NUMBER 列
    if 'FL_NUMBER' not in df.columns:
        raise ValueError("CSV文件中缺少 FL_NUMBER 列，无法识别航班号。")

    # 统计每个航班的记录数
    flight_counts = df['FL_NUMBER'].value_counts()
    print(f"共有 {len(flight_counts)} 个唯一航班号")
    print(f"航班记录数统计: min={flight_counts.min()}, max={flight_counts.max()}, mean={flight_counts.mean():.1f}")

    # 筛选出记录数 >= min_records 的航班
    valid_flights = flight_counts[flight_counts >= min_records].index.tolist()
    print(f"记录数 ≥ {min_records} 的航班数量: {len(valid_flights)}")

    if len(valid_flights) == 0:
        raise ValueError(f"没有找到记录数 ≥ {min_records} 的航班，请减小 min_records 或检查数据。")

    # 随机选择指定数量的航班
    if num_flights > len(valid_flights):
        print(f"警告: 请求的航班数 {num_flights} 超过符合条件的航班数量 {len(valid_flights)}，将选择全部。")
        selected_flights = valid_flights
    else:
        selected_flights = random.sample(valid_flights, num_flights)

    print(f"随机选择了 {len(selected_flights)} 个航班")

    # 筛选这些航班的所有记录
    selected_set = set(selected_flights)
    selected_df = df[df['FL_NUMBER'].isin(selected_set)].copy()

    # 按航班号和日期排序
    if 'FL_DATE' in selected_df.columns:
        selected_df['FL_DATE'] = pd.to_datetime(selected_df['FL_DATE'])
        selected_df = selected_df.sort_values(['FL_NUMBER', 'FL_DATE'])
        print("已按航班号和日期排序")

    print(f"筛选后数据总行数: {len(selected_df)}")

    # 再次统计选中航班的记录数（确认都满足条件）
    selected_counts = selected_df['FL_NUMBER'].value_counts()
    print("选中航班的记录数统计:")
    print(f"  最小记录数: {selected_counts.min()}")
    print(f"  最大记录数: {selected_counts.max()}")
    print(f"  平均记录数: {selected_counts.mean():.1f}")

    # 保存到新CSV
    selected_df.to_csv(output_csv, index=False)
    print(f"数据已保存至: {output_csv}")
    print("完成！")
    return selected_df


if __name__ == "__main__":
    # 设置输入输出路径
    base_dir = "../train/data_set"
    input_file = os.path.join(base_dir, "flights_sample_3m.csv")
    output_file = os.path.join(base_dir, "flight_data.csv")

    # 执行选择
    select_flights_with_min_records(
        input_csv=input_file,
        output_csv=output_file,
        num_flights=50,  # 选择2000个航班
        min_records=200,  # 每个航班至少200条历史记录
        random_seed=42
    )