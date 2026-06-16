import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import holidays

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 加载和预处理数据 
def load_and_preprocess_data(file_path, sample_size=None):
    """加载并预处理航班数据，按时间排序"""
    print("=" * 80)
    print("航班延误预测 - 数据特性分析（时间排序版）")
    print("=" * 80)

    try:
        # 读取数据
        if sample_size:
            df = pd.read_csv(file_path, nrows=sample_size)
        else:
            df = pd.read_csv(file_path)
    except Exception as e:
        print(f"读取文件错误: {e}")
        return None

    print(f"原始数据形状: {df.shape}")
    print(f"原始列名: {df.columns.tolist()}")

    # 根据提供的列信息和数据示例匹配列名
    column_mapping = {
        'FL_DATE': '飞行日期',
        'AIRLINE': '航空公司',
        'AIRLINE_DOT': '航空公司_DOT',
        'AIRLINE_CODE': '航空公司代码',
        'DOT_CODE': '运营商代码',
        'FL_NUMBER': '航班号',
        'ORIGIN': '起飞机场',
        'ORIGIN_CITY': '起飞城市',
        'DEST': '到达机场',
        'DEST_CITY': '到达城市',
        'CRS_DEP_TIME': '计划起飞时间',
        'DEP_TIME': '实际起飞时间',
        'DEP_DELAY': '起飞延误',
        'TAXI_OUT': '滑出时间',
        'WHEELS_OFF': '离地时间',
        'WHEELS_ON': '落地时间',
        'TAXI_IN': '滑入时间',
        'CRS_ARR_TIME': '计划到达时间',
        'ARR_TIME': '实际到达时间',
        'ARR_DELAY': '到达延误',
        'CANCELLED': '是否取消',
        'CANCELLATION_CODE': '取消代码',
        'DIVERTED': '是否改航',
        'CRS_ELAPSED_TIME': '计划经过时间',
        'ELAPSED_TIME': '实际经过时间',
        'AIR_TIME': '空中时间',
        'DISTANCE': '距离',
        'DELAY_DUE_CARRIER': '承运人延误',
        'DELAY_DUE_WEATHER': '天气延误',
        'DELAY_DUE_NAS': '航空系统延误',
        'DELAY_DUE_SECURITY': '安全延误',
        'DELAY_DUE_LATE_AIRCRAFT': '飞机延误'
    }

    # 重命名列
    df = df.rename(columns=column_mapping)
    print(f"重命名后的列: {list(df.columns)[:15]}...")  # 只显示前15个

    # 按飞行日期排序
    if '飞行日期' in df.columns:
        try:
            df['飞行日期'] = pd.to_datetime(df['飞行日期'])
            df = df.sort_values('飞行日期').reset_index(drop=True)
            print(f"已按飞行日期排序，时间范围: {df['飞行日期'].min()} 到 {df['飞行日期'].max()}")
        except Exception as e:
            print(f"日期转换错误: {e}")
            df['飞行日期'] = pd.to_datetime(df['飞行日期'], errors='coerce')
            df = df.sort_values('飞行日期').reset_index(drop=True)
            print(f"时间范围（部分转换）: {df['飞行日期'].min()} 到 {df['飞行日期'].max()}")

    # 数据清洗
    print("\n数据清洗:")
    print(f"原始数据行数: {len(df):,}")

    # 移除取消的航班 - 修复dtype错误
    if '是否取消' in df.columns:
        # 检查列是否存在
        if df['是否取消'].dtype == 'object':
            # 如果列是对象类型（字符串），检查是否包含'1'
            cancelled_mask = df['是否取消'].astype(str).str.contains('1')
        else:
            # 如果是数值类型，检查是否等于1
            cancelled_mask = df['是否取消'] == 1

        cancelled_count = cancelled_mask.sum()
        print(f"取消航班数量: {cancelled_count:,}")
        df = df[~cancelled_mask].copy()
        print(f"移除取消航班后: {len(df):,}")

    # 处理延误数据
    delay_cols = ['到达延误', '起飞延误']
    for col in delay_cols:
        if col in df.columns:
            # 转换数据类型
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # 移除极端异常值
            if df[col].notna().sum() > 0:
                q1, q3 = df[col].quantile(0.01), df[col].quantile(0.99)
                iqr = q3 - q1
                lower_bound = q1 - 3 * iqr
                upper_bound = q3 + 3 * iqr
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)].copy()
                print(f"  {col}: 移除{outliers:,}个异常值")

    # 创建时间特征
    if '飞行日期' in df.columns:
        df['年份'] = df['飞行日期'].dt.year
        df['月份'] = df['飞行日期'].dt.month
        df['星期'] = df['飞行日期'].dt.dayofweek
        df['日期'] = df['飞行日期'].dt.day
        df['季度'] = df['飞行日期'].dt.quarter

        # 创建时间序列ID
        df['时间序列ID'] = np.arange(len(df))

    # 创建时间特征（小时和分钟）- 修复转换问题
    time_cols = ['计划起飞时间', '实际起飞时间', '计划到达时间', '实际到达时间']
    for col in time_cols:
        if col in df.columns:
            try:
                # 首先转换为数值
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # 处理NaN值
                df[col] = df[col].fillna(0)

                # 提取小时和分钟
                hour_col = f'{col.replace("时间", "")}小时'
                minute_col = f'{col.replace("时间", "")}分钟'

                # 将浮点数转换为整数并处理小数
                df[hour_col] = (df[col] // 100).astype(int)
                df[minute_col] = (df[col] % 100).astype(int)

                print(f"  {col}: 成功提取小时和分钟")
            except Exception as e:
                print(f"  {col}: 转换失败 - {e}")
                # 如果转换失败，尝试字符串方式
                try:
                    # 转换为字符串并填充
                    time_str = df[col].astype(str).str.zfill(4)
                    df[hour_col] = time_str.str[:2].astype(int)
                    df[minute_col] = time_str.str[2:].astype(int)
                    print(f"  {col}: 使用字符串方式成功提取小时和分钟")
                except Exception as e2:
                    print(f"  {col}: 所有转换方式都失败 - {e2}")

    print(f"\n清洗后数据形状: {df.shape}")
    if '飞行日期' in df.columns:
        print(f"时间范围: {df['飞行日期'].min()} 到 {df['飞行日期'].max()}")

    # 显示一些基本统计信息
    print("\n基本统计信息:")

    # 到达延误统计
    if '到达延误' in df.columns:
        print(f"到达延误 - 均值: {df['到达延误'].mean():.2f}分钟, 标准差: {df['到达延误'].std():.2f}分钟")

    # 起飞延误统计 - 添加完整的统计指标
    if '起飞延误' in df.columns:
        dep_delay_mean = df['起飞延误'].mean()
        dep_delay_std = df['起飞延误'].std()
        dep_delay_min = df['起飞延误'].min()
        dep_delay_max = df['起飞延误'].max()
        dep_delay_median = df['起飞延误'].median()

        print(f"起飞延误 - 均值: {dep_delay_mean:.2f}分钟")
        print(f"起飞延误 - 标准差: {dep_delay_std:.2f}分钟")
        print(f"起飞延误 - 最小值: {dep_delay_min:.2f}分钟")
        print(f"起飞延误 - 最大值: {dep_delay_max:.2f}分钟")
        print(f"起飞延误 - 中位数: {dep_delay_median:.2f}分钟")

    if '距离' in df.columns:
        print(f"飞行距离 - 均值: {df['距离'].mean():.2f}英里, 范围: [{df['距离'].min()}, {df['距离'].max()}]")

    return df

# 特征工程 
def create_simple_features(df):
    """创建简化的特征集"""
    print("\n" + "=" * 80)
    print("创建简化的特征集")
    print("=" * 80)

    sequence_features = []
    categorical_features = []

    # 基础数值特征
    numeric_cols = ['起飞延误', '距离', '空中时间', '滑出时间', '滑入时间']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            sequence_features.append(col)

    # 时间特征
    time_cols = ['月份', '星期', '计划起飞小时', '季度']
    for col in time_cols:
        if col in df.columns:
            sequence_features.append(col)

    # 航空公司特征
    if '航空公司代码' in df.columns:
        categorical_features.append('航空公司代码')

    print(f"创建了 {len(sequence_features)} 个时序特征")
    print(f"创建了 {len(categorical_features)} 个分类特征")

    return df, sequence_features, categorical_features


# ==================== 快速分析函数 ====================
def quick_analysis(df):
    """快速数据分析和可视化"""
    print("\n" + "=" * 80)
    print("快速数据分析")
    print("=" * 80)

    # 基本统计
    print("\n1. 基本统计:")
    print(f"样本数量: {len(df):,}")
    print(f"时间范围: {df['飞行日期'].min()} 到 {df['飞行日期'].max()}")

    # 延误统计
    if '到达延误' in df.columns:
        delay_stats = df['到达延误'].describe()
        print(f"\n到达延误统计:")
        print(f"  最小值: {delay_stats['min']:.1f}分钟")
        print(f"  中位数: {delay_stats['50%']:.1f}分钟")
        print(f"  最大值: {delay_stats['max']:.1f}分钟")
        print(f"  平均值: {delay_stats['mean']:.1f}分钟")
        print(f"  标准差: {delay_stats['std']:.1f}分钟")

    # 起飞延误统计 - 添加完整的统计指标
    if '起飞延误' in df.columns:
        print(f"\n起飞延误详细统计:")
        dep_delay_mean = df['起飞延误'].mean()
        dep_delay_std = df['起飞延误'].std()
        dep_delay_min = df['起飞延误'].min()
        dep_delay_max = df['起飞延误'].max()
        dep_delay_median = df['起飞延误'].median()

        print(f"  最小值: {dep_delay_min:.1f}分钟")
        print(f"  中位数: {dep_delay_median:.1f}分钟")
        print(f"  最大值: {dep_delay_max:.1f}分钟")
        print(f"  平均值: {dep_delay_mean:.1f}分钟")
        print(f"  标准差: {dep_delay_std:.1f}分钟")

    # 创建简单可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('航班数据快速分析', fontsize=14)

    # 1. 延误分布
    if '到达延误' in df.columns:
        axes[0, 0].hist(df['到达延误'].dropna(), bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('到达延误(分钟)')
        axes[0, 0].set_ylabel('频数')
        axes[0, 0].set_title('到达延误分布')
        axes[0, 0].grid(True, alpha=0.3)

    # 2. 起飞vs到达延误关系
    if '起飞延误' in df.columns and '到达延误' in df.columns:
        sample = df.sample(min(1000, len(df)))
        axes[0, 1].scatter(sample['起飞延误'], sample['到达延误'],
                           alpha=0.5, s=10)
        axes[0, 1].set_xlabel('起飞延误(分钟)')
        axes[0, 1].set_ylabel('到达延误(分钟)')
        axes[0, 1].set_title('起飞vs到达延误')
        axes[0, 1].grid(True, alpha=0.3)

    # 3. 月份延误趋势
    if '月份' in df.columns and '到达延误' in df.columns:
        monthly_delay = df.groupby('月份')['到达延误'].mean()
        axes[1, 0].bar(monthly_delay.index, monthly_delay.values, alpha=0.7)
        axes[1, 0].set_xlabel('月份')
        axes[1, 0].set_ylabel('平均延误(分钟)')
        axes[1, 0].set_title('月度平均延误')
        axes[1, 0].grid(True, alpha=0.3)

    # 4. 小时延误趋势
    if '计划起飞小时' in df.columns and '到达延误' in df.columns:
        hourly_delay = df.groupby('计划起飞小时')['到达延误'].mean()
        axes[1, 1].plot(hourly_delay.index, hourly_delay.values, marker='o')
        axes[1, 1].set_xlabel('起飞小时')
        axes[1, 1].set_ylabel('平均延误(分钟)')
        axes[1, 1].set_title('小时平均延误')
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # ==================== 改进的特征相关性分析 ====================
    print("\n2. 所有特征相关性分析:")

    # 获取所有数值型特征
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()

    # 排除不需要分析的特征
    exclude_features = ['时间序列ID', '年份']  # 可以根据需要调整
    numeric_features = [f for f in numeric_features if f not in exclude_features]

    print(f"分析的特征数量: {len(numeric_features)}")
    print(f"分析的特征: {numeric_features}")

    # 限制特征数量，避免相关性矩阵过大
    if len(numeric_features) > 20:
        print(f"\n特征数量较多({len(numeric_features)})，将选择最重要的特征进行分析...")

        # 计算每个特征与目标变量的相关性（如果有）
        if '到达延误' in numeric_features:
            corr_with_target = df[numeric_features].corr()['到达延误'].abs().sort_values(ascending=False)
            top_features = corr_with_target.head(20).index.tolist()
            print(f"选择与目标变量相关性最高的20个特征进行分析")
            numeric_features = top_features
        else:
            # 如果没有目标变量，选择前20个特征
            numeric_features = numeric_features[:20]
            print(f"选择前20个特征进行分析")

    if len(numeric_features) >= 2:
        # 计算相关系数矩阵
        corr_matrix = df[numeric_features].corr()

        print("\n特征相关系数矩阵(前10行):")
        # 只显示前10行前10列以避免输出过长
        display_rows = min(10, len(corr_matrix))
        display_cols = min(10, len(corr_matrix.columns))
        print(corr_matrix.iloc[:display_rows, :display_cols].round(3))

        print("\n与到达延误最相关的特征:")
        if '到达延误' in corr_matrix.columns:
            target_corr = corr_matrix['到达延误'].abs().sort_values(ascending=False)
            # 显示前15个最相关的特征
            top_corr_features = target_corr.head(15)
            for feature, corr_value in top_corr_features.items():
                if feature != '到达延误':
                    print(f"  {feature}: {corr_matrix.loc[feature, '到达延误']:.3f}")

        # 可视化相关系数矩阵
        fig, ax = plt.subplots(figsize=(12, 10))

        # 使用更清晰的颜色方案
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        sns.heatmap(corr_matrix,
                    mask=mask,
                    annot=True,
                    fmt='.2f',
                    cmap='RdBu_r',
                    center=0,
                    square=True,
                    linewidths=0.5,
                    cbar_kws={"shrink": 0.8},
                    ax=ax)
        ax.set_title('所有特征相关系数热力图', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

        # 创建特征相关性条形图
        if '到达延误' in corr_matrix.columns:
            fig, ax = plt.subplots(figsize=(10, 8))

            # 获取与到达延误相关性最高的特征（排除自身）
            target_corr = corr_matrix['到达延误'].drop('到达延误', errors='ignore')
            target_corr_abs = target_corr.abs().sort_values(ascending=False).head(15)

            # 创建条形图
            colors = ['red' if x < 0 else 'green' for x in target_corr[target_corr_abs.index]]
            bars = ax.barh(range(len(target_corr_abs)), target_corr_abs.values)

            # 设置颜色
            for i, bar in enumerate(bars):
                bar.set_color(colors[i])

            ax.set_yticks(range(len(target_corr_abs)))
            ax.set_yticklabels(target_corr_abs.index)
            ax.set_xlabel('与到达延误的相关系数绝对值')
            ax.set_title('与到达延误最相关的15个特征')
            ax.grid(True, alpha=0.3)

            # 添加相关系数值标签
            for i, (feature, value) in enumerate(target_corr_abs.items()):
                ax.text(value + 0.01, i, f'{target_corr[feature]:.3f}',
                        va='center', fontsize=9)

            plt.tight_layout()
            plt.show()
    else:
        print("数值特征不足，无法进行相关性分析")


# 模型推荐
def model_recommendation(df):
    """基于数据特性的模型推荐"""
    print("\n" + "=" * 80)
    print("模型推荐")
    print("=" * 80)

    print("\n1. 数据特性总结:")
    print(f"  样本数量: {len(df):,}")

    if '飞行日期' in df.columns:
        date_range = df['飞行日期'].max() - df['飞行日期'].min()
        print(f"  时间跨度: {date_range.days}天")
        print(f"  数据频率: {len(df) / date_range.days:.1f} 航班/天")

    if '到达延误' in df.columns:
        print(f"  平均延误: {df['到达延误'].mean():.2f}分钟")
        print(f"  延误标准差: {df['到达延误'].std():.2f}分钟")

    # 起飞延误统计 - 添加到模型推荐部分
    if '起飞延误' in df.columns:
        print(f"\n起飞延误统计:")
        print(f"  平均起飞延误: {df['起飞延误'].mean():.2f}分钟")
        print(f"  起飞延误标准差: {df['起飞延误'].std():.2f}分钟")
        print(f"  起飞延误最小值: {df['起飞延误'].min():.2f}分钟")
        print(f"  起飞延误最大值: {df['起飞延误'].max():.2f}分钟")
        print(f"  起飞延误中位数: {df['起飞延误'].median():.2f}分钟")

    # 基于相关性分析的模型推荐
    print("\n2. 基于相关性分析的模型建议:")

    # 获取数值特征
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()

    if '到达延误' in numeric_features and len(numeric_features) > 1:
        # 计算特征与目标的相关性
        corr_matrix = df[numeric_features].corr()
        if '到达延误' in corr_matrix.columns:
            target_corr = corr_matrix['到达延误'].abs()
            high_corr_features = target_corr[target_corr > 0.3].index.tolist()
            high_corr_features = [f for f in high_corr_features if f != '到达延误']

            if high_corr_features:
                print(f"  发现 {len(high_corr_features)} 个与到达延误高度相关的特征")
                print(f"  高度相关特征: {high_corr_features[:5]}...")  # 只显示前5个

                # 基于特征相关性推荐模型
                if len(high_corr_features) > 5:
                    print("  建议使用: 梯度提升树(如XGBoost、LightGBM)或随机森林")
                else:
                    print("  建议使用: 线性回归或简单树模型")
            else:
                print("  未发现与到达延误高度相关的特征，可能需要特征工程")


#主程序
if __name__ == "__main__":
    # 加载数据 - 使用您的文件路径
    file_path = "../data_set/flights_sample_3m.csv"  # 修改为您的文件路径

    print("开始加载数据...")

    # 首先尝试小样本
    try:
        df = load_and_preprocess_data(file_path, sample_size=3000000)
    except Exception as e:
        print(f"加载数据失败: {e}")
        print("尝试加载更小的样本...")
        try:
            df = load_and_preprocess_data(file_path, sample_size=10000)
        except Exception as e2:
            print(f"仍然失败: {e2}")
            print("尝试最小样本...")
            df = load_and_preprocess_data(file_path, sample_size=5000)

    if df is not None and len(df) > 0:
        print(f"\n成功加载数据: {len(df):,}行")

        # 快速分析
        quick_analysis(df)

        # 创建特征
        df_with_features, sequence_features, categorical_features = create_simple_features(df)

        # 模型推荐
        model_recommendation(df_with_features)

        # 保存处理后的数据
        output_path = "flight_delay_processed.csv"
        # 只保存部分列以减小文件大小
        columns_to_save = ['飞行日期', '航空公司代码', '起飞机场', '到达机场',
                           '到达延误', '起飞延误', '距离', '空中时间',
                           '月份', '星期', '计划起飞小时'] + sequence_features

        # 只保留实际存在的列
        columns_to_save = [col for col in columns_to_save if col in df_with_features.columns]
        df_with_features[columns_to_save].to_csv(output_path, index=False)
        print(f"\n处理后的数据已保存到: {output_path}")
        print(f"保存列: {len(columns_to_save)}个")

        # 保存特征信息
        features_info = {
            'sequence_features': sequence_features,
            'categorical_features': categorical_features,
            'total_features': len(sequence_features) + len(categorical_features)
        }

        import json

        with open('features_info.json', 'w', encoding='utf-8') as f:
            json.dump(features_info, f, ensure_ascii=False, indent=2)
        print(f"特征信息已保存到: features_info.json")

        print("\n分析完成！")
    else:
        print("数据加载失败或数据为空")