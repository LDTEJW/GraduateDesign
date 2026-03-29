# predict.py (修复版)
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import xgboost as xgb
import json
from datetime import datetime
import holidays
import warnings
import traceback

warnings.filterwarnings('ignore')

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


class HybridLSTM(nn.Module):
    """混合LSTM模型"""

    def __init__(self, seq_input_size, static_input_size=0,
                 lstm_hidden_size=64, num_lstm_layers=1,
                 dropout=0.2):
        super(HybridLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=seq_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
            bidirectional=True
        )

        if static_input_size > 0:
            self.static_encoder = nn.Sequential(
                nn.Linear(static_input_size, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            static_output_size = 64
        else:
            self.static_encoder = None
            static_output_size = 0

        lstm_output_size = lstm_hidden_size * 2
        fusion_input_size = lstm_output_size + static_output_size

        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )

        self.output_layer = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout * 0.3),
            nn.Linear(32, 1)
        )

    def forward(self, x_seq, x_static=None):
        lstm_out, (h_n, c_n) = self.lstm(x_seq)
        lstm_last = lstm_out[:, -1, :]

        if self.static_encoder is not None and x_static is not None:
            static_encoded = self.static_encoder(x_static)
            combined = torch.cat([lstm_last, static_encoded], dim=1)
        else:
            combined = lstm_last

        fused = self.fusion(combined)
        output = self.output_layer(fused)

        return output


class EnsembleModel:
    """集成模型"""

    def __init__(self, lstm_model, xgb_model, meta_model):
        self.lstm_model = lstm_model
        self.xgb_model = xgb_model
        self.meta_model = meta_model

    def predict(self, X_lstm, X_xgb, device='cpu'):
        lstm_pred = self._get_lstm_predictions(X_lstm, device)
        xgb_pred = self._get_xgb_predictions(X_xgb)

        meta_features = np.hstack([
            lstm_pred,
            xgb_pred,
            lstm_pred - xgb_pred,
            (lstm_pred + xgb_pred) / 2,
            np.abs(lstm_pred - xgb_pred)
        ])

        ensemble_pred = self.meta_model.predict(meta_features)
        return ensemble_pred.reshape(-1, 1)

    def _get_lstm_predictions(self, X_lstm, device='cpu'):
        self.lstm_model.eval()

        if isinstance(X_lstm, tuple):
            X_seq, X_static = X_lstm
        else:
            X_seq = X_lstm
            X_static = None

        with torch.no_grad():
            X_seq_tensor = torch.FloatTensor(X_seq).to(device)
            if X_static is not None:
                X_static_tensor = torch.FloatTensor(X_static).to(device)
                pred = self.lstm_model(X_seq_tensor, X_static_tensor).cpu().numpy()
            else:
                pred = self.lstm_model(X_seq_tensor, None).cpu().numpy()

        return pred

    def _get_xgb_predictions(self, X_xgb):
        dmatrix = xgb.DMatrix(X_xgb)
        return self.xgb_model.predict(dmatrix).reshape(-1, 1)


class FlightDelayPredictor:
    """航班延误预测服务"""

    def __init__(self, model_dir='../saved_models'):
        self.model_dir = model_dir
        self.device = device
        self.models = {}
        self.scalers = {}
        self.feature_info = {}
        self.label_encoders = {}
        self.seq_features = []  # 时序特征列表
        self.static_features = []  # 静态特征列表
        self.seq_length = 10
        self.us_holidays = holidays.US()

        self._load_models()

    def _load_models(self):
        """加载所有模型和配置文件"""
        print("=" * 60)
        print("加载模型和配置文件...")
        print("=" * 60)

        try:
            # 1. 加载特征信息 (优先使用features_info.json)
            json_path = os.path.join(self.model_dir, 'features_info.json')
            pkl_path = os.path.join(self.model_dir, 'feature_info.pkl')

            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    self.feature_info = json.load(f)
                print(f"✓ 从JSON加载特征信息成功")
            elif os.path.exists(pkl_path):
                self.feature_info = joblib.load(pkl_path)
                print(f"✓ 从PKL加载特征信息成功")

            # 2. 加载标准化器
            scalers_path = os.path.join(self.model_dir, 'scalers.pkl')
            if os.path.exists(scalers_path):
                self.scalers = joblib.load(scalers_path)
                print(f"✓ 标准化器加载成功")

                # 从scaler中提取特征列表
                if 'seq_features' in self.scalers:
                    self.seq_features = self.scalers['seq_features']
                if 'static_features' in self.scalers:
                    self.static_features = self.scalers['static_features']

            # 3. 加载标签编码器
            encoders_path = os.path.join(self.model_dir, 'label_encoders.pkl')
            if os.path.exists(encoders_path):
                self.label_encoders = joblib.load(encoders_path)
                print(f"✓ 标签编码器加载成功")
                print(f"  编码器类型: {list(self.label_encoders.keys())}")

            # 4. 加载LSTM模型
            lstm_path = os.path.join(self.model_dir, 'hybrid_lstm_model.pth')
            if os.path.exists(lstm_path):
                checkpoint = torch.load(lstm_path, map_location=self.device)

                # 获取模型参数
                if 'seq_input_size' in checkpoint:
                    seq_input_size = checkpoint['seq_input_size']
                else:
                    seq_input_size = len(self.seq_features) if self.seq_features else 24

                if 'static_input_size' in checkpoint:
                    static_input_size = checkpoint['static_input_size']
                else:
                    static_input_size = len(self.static_features) if self.static_features else 0

                if 'seq_length' in checkpoint:
                    self.seq_length = checkpoint['seq_length']

                self.models['lstm'] = HybridLSTM(
                    seq_input_size=seq_input_size,
                    static_input_size=static_input_size,
                    lstm_hidden_size=64,
                    num_lstm_layers=1,
                    dropout=0.2
                ).to(self.device)

                self.models['lstm'].load_state_dict(checkpoint['model_state_dict'])
                self.models['lstm'].eval()

                print(f"✓ LSTM模型加载成功")
                print(f"  序列长度: {self.seq_length}")
                print(f"  时序特征数: {seq_input_size}")
                print(f"  静态特征数: {static_input_size}")
            else:
                print(f"⚠ LSTM模型文件不存在: {lstm_path}")

            # 5. 加载XGBoost模型
            xgb_path = os.path.join(self.model_dir, 'optimized_xgboost_model.json')
            if os.path.exists(xgb_path):
                self.models['xgb'] = xgb.Booster()
                self.models['xgb'].load_model(xgb_path)
                print(f"✓ XGBoost模型加载成功")

            # 6. 加载集成模型
            ensemble_path = os.path.join(self.model_dir, 'ensemble_model.pkl')
            if os.path.exists(ensemble_path):
                meta_model = joblib.load(ensemble_path)
                self.models['ensemble'] = EnsembleModel(
                    self.models['lstm'],
                    self.models['xgb'],
                    meta_model.meta_model if hasattr(meta_model, 'meta_model') else meta_model
                )
                print(f"✓ 集成模型加载成功")

            print("\n" + "=" * 60)
            print("特征信息:")
            print(f"  时序特征 ({len(self.seq_features)}个):")
            for i, feat in enumerate(self.seq_features[:10]):
                print(f"    {i + 1}. {feat}")
            if len(self.seq_features) > 10:
                print(f"    ... 还有 {len(self.seq_features) - 10} 个")

            print(f"\n  静态特征 ({len(self.static_features)}个):")
            for i, feat in enumerate(self.static_features[:10]):
                print(f"    {i + 1}. {feat}")
            if len(self.static_features) > 10:
                print(f"    ... 还有 {len(self.static_features) - 10} 个")
            print("=" * 60)

        except Exception as e:
            print(f"❌ 加载模型失败: {e}")
            traceback.print_exc()

    def preprocess_single_flight(self, flight_data):
        """预处理单条航班数据 - 生成所有需要的特征"""

        # 创建DataFrame
        df = pd.DataFrame([flight_data])

        # 转换日期时间
        if 'flight_date' in df.columns:
            df['flight_date'] = pd.to_datetime(df['flight_date'])

        # ========== 基础时间特征 ==========
        if 'flight_date' in df.columns:
            df['year'] = df['flight_date'].dt.year
            df['month'] = df['flight_date'].dt.month
            df['day'] = df['flight_date'].dt.day
            df['day_of_week'] = df['flight_date'].dt.dayofweek
            df['day_of_year'] = df['flight_date'].dt.dayofyear
            df['week_of_year'] = df['flight_date'].dt.isocalendar().week.astype(int)
            df['quarter'] = df['flight_date'].dt.quarter
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            df['is_holiday'] = df['flight_date'].apply(lambda x: 1 if x in self.us_holidays else 0)
            df['season'] = (df['month'] % 12 // 3 + 1).astype(int)

        # ========== 时间特征（小时分钟） ==========
        time_cols = ['scheduled_departure_time', 'scheduled_arrival_time']

        for col in time_cols:
            if col in df.columns:
                hour_col = f"{col}_hour"
                minute_col = f"{col}_minute"
                period_col = f"{col}_period"

                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                df[hour_col] = (df[col] // 100).astype(int) % 24
                df[minute_col] = (df[col] % 100).astype(int)

                # 时间段分类
                df[period_col] = pd.cut(df[hour_col],
                                        bins=[-1, 6, 12, 18, 24],
                                        labels=[0, 1, 2, 3]).astype(float)

        # ========== 距离特征 ==========
        if 'distance' in df.columns:
            df['distance'] = pd.to_numeric(df['distance'], errors='coerce').fillna(0)
            df['distance_category'] = pd.cut(
                df['distance'],
                bins=[0, 500, 1000, 1500, 2500, float('inf')],
                labels=[0, 1, 2, 3, 4]
            ).astype(float)

        # ========== 延误相关特征（创建默认值） ==========
        df['departure_delay'] = 0
        df['is_delayed_departure'] = 0
        df['departure_delay_abs'] = 0
        df['departure_delay_category'] = 3  # 默认正常

        # ========== 路线特征 ==========
        if 'origin' in df.columns and 'destination' in df.columns:
            df['origin'] = df['origin'].fillna('UNKNOWN')
            df['destination'] = df['destination'].fillna('UNKNOWN')
            df['route'] = df['origin'] + '_' + df['destination']

        # ========== 机场统计特征（使用编码代替） ==========
        for col in ['origin', 'destination']:
            if col in df.columns:
                # 机场编码
                if col in self.label_encoders:
                    try:
                        df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col])
                    except:
                        df[f'{col}_encoded'] = -1
                else:
                    df[f'{col}_encoded'] = -1

                # 机场统计特征（使用默认值）
                df[f'{col}_mean_delay'] = 0
                df[f'{col}_std_delay'] = 0
                df[f'{col}_flight_count'] = 100  # 默认值

        # ========== 航空公司特征 ==========
        if 'airline_code' in df.columns:
            df['airline_code'] = df['airline_code'].fillna('UNKNOWN')

            if 'airline_code' in self.label_encoders:
                try:
                    df['airline_code_encoded'] = self.label_encoders['airline_code'].transform(df['airline_code'])
                except:
                    df['airline_code_encoded'] = -1
            else:
                df['airline_code_encoded'] = -1

            # 航空公司统计特征（使用默认值）
            df['airline_mean_delay'] = 0
            df['airline_std_delay'] = 0
            df['airline_flight_count'] = 100
            df['airline_dep_delay_mean'] = 0

        # ========== 路线特征 ==========
        if 'route' in df.columns:
            if 'route' in self.label_encoders:
                try:
                    df['route_encoded'] = self.label_encoders['route'].transform(df['route'])
                except:
                    df['route_encoded'] = -1
            else:
                df['route_encoded'] = -1

            # 路线统计特征（使用默认值）
            df['route_mean_delay'] = 0
            df['route_std_delay'] = 0
            df['route_flight_count'] = 50
            df['route_mean_distance'] = df['distance'] if 'distance' in df.columns else 1000

        # ========== 时间交互特征 ==========
        if 'month' in df.columns and 'day_of_week' in df.columns:
            df['month_day_interaction'] = df['month'] * 10 + df['day_of_week']

        # ========== 滞后特征（创建默认值） ==========
        lag_features = ['arrival_delay_lag_1', 'arrival_delay_lag_2', 'arrival_delay_lag_3',
                        'arrival_delay_lag_7', 'departure_delay_lag_1', 'departure_delay_lag_2',
                        'departure_delay_lag_3', 'departure_delay_lag_7',
                        'arrival_delay_ma_3', 'arrival_delay_ma_7', 'arrival_delay_ma_14',
                        'departure_delay_ma_3', 'departure_delay_ma_7', 'departure_delay_ma_14']

        for lag in lag_features:
            df[lag] = 0

        # 填充所有缺失值
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(0)

        # 确保所有需要的特征都存在
        all_required_features = set(self.seq_features + self.static_features)
        for feat in all_required_features:
            if feat not in df.columns:
                df[feat] = 0

        return df

    def create_sequences(self, df):
        """为单条数据创建序列"""

        print(f"\n创建序列:")
        print(f"  时序特征列表: {self.seq_features[:10]}... (共{len(self.seq_features)}个)")
        print(f"  静态特征列表: {self.static_features[:10]}... (共{len(self.static_features)}个)")

        # 准备时序数据
        if self.seq_features:
            # 只选择存在的特征
            existing_seq_features = [f for f in self.seq_features if f in df.columns]
            print(f"  存在的时序特征: {len(existing_seq_features)}/{len(self.seq_features)}")

            if existing_seq_features:
                seq_df = df[existing_seq_features].copy()

                # 标准化
                if 'seq_scaler' in self.scalers:
                    try:
                        seq_scaled = self.scalers['seq_scaler'].transform(seq_df)
                        print(f"  时序数据标准化成功，形状: {seq_scaled.shape}")
                    except Exception as e:
                        print(f"  时序数据标准化失败: {e}")
                        # 如果标准化失败，使用原始数据
                        seq_scaled = seq_df.values.astype(np.float32)
                else:
                    seq_scaled = seq_df.values.astype(np.float32)

                # 创建序列（重复数据）
                seq_array = np.repeat(seq_scaled, self.seq_length, axis=0)
                seq_array = seq_array.reshape(1, self.seq_length, -1)
                print(f"  时序数据形状: {seq_array.shape}")
            else:
                print(f"  警告: 没有时序特征，创建零数组")
                seq_array = np.zeros((1, self.seq_length, 1))
        else:
            print(f"  警告: 时序特征列表为空，创建零数组")
            seq_array = np.zeros((1, self.seq_length, 24))  # 假设24个时序特征

        # 准备静态数据
        if self.static_features:
            existing_static_features = [f for f in self.static_features if f in df.columns]
            print(f"  存在的静态特征: {len(existing_static_features)}/{len(self.static_features)}")

            if existing_static_features:
                static_df = df[existing_static_features].copy()

                if 'static_scaler' in self.scalers and self.scalers['static_scaler'] is not None:
                    try:
                        static_scaled = self.scalers['static_scaler'].transform(static_df)
                        print(f"  静态数据标准化成功，形状: {static_scaled.shape}")
                    except Exception as e:
                        print(f"  静态数据标准化失败: {e}")
                        static_scaled = static_df.values.astype(np.float32)
                else:
                    static_scaled = static_df.values.astype(np.float32)

                static_array = static_scaled.reshape(1, -1)
            else:
                print(f"  警告: 没有静态特征，创建零数组")
                static_array = np.zeros((1, 1))
        else:
            print(f"  警告: 静态特征列表为空，创建零数组")
            static_array = np.zeros((1, 1))

        print(f"  静态数据形状: {static_array.shape}")

        return seq_array, static_array

    def extract_xgb_features(self, seq_array, static_array):
        """提取XGBoost特征"""

        n_timesteps = seq_array.shape[1]
        n_features = seq_array.shape[2]

        features = []

        # 对每个时序特征计算统计量
        for i in range(n_features):
            feature_data = seq_array[0, :, i]

            features.append(np.mean(feature_data))  # 均值
            features.append(np.std(feature_data))  # 标准差
            features.append(np.max(feature_data))  # 最大值
            features.append(np.min(feature_data))  # 最小值
            features.append(feature_data[-1])  # 最后值
            features.append(feature_data[-1] - feature_data[0])  # 变化

        # 添加静态特征
        if static_array is not None and static_array.shape[1] > 1:
            for i in range(static_array.shape[1]):
                features.append(static_array[0, i])

        feature_array = np.array(features).reshape(1, -1)
        print(f"  XGBoost特征形状: {feature_array.shape}")

        return feature_array

    def predict(self, flight_data):
        """预测单条航班数据"""

        try:
            print("\n" + "=" * 60)
            print("开始预测单条航班数据")
            print("=" * 60)

            # 预处理
            df = self.preprocess_single_flight(flight_data)
            print(f"\n预处理完成，DataFrame形状: {df.shape}")
            print(f"DataFrame列: {list(df.columns)}")

            # 创建序列
            seq_array, static_array = self.create_sequences(df)

            # 提取XGBoost特征
            xgb_features = self.extract_xgb_features(seq_array, static_array)

            # 预测
            predictions = {}

            # LSTM预测
            if 'lstm' in self.models:
                print("\n执行LSTM预测...")
                with torch.no_grad():
                    seq_tensor = torch.FloatTensor(seq_array).to(self.device)
                    static_tensor = torch.FloatTensor(static_array).to(self.device)
                    lstm_pred = self.models['lstm'](seq_tensor, static_tensor).cpu().numpy()[0, 0]
                    predictions['lstm'] = float(lstm_pred)
                    print(f"  LSTM预测结果: {lstm_pred:.2f}分钟")

            # XGBoost预测
            if 'xgb' in self.models:
                print("\n执行XGBoost预测...")
                dmatrix = xgb.DMatrix(xgb_features)
                xgb_pred = self.models['xgb'].predict(dmatrix)[0]
                predictions['xgboost'] = float(xgb_pred)
                print(f"  XGBoost预测结果: {xgb_pred:.2f}分钟")

            # 集成模型预测
            if 'ensemble' in self.models:
                print("\n执行集成模型预测...")
                ensemble_pred = self.models['ensemble'].predict(
                    (seq_array, static_array),
                    xgb_features,
                    self.device
                )[0, 0]
                predictions['ensemble'] = float(ensemble_pred)
                print(f"  集成模型预测结果: {ensemble_pred:.2f}分钟")

            # 添加不确定性估计
            if 'lstm' in predictions and 'xgboost' in predictions:
                predictions['uncertainty'] = float(abs(predictions['lstm'] - predictions['xgboost']))
                print(f"  模型不确定性: {predictions['uncertainty']:.2f}")

            # 添加延误等级
            if 'ensemble' in predictions:
                delay = predictions['ensemble']
                if delay <= -15:
                    predictions['delay_level'] = 'early'
                    predictions['delay_description'] = '提前到达'
                elif delay <= 15:
                    predictions['delay_level'] = 'on_time'
                    predictions['delay_description'] = '准点'
                elif delay <= 60:
                    predictions['delay_level'] = 'moderate'
                    predictions['delay_description'] = '中等延误'
                else:
                    predictions['delay_level'] = 'severe'
                    predictions['delay_description'] = '严重延误'

                print(f"\n延误等级: {predictions['delay_description']}")

            return {
                'success': True,
                'predictions': predictions,
                'input_data': flight_data
            }

        except Exception as e:
            print(f"\n❌ 预测失败: {e}")
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }

    def predict_batch(self, flights_data):
        """批量预测多条航班数据"""

        results = []
        for i, flight in enumerate(flights_data):
            print(f"\n处理第 {i + 1}/{len(flights_data)} 条数据")
            result = self.predict(flight)
            results.append(result)

        return {
            'success': True,
            'total': len(results),
            'results': results
        }


def load_test_data(file_path='test_flight_data.csv'):
    """加载测试数据"""
    try:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            print(f"加载测试数据: {file_path}, 形状: {df.shape}")
            return df.to_dict('records')
        else:
            print(f"测试文件不存在: {file_path}")
            return None
    except Exception as e:
        print(f"加载测试数据失败: {e}")
        return None


def main():
    """主函数"""

    print("\n" + "=" * 60)
    print("航班延误预测系统启动")
    print("=" * 60)

    # 初始化预测器
    predictor = FlightDelayPredictor(model_dir='../saved_models')

    # 测试单条预测
    print("\n" + "=" * 60)
    print("测试单条预测")
    print("=" * 60)

    test_flight = {
        'flight_date': '2024-01-15',
        'airline_code': 'AA',
        'flight_number': '1234',
        'origin': 'JFK',
        'destination': 'LAX',
        'scheduled_departure_time': 830,
        'scheduled_arrival_time': 1130,
        'distance': 2475
    }

    result = predictor.predict(test_flight)

    if result['success']:
        print("\n✅ 预测成功!")
        print(f"预测结果: {result['predictions']}")
    else:
        print(f"\n❌ 预测失败: {result['error']}")

    # 测试批量预测
    print("\n" + "=" * 60)
    print("测试批量预测")
    print("=" * 60)

    # 从文件加载测试数据
    test_data = load_test_data('test_flight_data.csv')

    if test_data:
        batch_result = predictor.predict_batch(test_data[:2])  # 只预测前2条

        if batch_result['success']:
            print(f"\n✅ 批量预测成功! 共 {batch_result['total']} 条")
            for i, res in enumerate(batch_result['results']):
                if res['success']:
                    print(f"\n第 {i + 1} 条: {res['predictions']}")
                else:
                    print(f"\n第 {i + 1} 条: 失败 - {res['error']}")
    else:
        print("跳过批量预测（无测试数据）")


if __name__ == '__main__':
    main()