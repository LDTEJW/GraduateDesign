import os
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import xgboost as xgb
import holidays
import warnings
import traceback

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

PRED_DELAY_CLIP_MIN = -180.0
PRED_DELAY_CLIP_MAX = 2880.0  # 48 小时


class HybridLSTM(nn.Module):
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
        lstm_out, _ = self.lstm(x_seq)
        lstm_last = lstm_out[:, -1, :]
        if self.static_encoder is not None and x_static is not None:
            static_encoded = self.static_encoder(x_static)
            combined = torch.cat([lstm_last, static_encoded], dim=1)
        else:
            combined = lstm_last
        fused = self.fusion(combined)
        output = self.output_layer(fused)
        return output


def _scaler_feature_names(scaler):
    """兼容 sklearn 1.0+ 的 feature_names_in_，缺失时抛出明确错误。"""
    names = getattr(scaler, 'feature_names_in_', None)
    if names is not None:
        return list(names)
    raise RuntimeError(
        '标准化器缺少 feature_names_in_，请使用与当前训练流程一致的 scikit-learn 版本重新保存 scalers.pkl'
    )


class EnsembleModel:
    def __init__(self, lstm_model, xgb_model, meta_model_type='ridge'):
        self.lstm_model = lstm_model
        self.xgb_model = xgb_model
        self.meta_model_type = meta_model_type
        self.meta_model = None


class EnsembleModelWrapper:
    def __init__(self, lstm_model, xgb_model, meta_model):
        self.lstm_model = lstm_model
        self.xgb_model = xgb_model
        self.meta_model = meta_model
        self.device = next(lstm_model.parameters()).device

    def predict(self, X_lstm, X_xgb, batch_size=1024):
        lstm_pred = self._predict_lstm_batch(X_lstm, batch_size)
        xgb_pred = self._predict_xgb(X_xgb)
        meta_features = np.hstack([
            lstm_pred, xgb_pred,
            lstm_pred - xgb_pred,
            (lstm_pred + xgb_pred) / 2,
            np.abs(lstm_pred - xgb_pred)
        ])
        return self.meta_model.predict(meta_features).reshape(-1, 1)

    def _predict_lstm_batch(self, X_lstm, batch_size):
        self.lstm_model.eval()
        if isinstance(X_lstm, tuple):
            X_seq, X_static = X_lstm
        else:
            X_seq, X_static = X_lstm, None
        n_samples = X_seq.shape[0]
        predictions = []
        for i in range(0, n_samples, batch_size):
            end = min(i + batch_size, n_samples)
            with torch.no_grad():
                # 安全转换 seq
                seq_slice = X_seq[i:end]
                if seq_slice.dtype != np.float32:
                    seq_slice = seq_slice.astype(np.float32)
                X_seq_batch = torch.FloatTensor(np.ascontiguousarray(seq_slice)).to(self.device)
                if X_static is not None:
                    static_slice = X_static[i:end]
                    if static_slice.dtype != np.float32:
                        static_slice = static_slice.astype(np.float32)
                    X_static_batch = torch.FloatTensor(np.ascontiguousarray(static_slice)).to(self.device)
                    pred = self.lstm_model(X_seq_batch, X_static_batch).cpu().numpy()
                else:
                    pred = self.lstm_model(X_seq_batch, None).cpu().numpy()
            predictions.append(pred)
        return np.vstack(predictions)

    def _predict_xgb(self, X_xgb):
        dmatrix = xgb.DMatrix(X_xgb)
        return self.xgb_model.predict(dmatrix).reshape(-1, 1)


class FlightDelayPredictor:
    def __init__(self, model_dir='../train/saved_models'):
        self.model_dir = model_dir
        self.device = device
        self.models = {}
        self.scalers = {}
        self.statistics = {}
        self.label_encoders = {}
        self.seq_length = 10
        self.us_holidays = holidays.US()
        self.global_mean_delay = 0
        self.feature_info = None
        self._load_models()

    @staticmethod
    def _encode_label(le, value):
        try:
            return le.transform([str(value)])[0]
        except ValueError:
            classes = getattr(le, 'classes_', None)
            if classes is not None and 'Unknown' in classes:
                return le.transform(['Unknown'])[0]
            return -1

    def _load_models(self):
        print("加载模型和配置文件...")
        try:
            pkl_path = os.path.join(self.model_dir, 'feature_info.pkl')
            if os.path.exists(pkl_path):
                self.feature_info = joblib.load(pkl_path)
                print("✓ 从PKL加载特征信息成功")

            scalers_path = os.path.join(self.model_dir, 'scalers.pkl')
            if os.path.exists(scalers_path):
                self.scalers = joblib.load(scalers_path)
                print("✓ 标准化器加载成功")

            stats_path = os.path.join(self.model_dir, 'statistics.pkl')
            if os.path.exists(stats_path):
                self.statistics = joblib.load(stats_path)
                print("✓ 统计映射加载成功")
                if 'global_mean_delay' in self.statistics:
                    self.global_mean_delay = self.statistics['global_mean_delay']
                else:
                    all_means = []
                    for key in self.statistics:
                        if key.endswith('_mean_delay'):
                            all_means.extend(self.statistics[key].values())
                    if all_means:
                        self.global_mean_delay = np.mean(all_means)
                    else:
                        self.global_mean_delay = 0

            encoders_path = os.path.join(self.model_dir, 'label_encoders.pkl')
            if os.path.exists(encoders_path):
                self.label_encoders = joblib.load(encoders_path)
                print("✓ 标签编码器加载成功")

            lstm_path = os.path.join(self.model_dir, 'hybrid_lstm_model.pth')
            if os.path.exists(lstm_path):
                try:
                    checkpoint = torch.load(
                        lstm_path, map_location=self.device, weights_only=False
                    )
                except TypeError:
                    checkpoint = torch.load(lstm_path, map_location=self.device)
                seq_input_size = checkpoint.get('seq_input_size', 0)
                static_input_size = checkpoint.get('static_input_size', 0)
                self.seq_length = checkpoint.get('seq_length', 10)
                self.models['lstm'] = HybridLSTM(
                    seq_input_size=seq_input_size,
                    static_input_size=static_input_size,
                    lstm_hidden_size=64,
                    num_lstm_layers=1,
                    dropout=0.2
                ).to(self.device)
                self.models['lstm'].load_state_dict(checkpoint['model_state_dict'])
                self.models['lstm'].eval()
                print(f"✓ LSTM模型加载成功 (seq_len={self.seq_length})")

            xgb_path = os.path.join(self.model_dir, 'optimized_xgboost_model.json')
            if os.path.exists(xgb_path):
                self.models['xgb'] = xgb.Booster()
                self.models['xgb'].load_model(xgb_path)
                print("✓ XGBoost模型加载成功")

            ensemble_path = os.path.join(self.model_dir, 'ensemble_model.pkl')
            if os.path.exists(ensemble_path) and 'lstm' in self.models and 'xgb' in self.models:
                meta_model_obj = joblib.load(ensemble_path)
                meta_model = meta_model_obj.meta_model if hasattr(meta_model_obj, 'meta_model') else meta_model_obj
                self.models['ensemble'] = EnsembleModelWrapper(
                    self.models['lstm'],
                    self.models['xgb'],
                    meta_model
                )
                print("✓ 集成模型加载成功")
            elif os.path.exists(ensemble_path):
                print("存在 ensemble_model.pkl 但 LSTM/XGB 未全部加载，已跳过集成模型")

        except Exception as e:
            print(f"加载模型失败: {e}")
            traceback.print_exc()

    def _fill_statistics(self, df):
        for col in ['origin', 'destination']:
            if col not in df.columns:
                continue
            mean_map = self.statistics.get(f'{col}_mean_delay', {})
            std_map = self.statistics.get(f'{col}_std_delay', {})
            count_map = self.statistics.get(f'{col}_flight_count', {})
            df[f'{col}_mean_delay'] = df[col].map(mean_map).fillna(self.global_mean_delay)
            df[f'{col}_std_delay'] = df[col].map(std_map).fillna(0)
            df[f'{col}_flight_count'] = df[col].map(count_map).fillna(100)

        if 'airline_code' in df.columns:
            mean_map = self.statistics.get('airline_mean_delay', {})
            std_map = self.statistics.get('airline_std_delay', {})
            count_map = self.statistics.get('airline_flight_count', {})
            dep_mean_map = self.statistics.get('airline_dep_delay_mean', {})
            df['airline_mean_delay'] = df['airline_code'].map(mean_map).fillna(self.global_mean_delay)
            df['airline_std_delay'] = df['airline_code'].map(std_map).fillna(0)
            df['airline_flight_count'] = df['airline_code'].map(count_map).fillna(100)
            df['airline_dep_delay_mean'] = df['airline_code'].map(dep_mean_map).fillna(0)

        if 'route' in df.columns:
            mean_map = self.statistics.get('route_mean_delay', {})
            std_map = self.statistics.get('route_std_delay', {})
            count_map = self.statistics.get('route_flight_count', {})
            dist_map = self.statistics.get('route_mean_distance', {})
            df['route_mean_delay'] = df['route'].map(mean_map).fillna(self.global_mean_delay)
            df['route_std_delay'] = df['route'].map(std_map).fillna(0)
            df['route_flight_count'] = df['route'].map(count_map).fillna(50)
            df['route_mean_distance'] = df['route'].map(dist_map).fillna(df['distance'] if 'distance' in df else 1000)

        if 'estimated_speed' not in df.columns and 'distance' in df.columns and 'air_time' in df.columns:
            air_time_hours = df['air_time'].replace(0, 0.001) / 60
            df['estimated_speed'] = df['distance'] / air_time_hours
        else:
            df['estimated_speed'] = 0

        return df

    def preprocess_dataframe(self, df):
        df = df.copy()
        rename_map = {
            'FL_DATE': 'flight_date', 'AIRLINE_CODE': 'airline_code',
            'FL_NUMBER': 'flight_number', 'ORIGIN': 'origin', 'DEST': 'destination',
            'CRS_DEP_TIME': 'scheduled_departure_time', 'DEP_TIME': 'actual_departure_time',
            'DEP_DELAY': 'departure_delay', 'TAXI_OUT': 'taxi_out', 'TAXI_IN': 'taxi_in',
            'CRS_ARR_TIME': 'scheduled_arrival_time', 'ARR_TIME': 'actual_arrival_time',
            'ARR_DELAY': 'arrival_delay', 'CANCELLED': 'cancelled', 'DIVERTED': 'diverted',
            'CRS_ELAPSED_TIME': 'scheduled_elapsed_time', 'ELAPSED_TIME': 'actual_elapsed_time',
            'AIR_TIME': 'air_time', 'DISTANCE': 'distance',
            'DELAY_DUE_CARRIER': 'carrier_delay', 'DELAY_DUE_WEATHER': 'weather_delay',
            'DELAY_DUE_NAS': 'nas_delay', 'DELAY_DUE_SECURITY': 'security_delay',
            'DELAY_DUE_LATE_AIRCRAFT': 'late_aircraft_delay'
        }
        df.rename(columns={k:v for k,v in rename_map.items() if k in df.columns}, inplace=True)
        if 'flight_date' in df.columns:
            df['flight_date'] = pd.to_datetime(df['flight_date'], errors='coerce')
        if 'cancelled' in df.columns:
            df = df[df['cancelled'] == 0]
        if 'diverted' in df.columns:
            df = df[df['diverted'] == 0]
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
        for col in ['scheduled_departure_time', 'actual_departure_time', 'scheduled_arrival_time', 'actual_arrival_time']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                df[f'{col}_hour'] = (df[col] // 100).astype(int) % 24
                df[f'{col}_minute'] = (df[col] % 100).astype(int)
                period = pd.cut(df[f'{col}_hour'], bins=[-1, 6, 12, 18, 24], labels=[0, 1, 2, 3])
                df[f'{col}_period'] = period.astype(float)
        if 'departure_delay' in df.columns:
            df['departure_delay'] = pd.to_numeric(df['departure_delay'], errors='coerce').fillna(0)
            df['is_delayed_departure'] = (df['departure_delay'] > 0).astype(int)
            df['departure_delay_abs'] = df['departure_delay'].abs()
            df['departure_delay_category'] = pd.cut(df['departure_delay'],
                bins=[-float('inf'), -30, -15, 0, 15, 30, 60, float('inf')],
                labels=[0, 1, 2, 3, 4, 5, 6]).astype(float)
        else:
            df['departure_delay'] = 0
            df['is_delayed_departure'] = 0
            df['departure_delay_abs'] = 0
            df['departure_delay_category'] = 3
        if 'distance' in df.columns:
            df['distance'] = pd.to_numeric(df['distance'], errors='coerce').fillna(0)
            df['distance_category'] = pd.cut(df['distance'],
                bins=[0, 500, 1000, 1500, 2500, float('inf')],
                labels=[0, 1, 2, 3, 4]).astype(float)
        if 'origin' in df.columns and 'destination' in df.columns:
            df['origin'] = df['origin'].fillna('UNKNOWN')
            df['destination'] = df['destination'].fillna('UNKNOWN')
            df['route'] = df['origin'] + '_' + df['destination']
        for col in ['origin', 'destination', 'airline_code', 'route']:
            if col in df.columns and col in self.label_encoders:
                le = self.label_encoders[col]
                df[f'{col}_encoded'] = df[col].apply(lambda x, _le=le: self._encode_label(_le, x))
            elif col in df.columns:
                df[f'{col}_encoded'] = -1
        if 'month' in df.columns and 'day_of_week' in df.columns:
            df['month_day_interaction'] = df['month'] * 10 + df['day_of_week']
        df = self._fill_statistics(df)
        if not self.scalers or 'seq_scaler' not in self.scalers:
            raise RuntimeError(
                f'未找到 seq_scaler，请确认 {self.model_dir} 中存在有效的 scalers.pkl'
            )
        seq_scaler = self.scalers['seq_scaler']
        static_scaler = self.scalers.get('static_scaler')
        seq_features = _scaler_feature_names(seq_scaler)
        static_features = _scaler_feature_names(static_scaler) if static_scaler is not None else []
        required_features = seq_features + static_features
        for feat in required_features:
            if feat not in df.columns:
                df[feat] = 0
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        return df

    def extract_xgb_features(self, seq_array, static_array):
        """与 main.py 中 HybridModelTrainer._extract_xgb_features（单样本）一致。"""
        n_features = seq_array.shape[2]
        features = []
        for i in range(n_features):
            data = seq_array[0, :, i]
            features.extend([
                np.mean(data), np.std(data), np.max(data), np.min(data),
                data[-1], data[-1] - data[0],
            ])
        # 训练时对 X_static 的每一列都拼入，包含仅 1 个静态特征的情况
        if static_array is not None and static_array.ndim >= 2 and static_array.shape[1] > 0:
            features.extend(static_array[0].astype(np.float64).ravel().tolist())
        return np.array(features, dtype=np.float64).reshape(1, -1)

    def create_sequence_from_row(self, row_df):
        """
        单行预测：将当前行时序特征标准化后沿时间维重复 seq_length 次。
        用于无足够历史窗口时的回退。
        """
        seq_scaler = self.scalers['seq_scaler']
        seq_features = _scaler_feature_names(seq_scaler)
        seq_vals = [
            float(row_df[feat].iloc[0]) if feat in row_df.columns else 0.0
            for feat in seq_features
        ]
        seq_row = np.array(seq_vals, dtype=np.float64).reshape(1, -1)
        seq_scaled = seq_scaler.transform(seq_row).astype(np.float32).ravel()
        seq_array = np.tile(seq_scaled, (self.seq_length, 1)).reshape(1, self.seq_length, -1)

        static_scaler = self.scalers.get('static_scaler')
        if static_scaler is not None:
            static_features = _scaler_feature_names(static_scaler)
            static_vals = [
                float(row_df[feat].iloc[0]) if feat in row_df.columns else 0.0
                for feat in static_features
            ]
            static_row = np.array(static_vals, dtype=np.float64).reshape(1, -1)
            static_array = static_scaler.transform(static_row).astype(np.float32)
        else:
            static_array = None
        return seq_array, static_array

    def _predict_one(self, seq_array, static_array):
        xgb_static = static_array if static_array is not None else np.zeros((1, 1), dtype=np.float32)
        xgb_features = self.extract_xgb_features(seq_array, xgb_static)
        if 'ensemble' in self.models:
            lstm_in = (seq_array, static_array) if static_array is not None else (seq_array, None)
            pred = self.models['ensemble'].predict(lstm_in, xgb_features).flatten()[0]
        elif 'xgb' in self.models:
            pred = self.models['xgb'].predict(xgb.DMatrix(xgb_features))[0]
        elif 'lstm' in self.models:
            with torch.no_grad():
                seq_tensor = torch.FloatTensor(seq_array).to(self.device)
                if static_array is not None:
                    st = torch.FloatTensor(static_array).to(self.device)
                    pred = self.models['lstm'](seq_tensor, st).cpu().numpy()[0, 0]
                else:
                    pred = self.models['lstm'](seq_tensor, None).cpu().numpy()[0, 0]
        else:
            raise ValueError('没有可用的预测模型')
        p = float(pred)
        if not np.isfinite(p):
            p = float(self.global_mean_delay)
        return float(np.clip(p, PRED_DELAY_CLIP_MIN, PRED_DELAY_CLIP_MAX))

    def predict_dataframe(self, df):
        processed_df = self.preprocess_dataframe(df)
        if 'arrival_delay' not in processed_df.columns:
            processed_df['arrival_delay'] = 0
        if 'flight_number' not in processed_df.columns:
            processed_df['flight_number'] = '__default__'
        if 'flight_date' in processed_df.columns:
            processed_df['flight_date'] = pd.to_datetime(processed_df['flight_date'])

        sort_keys = ['flight_number', 'flight_date']
        if 'scheduled_departure_time_hour' in processed_df.columns:
            sort_keys.append('scheduled_departure_time_hour')
        processed_df = processed_df.sort_values(sort_keys)

        def add_lag_features(group):
            group = group.copy()
            for lag in [1, 2, 3, 7]:
                group[f'arrival_delay_lag_{lag}'] = group['arrival_delay'].shift(lag)
                group[f'departure_delay_lag_{lag}'] = group['departure_delay'].shift(lag)
            for window in [3, 7, 14]:
                group[f'arrival_delay_ma_{window}'] = (
                    group['arrival_delay'].rolling(window, min_periods=1).mean()
                )
                group[f'departure_delay_ma_{window}'] = (
                    group['departure_delay'].rolling(window, min_periods=1).mean()
                )
            return group

        processed_df = processed_df.groupby('flight_number', group_keys=False).apply(add_lag_features)
        lag_cols = [c for c in processed_df.columns if 'lag' in c or 'ma_' in c]
        if lag_cols:
            processed_df[lag_cols] = processed_df[lag_cols].fillna(0)

        # 未标准化特征，供滑动窗口与逐行回退共用（与训练：先特征再按行标准化一致）
        df_before_scale = processed_df.copy()

        seq_scaler = self.scalers['seq_scaler']
        seq_features = _scaler_feature_names(seq_scaler)
        for feat in seq_features:
            if feat not in processed_df.columns:
                processed_df[feat] = 0
        seq_data = processed_df[seq_features].values
        processed_df[seq_features] = seq_scaler.transform(seq_data)

        static_features = []
        static_scaler = self.scalers.get('static_scaler')
        if static_scaler is not None:
            static_features = _scaler_feature_names(static_scaler)
            for feat in static_features:
                if feat not in processed_df.columns:
                    processed_df[feat] = 0
            static_data = processed_df[static_features].values
            processed_df[static_features] = static_scaler.transform(static_data)

        predictions_series = pd.Series(index=processed_df.index, dtype=float)

        for flight_num, group in processed_df.groupby('flight_number'):
            group = group.reset_index(drop=False)
            if len(group) < self.seq_length + 1:
                continue

            for i in range(len(group) - self.seq_length):
                # 训练时：X_seq = 行 [i, i+seq_length)；静态与标签对齐在目标行 i+seq_length
                window = group.iloc[i:i + self.seq_length]
                target = group.iloc[i + self.seq_length]
                target_idx = target['index']

                seq_array = window[seq_features].values.astype(np.float32).reshape(1, self.seq_length, -1)

                if static_features:
                    static_vals = target[static_features].values
                    static_vals = np.nan_to_num(static_vals, nan=0.0, posinf=0.0, neginf=0.0)
                    static_array = static_vals.astype(np.float32).reshape(1, -1)
                else:
                    static_array = None

                try:
                    predictions_series[target_idx] = self._predict_one(seq_array, static_array)
                except Exception as e:
                    logger.warning('滑动窗口预测失败 idx=%s: %s', target_idx, e)

        # 无足够历史：逐行平铺序列（与 train/predict.py 一致），避免大量 NaN 被当作 0 参与评估
        missing_idx = predictions_series.index[predictions_series.isna()]
        for idx in missing_idx:
            try:
                row_df = df_before_scale.loc[[idx]]
                seq_array, static_array = self.create_sequence_from_row(row_df)
                predictions_series[idx] = self._predict_one(seq_array, static_array)
            except Exception as e:
                logger.warning('逐行回退预测失败 idx=%s: %s', idx, e)

        predictions_series = predictions_series.fillna(self.global_mean_delay)
        result = predictions_series.reindex(df.index)
        return result.values.astype(np.float64)