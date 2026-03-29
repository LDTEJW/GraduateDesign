import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
from scipy import stats
from datetime import datetime, timedelta
import holidays

# 设置
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


class EnhancedFlightDataProcessor:
	"""增强的航班数据处理类"""

	def __init__(self, file_path, sample_size=None):
		self.file_path = file_path
		self.sample_size = sample_size
		self.df = None
		self.us_holidays = holidays.US()
		self.label_encoders = {}  # 存储标签编码器

	def load_data(self):
		"""加载数据"""
		print(f"加载航班数据...")

		try:
			if self.sample_size:
				self.df = pd.read_csv(self.file_path, nrows=self.sample_size, low_memory=False)
			else:
				self.df = pd.read_csv(self.file_path, low_memory=False)
		except Exception as e:
			print(f"读取文件错误: {e}")
			try:
				self.df = pd.read_csv(self.file_path, encoding='utf-8', low_memory=False,
				                      nrows=self.sample_size if self.sample_size else None)
			except:
				self.df = pd.read_csv(self.file_path, encoding='latin-1', low_memory=False,
				                      nrows=self.sample_size if self.sample_size else None)

		print(f"原始数据形状: {self.df.shape}")
		return self.df

	def clean_data(self):
		"""数据清洗"""
		print("\n开始数据清洗...")

		# 重命名列
		column_mapping = {
			'FL_DATE': 'flight_date',
			'AIRLINE_CODE': 'airline_code',
			'FL_NUMBER': 'flight_number',
			'ORIGIN': 'origin',
			'DEST': 'destination',
			'CRS_DEP_TIME': 'scheduled_departure_time',
			'DEP_TIME': 'actual_departure_time',
			'DEP_DELAY': 'departure_delay',
			'TAXI_OUT': 'taxi_out',
			'TAXI_IN': 'taxi_in',
			'CRS_ARR_TIME': 'scheduled_arrival_time',
			'ARR_TIME': 'actual_arrival_time',
			'ARR_DELAY': 'arrival_delay',
			'CANCELLED': 'cancelled',
			'DIVERTED': 'diverted',
			'CRS_ELAPSED_TIME': 'scheduled_elapsed_time',
			'ELAPSED_TIME': 'actual_elapsed_time',
			'AIR_TIME': 'air_time',
			'DISTANCE': 'distance',
			'DELAY_DUE_CARRIER': 'carrier_delay',
			'DELAY_DUE_WEATHER': 'weather_delay',
			'DELAY_DUE_NAS': 'nas_delay',
			'DELAY_DUE_SECURITY': 'security_delay',
			'DELAY_DUE_LATE_AIRCRAFT': 'late_aircraft_delay'
		}

		# 只重命名存在的列
		rename_dict = {old: new for old, new in column_mapping.items() if old in self.df.columns}
		self.df = self.df.rename(columns=rename_dict)

		# 移除取消和改航的航班
		if 'cancelled' in self.df.columns:
			self.df = self.df[self.df['cancelled'] == 0]
		if 'diverted' in self.df.columns:
			self.df = self.df[self.df['diverted'] == 0]

		# 转换日期时间
		if 'flight_date' in self.df.columns:
			self.df['flight_date'] = pd.to_datetime(self.df['flight_date'], errors='coerce')

		# 确保目标变量存在
		if 'arrival_delay' not in self.df.columns:
			raise ValueError("未找到到达延误列")

		# 移除异常值（使用IQR方法）
		Q1 = self.df['arrival_delay'].quantile(0.01)
		Q3 = self.df['arrival_delay'].quantile(0.99)
		IQR = Q3 - Q1
		lower_bound = Q1 - 3 * IQR
		upper_bound = Q3 + 3 * IQR
		self.df = self.df[(self.df['arrival_delay'] >= lower_bound) &
		                  (self.df['arrival_delay'] <= upper_bound)]

		print(f"清洗后数据形状: {self.df.shape}")
		return self.df

	def create_features(self):
		"""创建特征工程"""
		print("\n创建特征工程...")

		# 时间特征
		if 'flight_date' in self.df.columns:
			self.df['year'] = self.df['flight_date'].dt.year
			self.df['month'] = self.df['flight_date'].dt.month
			self.df['day'] = self.df['flight_date'].dt.day
			self.df['day_of_week'] = self.df['flight_date'].dt.dayofweek
			self.df['day_of_year'] = self.df['flight_date'].dt.dayofyear
			self.df['week_of_year'] = self.df['flight_date'].dt.isocalendar().week
			self.df['quarter'] = self.df['flight_date'].dt.quarter
			self.df['is_weekend'] = (self.df['day_of_week'] >= 5).astype(int)

			# 节假日特征
			self.df['is_holiday'] = self.df['flight_date'].apply(
				lambda x: 1 if x in self.us_holidays else 0
			)

			# 季节性特征 - 使用数值编码
			self.df['season'] = self.df['month'] % 12 // 3 + 1

		# 时间特征（小时分钟）
		time_cols = ['scheduled_departure_time', 'actual_departure_time',
		             'scheduled_arrival_time', 'actual_arrival_time']

		for col in time_cols:
			if col in self.df.columns:
				hour_col = f"{col}_hour"
				minute_col = f"{col}_minute"

				# 处理缺失值
				self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
				self.df[col] = self.df[col].fillna(0)

				# 提取小时和分钟
				self.df[hour_col] = (self.df[col] // 100).astype(int) % 24
				self.df[minute_col] = (self.df[col] % 100).astype(int)

				# 时间段（0:深夜, 1:早晨, 2:下午, 3:晚上）
				period_col = f"{col}_period"
				self.df[period_col] = pd.cut(self.df[hour_col],
				                             bins=[-1, 6, 12, 18, 24],
				                             labels=[0, 1, 2, 3])
				# 转换为数值
				self.df[period_col] = self.df[period_col].astype(float)

		# 延误相关特征
		if 'departure_delay' in self.df.columns:
			self.df['is_delayed_departure'] = (self.df['departure_delay'] > 0).astype(int)
			self.df['departure_delay_abs'] = self.df['departure_delay'].abs()

			# 起飞延误类别 - 使用数值编码
			self.df['departure_delay_category'] = pd.cut(
				self.df['departure_delay'],
				bins=[-float('inf'), -30, -15, 0, 15, 30, 60, float('inf')],
				labels=[0, 1, 2, 3, 4, 5, 6]
			)
			self.df['departure_delay_category'] = self.df['departure_delay_category'].astype(float)

		# 距离特征
		if 'distance' in self.df.columns:
			# 距离类别 - 使用数值编码
			self.df['distance_category'] = pd.cut(
				self.df['distance'],
				bins=[0, 500, 1000, 1500, 2500, float('inf')],
				labels=[0, 1, 2, 3, 4]
			)
			self.df['distance_category'] = self.df['distance_category'].astype(float)

			# 计算估计速度
			if 'air_time' in self.df.columns:
				# 避免除零
				air_time_hours = self.df['air_time'] / 60
				air_time_hours = air_time_hours.replace(0, 0.001)  # 避免除零
				self.df['estimated_speed'] = self.df['distance'] / air_time_hours

		# 机场相关特征（目标编码）
		airport_cols = ['origin', 'destination']
		for col in airport_cols:
			if col in self.df.columns:
				# 计算机场平均延误
				airport_stats = self.df.groupby(col).agg({
					'arrival_delay': ['mean', 'std', 'count']
				}).round(2)

				# 处理多层列名
				airport_stats.columns = [f'{col}_{stat}' for stat in ['mean_delay', 'std_delay', 'flight_count']]

				# 合并回主数据
				self.df = self.df.merge(
					airport_stats,
					left_on=col,
					right_index=True,
					how='left'
				)

				# 对机场代码进行标签编码
				if self.df[col].dtype == 'object':
					le = LabelEncoder()
					self.df[f'{col}_encoded'] = le.fit_transform(self.df[col].fillna('Unknown'))
					self.label_encoders[col] = le

		# 航空公司特征
		if 'airline_code' in self.df.columns:
			airline_stats = self.df.groupby('airline_code').agg({
				'arrival_delay': ['mean', 'std', 'count'],
				'departure_delay': 'mean'
			}).round(2)

			# 处理列名
			airline_stats.columns = ['airline_mean_delay', 'airline_std_delay',
			                         'airline_flight_count', 'airline_dep_delay_mean']

			self.df = self.df.merge(
				airline_stats,
				left_on='airline_code',
				right_index=True,
				how='left'
			)

			# 对航空公司代码进行标签编码
			if self.df['airline_code'].dtype == 'object':
				le = LabelEncoder()
				self.df['airline_code_encoded'] = le.fit_transform(self.df['airline_code'].fillna('Unknown'))
				self.label_encoders['airline_code'] = le

		# 路线特征
		if all(col in self.df.columns for col in ['origin', 'destination']):
			self.df['route'] = self.df['origin'] + '_' + self.df['destination']
			route_stats = self.df.groupby('route').agg({
				'arrival_delay': ['mean', 'std', 'count'],
				'distance': 'mean'
			}).round(2)

			route_stats.columns = ['route_mean_delay', 'route_std_delay',
			                       'route_flight_count', 'route_mean_distance']

			self.df = self.df.merge(
				route_stats,
				left_on='route',
				right_index=True,
				how='left'
			)

			# 对路线进行标签编码
			if 'route' in self.df.columns and self.df['route'].dtype == 'object':
				le = LabelEncoder()
				self.df['route_encoded'] = le.fit_transform(self.df['route'].fillna('Unknown'))
				self.label_encoders['route'] = le

		# 时间交互特征
		if all(col in self.df.columns for col in ['month', 'day_of_week']):
			self.df['month_day_interaction'] = self.df['month'] * 10 + self.df['day_of_week']

		# 滞后特征（为时序模型准备）
		self._create_lag_features()

		# 处理缺失值
		self._handle_missing_values()

		print(f"特征工程后数据形状: {self.df.shape}")
		print(f"特征数量: {len(self.df.columns)}")

		return self.df

	def _create_lag_features(self):
		"""创建滞后特征"""
		print("创建滞后特征...")

		# 按航班号和日期排序
		if all(col in self.df.columns for col in ['flight_number', 'flight_date', 'scheduled_departure_time_hour']):
			self.df = self.df.sort_values(['flight_number', 'flight_date', 'scheduled_departure_time_hour'])

			# 创建滞后特征
			lag_cols = ['arrival_delay', 'departure_delay']
			for col in lag_cols:
				if col in self.df.columns:
					for lag in [1, 2, 3, 7]:
						self.df[f'{col}_lag_{lag}'] = self.df.groupby('flight_number')[col].shift(lag)

					# 移动平均
					for window in [3, 7, 14]:
						self.df[f'{col}_ma_{window}'] = self.df.groupby('flight_number')[col].transform(
							lambda x: x.rolling(window, min_periods=1).mean()
						)

		return self.df

	def _handle_missing_values(self):
		"""处理缺失值"""
		print("处理缺失值...")

		# 对于数值列，用中位数填充
		numeric_cols = self.df.select_dtypes(include=[np.number]).columns
		for col in numeric_cols:
			if col != 'arrival_delay':  # 目标变量不填充
				missing_count = self.df[col].isna().sum()
				if missing_count > 0:
					median_val = self.df[col].median()
					self.df[col] = self.df[col].fillna(median_val)

		# 对于类别列，用众数填充
		categorical_cols = self.df.select_dtypes(include=['object']).columns
		for col in categorical_cols:
			missing_count = self.df[col].isna().sum()
			if missing_count > 0:
				mode_val = self.df[col].mode()[0] if not self.df[col].mode().empty else 'Unknown'
				self.df[col] = self.df[col].fillna(mode_val)

	def get_numeric_features(self):
		"""获取数值特征"""
		numeric_features = []

		for col in self.df.columns:
			if col == 'arrival_delay' or col == 'flight_date':
				continue

			# 只选择数值列
			if self.df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
				# 检查是否有太多缺失值
				if self.df[col].isna().sum() / len(self.df) < 0.5:  # 少于50%缺失值
					numeric_features.append(col)

		return numeric_features

	def save_feature_info(self, save_path='../saved_models/feature_info.pkl'):
		"""保存特征信息"""
		feature_info = {
			'label_encoders': self.label_encoders,
			'feature_columns': list(self.df.columns),
			'numeric_features': self.get_numeric_features(),
			'total_features': len(self.df.columns)
		}

		joblib.dump(feature_info, save_path)
		print(f"特征信息已保存到: {save_path}")
		print(f"特征总数: {len(self.df.columns)}")
		print(f"数值特征数量: {len(self.get_numeric_features())}")
		return feature_info


class HybridLSTM(nn.Module):
	"""混合LSTM模型"""

	def __init__(self, seq_input_size, static_input_size=0,
	             lstm_hidden_size=64, num_lstm_layers=1,
	             dropout=0.2):
		super(HybridLSTM, self).__init__()

		# LSTM层
		self.lstm = nn.LSTM(
			input_size=seq_input_size,
			hidden_size=lstm_hidden_size,
			num_layers=num_lstm_layers,
			batch_first=True,
			dropout=dropout if num_lstm_layers > 1 else 0,
			bidirectional=True
		)

		# 静态特征处理
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

		# LSTM输出处理
		lstm_output_size = lstm_hidden_size * 2

		# 特征融合层
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

		# 输出层
		self.output_layer = nn.Sequential(
			nn.Linear(64, 32),
			nn.ReLU(),
			nn.Dropout(dropout * 0.3),
			nn.Linear(32, 1)
		)

		# 初始化权重
		self._initialize_weights()

	def _initialize_weights(self):
		"""初始化权重 """
		for name, param in self.named_parameters():
			if param.dim() < 2:
				# 对偏置项使用常数初始化
				if 'bias' in name:
					nn.init.constant_(param, 0.1)
				continue

			# LSTM权重使用正交初始化
			if 'lstm' in name and 'weight' in name:
				nn.init.orthogonal_(param)
			# 其他二维权重使用Kaiming初始化
			elif 'weight' in name and param.dim() >= 2:
				nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='relu')
			# 偏置项使用常数初始化
			elif 'bias' in name:
				nn.init.constant_(param, 0.1)

	def forward(self, x_seq, x_static=None):
		# LSTM处理
		lstm_out, (h_n, c_n) = self.lstm(x_seq)

		# 取最后一个时间步的输出
		lstm_last = lstm_out[:, -1, :]

		# 静态特征处理
		if self.static_encoder is not None and x_static is not None:
			static_encoded = self.static_encoder(x_static)
			combined = torch.cat([lstm_last, static_encoded], dim=1)
		else:
			combined = lstm_last

		# 特征融合
		fused = self.fusion(combined)

		# 输出
		output = self.output_layer(fused)

		return output


class TimeSeriesDataset(Dataset):
	"""时间序列数据集"""

	def __init__(self, X_seq, X_static, y):
		self.X_seq = torch.FloatTensor(X_seq)
		self.X_static = torch.FloatTensor(X_static) if X_static is not None else None
		self.y = torch.FloatTensor(y)

	def __len__(self):
		return len(self.y)

	def __getitem__(self, idx):
		if self.X_static is not None:
			return self.X_seq[idx], self.X_static[idx], self.y[idx]
		else:
			return self.X_seq[idx], self.y[idx]


class XGBoostTrainer:
	"""XGBoost训练器"""

	def __init__(self):
		self.model = None

	def train_with_early_stopping(self, X_train, y_train, X_val, y_val, params):
		"""使用早停训练XGBoost模型"""
		print("使用xgb.train API训练XGBoost...")

		# 创建DMatrix
		dtrain = xgb.DMatrix(X_train, label=y_train)
		dval = xgb.DMatrix(X_val, label=y_val)

		# 设置早停
		evals = [(dtrain, 'train'), (dval, 'val')]

		# 训练模型
		self.model = xgb.train(
			params=params,
			dtrain=dtrain,
			num_boost_round=params.get('n_estimators', 200),
			evals=evals,
			early_stopping_rounds=20,
			verbose_eval=50
		)

		return self.model

	def predict(self, X):
		"""预测"""
		dmatrix = xgb.DMatrix(X)
		return self.model.predict(dmatrix)


class EnsembleModel:
	"""集成模型，结合LSTM和XGBoost"""

	def __init__(self, lstm_model, xgb_model, meta_model_type='ridge'):
		self.lstm_model = lstm_model
		self.xgb_model = xgb_model
		self.meta_model_type = meta_model_type
		self.meta_model = None

	def train_meta_model(self, X_train_lstm, X_train_xgb, y_train):
		"""训练元模型"""

		# 分批获取LSTM预测
		print("分批获取LSTM预测...")
		lstm_pred = self._get_lstm_predictions_batch(X_train_lstm, batch_size=1024)
		xgb_pred = self._get_xgb_predictions(X_train_xgb)

		print(f"LSTM预测形状: {lstm_pred.shape}, XGBoost预测形状: {xgb_pred.shape}")

		# 创建元特征
		meta_features = np.hstack([
			lstm_pred,
			xgb_pred,
			lstm_pred - xgb_pred,  # 差异特征
			(lstm_pred + xgb_pred) / 2,  # 平均特征
			np.abs(lstm_pred - xgb_pred)  # 绝对差异
		])

		print(f"元特征形状: {meta_features.shape}")

		# 训练元模型
		if self.meta_model_type == 'ridge':
			self.meta_model = Ridge(alpha=1.0)
		elif self.meta_model_type == 'lasso':
			self.meta_model = Lasso(alpha=0.1)
		elif self.meta_model_type == 'rf':
			self.meta_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
		elif self.meta_model_type == 'gbm':
			self.meta_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
		else:
			self.meta_model = Ridge(alpha=1.0)

		self.meta_model.fit(meta_features, y_train.ravel())

		return self.meta_model

	def predict(self, X_lstm, X_xgb):
		"""预测"""

		# 分批获取LSTM预测
		lstm_pred = self._get_lstm_predictions_batch(X_lstm, batch_size=1024)
		xgb_pred = self._get_xgb_predictions(X_xgb)

		# 创建元特征
		meta_features = np.hstack([
			lstm_pred,
			xgb_pred,
			lstm_pred - xgb_pred,
			(lstm_pred + xgb_pred) / 2,
			np.abs(lstm_pred - xgb_pred)
		])

		# 元模型预测
		ensemble_pred = self.meta_model.predict(meta_features)

		return ensemble_pred.reshape(-1, 1)

	def get_base_model_predictions(self, X_lstm, X_xgb):
		"""获取基模型预测值和绝对差值"""
		# 分批获取LSTM预测
		lstm_pred = self._get_lstm_predictions_batch(X_lstm, batch_size=1024)
		xgb_pred = self._get_xgb_predictions(X_xgb)

		# 计算绝对差值（不确定性度量）
		abs_diff = np.abs(lstm_pred - xgb_pred)

		return lstm_pred, xgb_pred, abs_diff

	def _get_lstm_predictions_batch(self, X_lstm, batch_size=1024):
		"""分批获取LSTM预测"""
		self.lstm_model.eval()

		if isinstance(X_lstm, tuple):
			X_seq, X_static = X_lstm
			n_samples = X_seq.shape[0]
		else:
			X_seq = X_lstm
			X_static = None
			n_samples = X_seq.shape[0]

		predictions = []

		# 分批处理
		for i in range(0, n_samples, batch_size):
			end_idx = min(i + batch_size, n_samples)

			with torch.no_grad():
				if X_static is not None:
					X_seq_batch = torch.FloatTensor(X_seq[i:end_idx]).to(device)
					X_static_batch = torch.FloatTensor(X_static[i:end_idx]).to(device)
					batch_pred = self.lstm_model(X_seq_batch, X_static_batch).cpu().numpy()
				else:
					X_seq_batch = torch.FloatTensor(X_seq[i:end_idx]).to(device)
					batch_pred = self.lstm_model(X_seq_batch, None).cpu().numpy()

			predictions.append(batch_pred)

			# 释放GPU内存
			if torch.cuda.is_available():
				torch.cuda.empty_cache()

		return np.vstack(predictions)

	def _get_xgb_predictions(self, X_xgb):
		"""获取XGBoost预测"""
		# xgb.Booster需要DMatrix
		dmatrix = xgb.DMatrix(X_xgb)
		return self.xgb_model.predict(dmatrix).reshape(-1, 1)


class HybridModelTrainer:
	"""混合模型训练器"""

	def __init__(self, processor, seq_length=30):
		self.processor = processor
		self.seq_length = seq_length
		self.models = {}
		self.scalers = {}
		self.results = {}
		self.feature_info = {}  # 存储特征信息

	def prepare_data(self, test_size=0.2, val_size=0.1):
		"""准备数据"""
		print("\n准备数据...")

		# 获取数值特征
		numeric_features = self.processor.get_numeric_features()

		# 从数值特征中分离时序特征和静态特征
		# 时序特征：滞后特征、时间相关特征
		seq_features = [f for f in numeric_features if 'lag' in f or 'ma_' in f or
		                f in ['year', 'month', 'day', 'day_of_week', 'day_of_year',
		                      'week_of_year', 'quarter', 'scheduled_departure_time_hour',
		                      'actual_departure_time_hour', 'scheduled_arrival_time_hour',
		                      'actual_arrival_time_hour']]

		# 静态特征：其他数值特征
		static_features = [f for f in numeric_features if f not in seq_features and f != 'arrival_delay']

		# 保存特征名称和数量
		self.feature_info = {
			'seq_features': seq_features,
			'static_features': static_features,
			'seq_features_count': len(seq_features),
			'static_features_count': len(static_features),
			'total_numeric_features': len(numeric_features)
		}

		print("=" * 60)
		print("特征信息统计:")
		print("=" * 60)
		print(f"时序特征数量: {len(seq_features)}")
		print(f"静态特征数量: {len(static_features)}")
		print(f"总数值特征数量: {len(numeric_features)}")
		print("\n时序特征列表:")
		for i, feat in enumerate(seq_features[:20], 1):
			print(f"  {i:3d}. {feat}")
		if len(seq_features) > 20:
			print(f"  ... 还有 {len(seq_features) - 20} 个时序特征")

		print("\n静态特征列表:")
		for i, feat in enumerate(static_features[:20], 1):
			print(f"  {i:3d}. {feat}")
		if len(static_features) > 20:
			print(f"  ... 还有 {len(static_features) - 20} 个静态特征")
		print("=" * 60)

		# 准备数据
		X_seq, X_static, y = self._prepare_sequences(
			seq_features, static_features, self.seq_length
		)

		# 分割数据（保持时序顺序）
		total_size = len(X_seq)
		train_size = int(total_size * (1 - test_size - val_size))
		val_size_actual = int(total_size * val_size)

		indices = np.arange(total_size)
		train_idx = indices[:train_size]
		val_idx = indices[train_size:train_size + val_size_actual]
		test_idx = indices[train_size + val_size_actual:]

		# 分割数据
		X_seq_train, X_static_train, y_train = X_seq[train_idx], X_static[train_idx], y[train_idx]
		X_seq_val, X_static_val, y_val = X_seq[val_idx], X_static[val_idx], y[val_idx]
		X_seq_test, X_static_test, y_test = X_seq[test_idx], X_static[test_idx], y[test_idx]

		print(f"训练集: {len(X_seq_train):,} 样本")
		print(f"验证集: {len(X_seq_val):,} 样本")
		print(f"测试集: {len(X_seq_test):,} 样本")

		# 为XGBoost准备特征
		X_xgb_train = self._extract_xgb_features(X_seq_train, X_static_train)
		X_xgb_val = self._extract_xgb_features(X_seq_val, X_static_val)
		X_xgb_test = self._extract_xgb_features(X_seq_test, X_static_test)

		# 保存特征形状信息
		self.feature_info.update({
			'seq_input_shape': X_seq_train.shape,
			'static_input_shape': X_static_train.shape if X_static_train is not None else None,
			'xgb_input_shape': X_xgb_train.shape,
			'y_shape': y_train.shape
		})

		return {
			'lstm': {
				'train': (X_seq_train, X_static_train, y_train),
				'val': (X_seq_val, X_static_val, y_val),
				'test': (X_seq_test, X_static_test, y_test)
			},
			'xgb': {
				'train': (X_xgb_train, y_train),
				'val': (X_xgb_val, y_val),
				'test': (X_xgb_test, y_test)
			}
		}

	def _prepare_sequences(self, seq_features, static_features, seq_length):
		"""准备时序序列"""

		# 选择特征
		df_seq = self.processor.df[seq_features + ['arrival_delay']].copy()
		df_static = self.processor.df[static_features].copy() if static_features else None

		# 检查数据
		print(f"时序数据形状: {df_seq.shape}")
		if df_static is not None:
			print(f"静态数据形状: {df_static.shape}")

		# 标准化
		seq_scaler = RobustScaler()
		df_seq_scaled = seq_scaler.fit_transform(df_seq)

		if df_static is not None and len(static_features) > 0:
			static_scaler = StandardScaler()
			df_static_scaled = static_scaler.fit_transform(df_static)
		else:
			df_static_scaled = None

		# 保存标准化器和特征信息
		self.scalers = {
			'seq_scaler': seq_scaler,
			'static_scaler': static_scaler if df_static is not None else None,
			'seq_features': seq_features,  # 保存特征名称
			'static_features': static_features if static_features else [],  # 保存特征名称
			'seq_scaler_n_features_in': seq_scaler.n_features_in_ if hasattr(seq_scaler, 'n_features_in_') else len(
				seq_features),
			'static_scaler_n_features_in': static_scaler.n_features_in_ if static_scaler and hasattr(static_scaler,
			                                                                                         'n_features_in_') else len(
				static_features) if static_features else 0
		}

		# 创建序列
		X_seq, X_static, y = [], [], []

		for i in range(len(df_seq_scaled) - seq_length):
			X_seq.append(df_seq_scaled[i:i + seq_length, :-1])  # 排除目标变量
			if df_static_scaled is not None:
				X_static.append(df_static_scaled[i + seq_length])
			else:
				X_static.append(np.zeros((1,)))
			y.append(df_seq_scaled[i + seq_length, -1])

		if len(X_seq) == 0:
			print("警告：没有足够的数据创建序列，减少序列长度")
			seq_length = min(10, len(df_seq_scaled) // 2)
			for i in range(len(df_seq_scaled) - seq_length):
				X_seq.append(df_seq_scaled[i:i + seq_length, :-1])
				if df_static_scaled is not None:
					X_static.append(df_static_scaled[i + seq_length])
				else:
					X_static.append(np.zeros((1,)))
				y.append(df_seq_scaled[i + seq_length, -1])

		X_seq = np.array(X_seq)
		X_static = np.array(X_static)
		y = np.array(y).reshape(-1, 1)

		print(f"最终序列数据形状 - X_seq: {X_seq.shape}, X_static: {X_static.shape}, y: {y.shape}")

		return X_seq, X_static, y

	def _extract_xgb_features(self, X_seq, X_static):
		"""提取XGBoost特征"""

		n_samples = X_seq.shape[0]
		n_timesteps = X_seq.shape[1]
		n_features = X_seq.shape[2]

		features = []

		# 基本统计特征
		for i in range(n_features):
			feature_data = X_seq[:, :, i]

			# 统计特征
			features.append(np.mean(feature_data, axis=1))
			features.append(np.std(feature_data, axis=1))
			features.append(np.max(feature_data, axis=1))
			features.append(np.min(feature_data, axis=1))
			features.append(feature_data[:, -1])  # 最后值

			# 变化特征
			features.append(feature_data[:, -1] - feature_data[:, 0])

		# 添加静态特征
		if X_static is not None and len(X_static.shape) > 1:
			for i in range(X_static.shape[1]):
				features.append(X_static[:, i])

		# 转换为数组
		features_array = np.array(features).T

		print(f"XGBoost特征形状: {features_array.shape}")

		return features_array

	def train_lstm(self, data, epochs=15, patience=5):
		"""训练LSTM模型"""
		print("\n训练LSTM模型...")

		X_seq_train, X_static_train, y_train = data['lstm']['train']
		X_seq_val, X_static_val, y_val = data['lstm']['val']

		# 检查数据形状
		print(f"训练数据形状 - X_seq: {X_seq_train.shape}, X_static: {X_static_train.shape}, y: {y_train.shape}")
		print(f"验证数据形状 - X_seq: {X_seq_val.shape}, X_static: {X_static_val.shape}, y: {y_val.shape}")

		# 创建数据集
		train_dataset = TimeSeriesDataset(X_seq_train, X_static_train, y_train)
		val_dataset = TimeSeriesDataset(X_seq_val, X_static_val, y_val)

		# 数据加载器 - 减小批大小以节省内存
		train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
		val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

		# 创建模型
		seq_input_size = X_seq_train.shape[2]
		static_input_size = X_static_train.shape[1] if X_static_train is not None else 0

		print(f"创建LSTM模型 - 序列输入大小: {seq_input_size}, 静态输入大小: {static_input_size}")

		model = HybridLSTM(
			seq_input_size=seq_input_size,
			static_input_size=static_input_size,
			lstm_hidden_size=64,
			num_lstm_layers=1,
			dropout=0.2
		).to(device)

		print(f"LSTM模型参数总数: {sum(p.numel() for p in model.parameters()):,}")

		# 训练
		criterion = nn.HuberLoss(delta=1.0)
		optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

		scheduler = optim.lr_scheduler.ReduceLROnPlateau(
			optimizer, mode='min', factor=0.5, patience=3, verbose=True
		)

		best_val_loss = float('inf')
		best_model_state = None
		patience_counter = 0

		history = {'train_loss': [], 'val_loss': [], 'train_mae': [], 'val_mae': []}

		for epoch in range(epochs):
			# 训练
			model.train()
			train_loss, train_mae = 0, 0

			for batch in train_loader:
				if len(batch) == 3:
					X_seq_batch, X_static_batch, y_batch = batch
					X_seq_batch, X_static_batch = X_seq_batch.to(device), X_static_batch.to(device)
				else:
					X_seq_batch, y_batch = batch
					X_seq_batch = X_seq_batch.to(device)
					X_static_batch = None

				y_batch = y_batch.to(device)

				optimizer.zero_grad()
				outputs = model(X_seq_batch, X_static_batch)
				loss = criterion(outputs, y_batch)

				loss.backward()
				torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
				optimizer.step()

				train_loss += loss.item()
				train_mae += torch.abs(outputs - y_batch).mean().item()

				# 定期清理GPU内存
				if torch.cuda.is_available() and train_loader.batch_size * 10 % 1000 == 0:
					torch.cuda.empty_cache()

			avg_train_loss = train_loss / len(train_loader)
			avg_train_mae = train_mae / len(train_loader)

			# 验证
			model.eval()
			val_loss, val_mae = 0, 0

			with torch.no_grad():
				for batch in val_loader:
					if len(batch) == 3:
						X_seq_batch, X_static_batch, y_batch = batch
						X_seq_batch, X_static_batch = X_seq_batch.to(device), X_static_batch.to(device)
					else:
						X_seq_batch, y_batch = batch
						X_seq_batch = X_seq_batch.to(device)
						X_static_batch = None

					y_batch = y_batch.to(device)
					outputs = model(X_seq_batch, X_static_batch)
					loss = criterion(outputs, y_batch)

					val_loss += loss.item()
					val_mae += torch.abs(outputs - y_batch).mean().item()

			avg_val_loss = val_loss / len(val_loader)
			avg_val_mae = val_mae / len(val_loader)

			# 学习率调度
			scheduler.step(avg_val_loss)

			# 记录历史
			history['train_loss'].append(avg_train_loss)
			history['val_loss'].append(avg_val_loss)
			history['train_mae'].append(avg_train_mae)
			history['val_mae'].append(avg_val_mae)

			# 打印进度
			if (epoch + 1) % 3 == 0 or epoch == 0:
				print(f"Epoch {epoch + 1:03d}/{epochs} | "
				      f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
				      f"Train MAE: {avg_train_mae:.4f} | Val MAE: {avg_val_mae:.4f}")

			# 早停
			if avg_val_loss < best_val_loss:
				best_val_loss = avg_val_loss
				best_model_state = model.state_dict().copy()
				patience_counter = 0
			else:
				patience_counter += 1
				if patience_counter >= patience:
					print(f"\n早停在 epoch {epoch + 1} 触发")
					break

			# 清理GPU内存
			if torch.cuda.is_available():
				torch.cuda.empty_cache()

		# 加载最佳模型
		if best_model_state:
			model.load_state_dict(best_model_state)

		self.models['lstm'] = model
		self.results['lstm_history'] = history

		print(f"LSTM训练完成！最佳验证损失: {best_val_loss:.4f}")

		return model, history

	def train_xgboost(self, data):
		"""训练XGBoost模型 - 仅使用xgb.train API"""
		print("\n训练XGBoost模型...")

		X_train, y_train = data['xgb']['train']
		X_val, y_val = data['xgb']['val']

		print(f"XGBoost训练数据形状: X_train: {X_train.shape}, y_train: {y_train.shape}")
		print(f"XGBoost验证数据形状: X_val: {X_val.shape}, y_val: {y_val.shape}")

		# XGBoost参数 - 针对xgb.train API
		params = {
			'objective': 'reg:squarederror',
			'eval_metric': ['mae', 'rmse'],
			'max_depth': 6,
			'learning_rate': 0.1,
			'subsample': 0.8,
			'colsample_bytree': 0.8,
			'min_child_weight': 3,
			'gamma': 0.1,
			'alpha': 0.1,
			'lambda': 1.0,
			'n_estimators': 200,
			'random_state': 42,
			'n_jobs': -1
		}

		# 使用XGBoost训练器
		xgb_trainer = XGBoostTrainer()
		model = xgb_trainer.train_with_early_stopping(
			X_train, y_train.ravel(),
			X_val, y_val.ravel(),
			params
		)

		# 评估验证集性能
		y_val_pred = xgb_trainer.predict(X_val)
		val_mae = mean_absolute_error(y_val, y_val_pred)
		val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
		val_r2 = r2_score(y_val, y_val_pred)

		print(f"XGBoost验证集性能:")
		print(f"  MAE: {val_mae:.4f}")
		print(f"  RMSE: {val_rmse:.4f}")
		print(f"  R²: {val_r2:.4f}")

		# 存储训练器
		self.models['xgb'] = xgb_trainer

		return xgb_trainer

	def create_ensemble(self, data):
		"""创建集成模型"""
		print("\n创建集成模型...")

		# 获取LSTM预测
		X_seq_train, X_static_train, y_train = data['lstm']['train']
		X_seq_val, X_static_val, y_val = data['lstm']['val']

		# 获取XGBoost数据
		X_xgb_train, y_train_xgb = data['xgb']['train']
		X_xgb_val, y_val_xgb = data['xgb']['val']

		# 确保目标变量一致
		y_train = y_train.ravel()
		y_val = y_val.ravel()

		print(f"集成模型训练数据 - y_train形状: {y_train.shape}")
		print(f"集成模型验证数据 - y_val形状: {y_val.shape}")

		# 创建集成模型
		ensemble = EnsembleModel(
			self.models['lstm'],
			self.models['xgb'].model,  # 传递xgb.Booster模型
			meta_model_type='ridge'
		)

		# 训练元模型
		print("训练元模型...")
		ensemble.train_meta_model(
			(X_seq_train, X_static_train),
			X_xgb_train,
			y_train
		)

		# 评估集成模型在验证集上的性能
		ensemble_val_pred = ensemble.predict((X_seq_val, X_static_val), X_xgb_val)
		ensemble_mae = mean_absolute_error(y_val, ensemble_val_pred)
		ensemble_rmse = np.sqrt(mean_squared_error(y_val, ensemble_val_pred))
		ensemble_r2 = r2_score(y_val, ensemble_val_pred)

		print(f"集成模型验证集性能:")
		print(f"  MAE: {ensemble_mae:.4f}")
		print(f"  RMSE: {ensemble_rmse:.4f}")
		print(f"  R²: {ensemble_r2:.4f}")

		self.models['ensemble'] = ensemble

		return ensemble

	def evaluate_models(self, data):
		"""评估所有模型"""
		print("\n评估所有模型...")

		# 准备测试数据
		X_seq_test, X_static_test, y_test = data['lstm']['test']
		X_xgb_test, y_test_xgb = data['xgb']['test']

		# 确保目标变量一致
		y_test = y_test.ravel()

		print(f"测试数据形状 - y_test: {y_test.shape}")

		results = {}

		# 1. 评估LSTM - 使用分批预测
		print("\n1. 评估LSTM模型...")
		lstm_pred = self._predict_lstm_batch(X_seq_test, X_static_test, batch_size=512)
		results['LSTM'] = self._calculate_metrics(y_test, lstm_pred)

		# 2. 评估XGBoost
		print("\n2. 评估XGBoost模型...")
		xgb_pred = self.models['xgb'].predict(X_xgb_test)
		results['XGBoost'] = self._calculate_metrics(y_test, xgb_pred)

		# 3. 评估集成模型
		print("\n3. 评估集成模型...")
		ensemble_pred = self.models['ensemble'].predict(
			(X_seq_test, X_static_test),
			X_xgb_test
		)
		results['Ensemble'] = self._calculate_metrics(y_test, ensemble_pred.flatten())

		# 4. 配对样本t检验：比较集成模型与XGBoost的预测误差
		print("\n4. 执行配对样本t检验：集成模型 vs XGBoost...")
		t_test_results = self._perform_paired_t_test(data)
		results['T-Test'] = t_test_results

		# 5. 按延误程度分析
		print("\n5. 按延误程度分析...")
		delay_analysis_results = self.analyze_by_delay_severity(data)
		results['Delay Severity Analysis'] = delay_analysis_results

		# 6. 按不确定性分析
		print("\n6. 按不确定性分析...")
		uncertainty_analysis_results = self.analyze_by_uncertainty(data)
		results['Uncertainty Analysis'] = uncertainty_analysis_results

		# 存储测试集预测结果供后续分析使用
		self.results['test_predictions'] = {
			'y_true': y_test,
			'LSTM': lstm_pred,
			'XGBoost': xgb_pred,
			'Ensemble': ensemble_pred.flatten()
		}

		# 存储结果
		self.results['model_performance'] = results

		# 打印详细结果
		print("\n" + "=" * 70)
		print("模型性能总结")
		print("=" * 70)

		for model_name, metrics in results.items():
			if model_name in ['T-Test', 'Delay Severity Analysis', 'Uncertainty Analysis']:
				continue
			print(f"\n{model_name}:")
			for metric_name, metric_value in metrics.items():
				if metric_name != 'weights':
					print(f"  {metric_name}: {metric_value:.4f}")

		# 打印t检验结果
		print("配对样本t检验结果：")
		print(f"零假设：集成模型和XGBoost的预测误差均值无显著差异")
		print(f"样本量: {t_test_results['sample_size']}")
		print(f"误差差值的均值: {t_test_results['mean_difference']:.4f}")
		print(f"误差差值的标准差: {t_test_results['std_difference']:.4f}")
		print(f"p值: {t_test_results['p_value']:.6f}")
		print(f"显著性水平: α = 0.05")
		return results

	def analyze_by_delay_severity(self, data):
		"""按延误程度分析：将测试集样本按真实延误时间分为三组"""
		print("\n按延误程度分析...")

		# 准备测试数据
		X_seq_test, X_static_test, y_test = data['lstm']['test']
		X_xgb_test, _ = data['xgb']['test']

		# 确保y_test是一维数组
		y_test = y_test.ravel()

		# 获取集成模型和XGBoost预测
		ensemble_pred = self.models['ensemble'].predict(
			(X_seq_test, X_static_test),
			X_xgb_test
		).flatten()

		xgb_pred = self.models['xgb'].predict(X_xgb_test)

		# 确保形状一致
		min_len = min(len(y_test), len(ensemble_pred), len(xgb_pred))
		y_test = y_test[:min_len]
		ensemble_pred = ensemble_pred[:min_len]
		if len(xgb_pred.shape) > 1:
			xgb_pred = xgb_pred[:min_len].ravel()
		else:
			xgb_pred = xgb_pred[:min_len]

		# 定义延误程度分组
		delay_severity_groups = {
			'轻微延误/提前 (-15至15分钟)': (-15, 15),
			'中等延误 (15至60分钟)': (15, 60),
			'严重延误 (>60分钟)': (60, float('inf'))
		}

		analysis_results = {}

		for group_name, (lower, upper) in delay_severity_groups.items():
			if lower == -15 and upper == 15:
				mask = (y_test >= lower) & (y_test <= upper)
			elif lower == 15 and upper == 60:
				mask = (y_test > lower) & (y_test <= upper)
			else:  # 严重延误
				mask = y_test > lower

			if mask.sum() > 0:
				# 计算该组的MAE
				ensemble_mae = mean_absolute_error(y_test[mask], ensemble_pred[mask])
				xgb_mae = mean_absolute_error(y_test[mask], xgb_pred[mask])

				# 计算相对改进率
				if xgb_mae > 0:
					improvement_rate = (xgb_mae - ensemble_mae) / xgb_mae * 100
				else:
					improvement_rate = 0

				analysis_results[group_name] = {
					'样本数量': mask.sum(),
					'Ensemble MAE': ensemble_mae,
					'XGBoost MAE': xgb_mae,
					'相对改进率 (%)': improvement_rate
				}

				print(f"{group_name}:")
				print(f"  样本数量: {mask.sum()}")
				print(f"  Ensemble MAE: {ensemble_mae:.4f}")
				print(f"  XGBoost MAE: {xgb_mae:.4f}")
				print(f"  相对改进率: {improvement_rate:.2f}%")

		# 存储分析结果
		self.results['delay_severity_analysis'] = {
			'results': analysis_results,
			'y_true': y_test,
			'ensemble_pred': ensemble_pred,
			'xgb_pred': xgb_pred
		}

		return analysis_results

	def analyze_by_uncertainty(self, data):
		"""按不确定性分析：将测试集按预测不确定性（基模型预测值的绝对差）分为三组"""
		print("\n按不确定性分析...")

		# 准备测试数据
		X_seq_test, X_static_test, y_test = data['lstm']['test']
		X_xgb_test, _ = data['xgb']['test']

		# 确保y_test是一维数组
		y_test = y_test.ravel()

		# 获取基模型预测值和绝对差值（不确定性度量）
		lstm_pred, xgb_pred, abs_diff = self.models['ensemble'].get_base_model_predictions(
			(X_seq_test, X_static_test),
			X_xgb_test
		)

		# 获取集成模型预测
		ensemble_pred = self.models['ensemble'].predict(
			(X_seq_test, X_static_test),
			X_xgb_test
		).flatten()

		# 确保形状一致
		min_len = min(len(y_test), len(ensemble_pred), len(lstm_pred), len(xgb_pred), len(abs_diff))
		y_test = y_test[:min_len]
		ensemble_pred = ensemble_pred[:min_len]
		lstm_pred = lstm_pred[:min_len].flatten() if len(lstm_pred.shape) > 1 else lstm_pred[:min_len]
		xgb_pred = xgb_pred[:min_len].flatten() if len(xgb_pred.shape) > 1 else xgb_pred[:min_len]
		abs_diff = abs_diff[:min_len].flatten() if len(abs_diff.shape) > 1 else abs_diff[:min_len]

		# 计算简单平均法预测
		simple_avg_pred = (lstm_pred + xgb_pred) / 2

		# 按绝对差值（不确定性）分位数分组
		percentiles = np.percentile(abs_diff, [33, 67])

		# 定义不确定性分组
		uncertainty_groups = {
			'低不确定性': (0, percentiles[0]),
			'中不确定性': (percentiles[0], percentiles[1]),
			'高不确定性': (percentiles[1], float('inf'))
		}

		analysis_results = {}

		for group_name, (lower, upper) in uncertainty_groups.items():
			if lower == 0:
				mask = (abs_diff >= lower) & (abs_diff < upper)
			else:
				mask = (abs_diff >= lower) & (abs_diff < upper) if upper != float('inf') else (abs_diff >= lower)

			if mask.sum() > 0:
				# 计算该组的MAE
				ensemble_mae = mean_absolute_error(y_test[mask], ensemble_pred[mask])
				simple_avg_mae = mean_absolute_error(y_test[mask], simple_avg_pred[mask])

				# 计算绝对差值和相对改进
				mae_diff = simple_avg_mae - ensemble_mae
				if simple_avg_mae > 0:
					improvement_rate = mae_diff / simple_avg_mae * 100
				else:
					improvement_rate = 0

				# 计算该组不确定性统计
				mean_uncertainty = np.mean(abs_diff[mask])
				median_uncertainty = np.median(abs_diff[mask])

				analysis_results[group_name] = {
					'样本数量': mask.sum(),
					'平均不确定性': mean_uncertainty,
					'中位数不确定性': median_uncertainty,
					'Ensemble MAE': ensemble_mae,
					'简单平均法 MAE': simple_avg_mae,
					'MAE差值': mae_diff,
					'相对改进率 (%)': improvement_rate
				}

				print(f"{group_name}:")
				print(f"  样本数量: {mask.sum()}")
				print(f"  平均不确定性: {mean_uncertainty:.4f}")
				print(f"  Ensemble MAE: {ensemble_mae:.4f}")
				print(f"  简单平均法 MAE: {simple_avg_mae:.4f}")
				print(f"  MAE差值: {mae_diff:.4f}")
				print(f"  相对改进率: {improvement_rate:.2f}%")

		# 存储分析结果
		self.results['uncertainty_analysis'] = {
			'results': analysis_results,
			'y_true': y_test,
			'ensemble_pred': ensemble_pred,
			'simple_avg_pred': simple_avg_pred,
			'abs_diff': abs_diff,
			'lstm_pred': lstm_pred,
			'xgb_pred': xgb_pred
		}

		return analysis_results

	def _predict_lstm_batch(self, X_seq, X_static, batch_size=512):
		"""分批LSTM预测 - 避免内存溢出"""
		self.models['lstm'].eval()

		n_samples = X_seq.shape[0]
		predictions = []

		# 分批处理
		for i in range(0, n_samples, batch_size):
			end_idx = min(i + batch_size, n_samples)

			with torch.no_grad():
				X_seq_batch = torch.FloatTensor(X_seq[i:end_idx]).to(device)
				if X_static is not None:
					X_static_batch = torch.FloatTensor(X_static[i:end_idx]).to(device)
					outputs = self.models['lstm'](X_seq_batch, X_static_batch)
				else:
					outputs = self.models['lstm'](X_seq_batch, None)

				predictions.append(outputs.cpu().numpy())

			# 释放GPU内存
			if torch.cuda.is_available():
				torch.cuda.empty_cache()

		return np.vstack(predictions).flatten()

	def _calculate_metrics(self, y_true, y_pred):
		"""计算评估指标"""
		return {
			'MAE': mean_absolute_error(y_true, y_pred),
			'MSE': mean_squared_error(y_true, y_pred),
			'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
			'R²': r2_score(y_true, y_pred)
		}

	def _perform_paired_t_test(self, data):
		"""执行配对样本t检验：比较集成模型与XGBoost的预测误差"""
		# 准备测试数据
		X_seq_test, X_static_test, y_test = data['lstm']['test']
		X_xgb_test, _ = data['xgb']['test']

		# 确保y_test是一维数组
		y_test = y_test.ravel()

		# 获取XGBoost预测
		print("获取XGBoost预测用于t检验...")
		xgb_pred = self.models['xgb'].predict(X_xgb_test)

		# 获取集成模型预测
		print("获取集成模型预测用于t检验...")
		ensemble_pred = self.models['ensemble'].predict(
			(X_seq_test, X_static_test),
			X_xgb_test
		).flatten()

		# 确保所有预测都是一维数组
		if len(xgb_pred.shape) > 1:
			xgb_pred = xgb_pred.ravel()
		if len(ensemble_pred.shape) > 1:
			ensemble_pred = ensemble_pred.ravel()

		# 检查形状是否匹配
		min_len = min(len(y_test), len(xgb_pred), len(ensemble_pred))
		y_test = y_test[:min_len]
		xgb_pred = xgb_pred[:min_len]
		ensemble_pred = ensemble_pred[:min_len]

		# 计算预测误差
		xgb_errors = y_test - xgb_pred
		ensemble_errors = y_test - ensemble_pred

		# 计算误差差值（集成模型误差 - XGBoost误差）
		error_differences = ensemble_errors - xgb_errors

		# 移除NaN和无穷大值
		valid_indices = np.isfinite(error_differences)
		error_differences = error_differences[valid_indices]

		if len(error_differences) < 2:
			print("警告：有效样本量太少，无法进行t检验")
			return {
				'sample_size': len(error_differences),
				'mean_difference': np.nan,
				'std_difference': np.nan,
				't_statistic': np.nan,
				'p_value': np.nan,
				'ci_lower': np.nan,
				'ci_upper': np.nan,
				'effect_size': np.nan,
				'normality_test_p': np.nan,
				'wilcoxon_statistic': np.nan,
				'wilcoxon_p': np.nan
			}

		# 执行配对样本t检验
		t_statistic, p_value = stats.ttest_rel(ensemble_errors, xgb_errors)

		# 计算描述性统计量
		mean_difference = np.mean(error_differences)
		std_difference = np.std(error_differences, ddof=1)  # 样本标准差

		# 计算95%置信区间
		ci_lower, ci_upper = stats.t.interval(
			0.95,
			len(error_differences) - 1,
			loc=mean_difference,
			scale=std_difference / np.sqrt(len(error_differences))
		)

		# 计算效应量 (Cohen's d)
		effect_size = mean_difference / std_difference

		# 正态性检验（Shapiro-Wilk检验）
		if len(error_differences) < 5000:  # Shapiro检验对大数据集不准确
			normality_test_statistic, normality_test_p = stats.shapiro(error_differences)
		else:
			# 对大数据集使用Kolmogorov-Smirnov检验
			normality_test_statistic, normality_test_p = stats.kstest(
				(error_differences - np.mean(error_differences)) / np.std(error_differences),
				'norm'
			)

		# Wilcoxon符号秩检验（非参数配对检验）
		wilcoxon_statistic, wilcoxon_p = stats.wilcoxon(ensemble_errors, xgb_errors)

		# 存储t检验结果
		t_test_results = {
			'sample_size': len(error_differences),
			'mean_difference': mean_difference,
			'std_difference': std_difference,
			't_statistic': t_statistic,
			'p_value': p_value,
			'ci_lower': ci_lower,
			'ci_upper': ci_upper,
			'effect_size': effect_size,
			'normality_test_p': normality_test_p,
			'wilcoxon_statistic': wilcoxon_statistic,
			'wilcoxon_p': wilcoxon_p
		}

		# 将误差数据存储在results中供后续可视化使用
		self.results['t_test_data'] = {
			'xgb_errors': xgb_errors,
			'ensemble_errors': ensemble_errors,
			'error_differences': error_differences,
			't_test_results': t_test_results
		}

		return t_test_results

	def visualize_results(self, data):
		"""可视化结果"""
		if 'model_performance' not in self.results:
			print("没有可用的结果数据")
			return

		results = self.results['model_performance']

		# 创建包含多个子图的图表
		fig, axes = plt.subplots(2, 2, figsize=(18, 12))
		fig.suptitle('模型性能对比分析', fontsize=20, fontweight='bold')

		# 获取模型列表，排除'T-Test'，因为它没有MAE等指标
		models = [model for model in results.keys() if
		          model not in ['T-Test', 'Delay Severity Analysis', 'Uncertainty Analysis']]

		# 1. MAE对比
		ax1 = axes[0, 0]
		# 只选择有MAE指标的模型
		mae_values = [results[model]['MAE'] for model in models if 'MAE' in results[model]]
		models_with_mae = [model for model in models if 'MAE' in results[model]]

		if mae_values:  # 确保有数据
			colors = plt.cm.Set3(np.linspace(0, 1, len(mae_values)))
			bars1 = ax1.bar(range(len(mae_values)), mae_values, color=colors)
			ax1.set_xlabel('模型', fontsize=14)
			ax1.set_ylabel('MAE', fontsize=14)
			ax1.set_title('平均绝对误差(MAE)对比', fontsize=16, fontweight='bold')
			ax1.set_xticks(range(len(mae_values)))
			ax1.set_xticklabels(models_with_mae, rotation=0, fontsize=12)
			ax1.grid(True, alpha=0.3)

			# 在柱子上添加数值
			for bar, value in zip(bars1, mae_values):
				height = bar.get_height()
				ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.001,
				         f'{value:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
		else:
			ax1.text(0.5, 0.5, '无MAE数据', ha='center', va='center', fontsize=14)
			ax1.set_title('平均绝对误差(MAE)对比', fontsize=16, fontweight='bold')

		# 2. RMSE对比
		ax2 = axes[0, 1]
		# 只选择有RMSE指标的模型
		rmse_values = [results[model]['RMSE'] for model in models if 'RMSE' in results[model]]
		models_with_rmse = [model for model in models if 'RMSE' in results[model]]

		if rmse_values:  # 确保有数据
			colors = plt.cm.Set3(np.linspace(0, 1, len(rmse_values)))
			bars2 = ax2.bar(range(len(rmse_values)), rmse_values, color=colors)
			ax2.set_xlabel('模型', fontsize=14)
			ax2.set_ylabel('RMSE', fontsize=14)
			ax2.set_title('均方根误差(RMSE)对比', fontsize=16, fontweight='bold')
			ax2.set_xticks(range(len(rmse_values)))
			ax2.set_xticklabels(models_with_rmse, rotation=0, fontsize=12)
			ax2.grid(True, alpha=0.3)

			# 在柱子上添加数值
			for bar, value in zip(bars2, rmse_values):
				height = bar.get_height()
				ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.001,
				         f'{value:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
		else:
			ax2.text(0.5, 0.5, '无RMSE数据', ha='center', va='center', fontsize=14)
			ax2.set_title('均方根误差(RMSE)对比', fontsize=16, fontweight='bold')

		# 3. R²对比
		ax3 = axes[1, 0]
		# 只选择有R²指标的模型
		r2_values = [results[model]['R²'] for model in models if 'R²' in results[model]]
		models_with_r2 = [model for model in models if 'R²' in results[model]]

		if r2_values:  # 确保有数据
			colors = plt.cm.Set3(np.linspace(0, 1, len(r2_values)))
			bars3 = ax3.bar(range(len(r2_values)), r2_values, color=colors)
			ax3.set_xlabel('模型', fontsize=14)
			ax3.set_ylabel('R²', fontsize=14)
			ax3.set_title('决定系数(R²)对比', fontsize=16, fontweight='bold')
			ax3.set_xticks(range(len(r2_values)))
			ax3.set_xticklabels(models_with_r2, rotation=0, fontsize=12)
			ax3.grid(True, alpha=0.3)

			# 设置合适的y轴范围
			min_r2 = min(r2_values) if r2_values else 0
			max_r2 = max(r2_values) if r2_values else 1
			ax3.set_ylim([min(0, min_r2 - 0.1), max(1, max_r2 + 0.1)])

			# 在柱子上添加数值
			for bar, value in zip(bars3, r2_values):
				height = bar.get_height()
				ax3.text(bar.get_x() + bar.get_width() / 2., height + 0.001,
				         f'{value:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
		else:
			ax3.text(0.5, 0.5, '无R²数据', ha='center', va='center', fontsize=14)
			ax3.set_title('决定系数(R²)对比', fontsize=16, fontweight='bold')

		# 4. MSE对比
		ax5 = axes[1, 1]
		# 只选择有MSE指标的模型
		mse_values = [results[model]['MSE'] for model in models if 'MSE' in results[model]]
		models_with_mse = [model for model in models if 'MSE' in results[model]]

		if mse_values:  # 确保有数据
			colors = plt.cm.Set3(np.linspace(0, 1, len(mse_values)))
			bars5 = ax5.bar(range(len(mse_values)), mse_values, color=colors)
			ax5.set_xlabel('模型', fontsize=14)
			ax5.set_ylabel('MSE', fontsize=14)
			ax5.set_title('均方误差(MSE)对比', fontsize=16, fontweight='bold')
			ax5.set_xticks(range(len(mse_values)))
			ax5.set_xticklabels(models_with_mse, rotation=0, fontsize=12)
			ax5.grid(True, alpha=0.3)

			# 在柱子上添加数值
			for bar, value in zip(bars5, mse_values):
				height = bar.get_height()
				ax5.text(bar.get_x() + bar.get_width() / 2., height + 0.001,
				         f'{value:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
		else:
			ax5.text(0.5, 0.5, '无MSE数据', ha='center', va='center', fontsize=14)
			ax5.set_title('均方误差(MSE)对比', fontsize=16, fontweight='bold')

		plt.tight_layout()
		plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
		plt.show()

		# 5. 性能改进分析图
		fig2, ax6 = plt.subplots(1, 1, figsize=(8, 6))
		fig2.suptitle('性能改进分析', fontsize=16, fontweight='bold')

		# 检查是否有必要的模型数据
		required_models = ['LSTM', 'XGBoost', 'Ensemble']
		has_all_models = all(model in results and all(key in results[model] for key in ['MAE', 'RMSE', 'R²'])
		                     for model in required_models)

		if has_all_models:
			lstm_mae = results['LSTM']['MAE']
			xgb_mae = results['XGBoost']['MAE']
			ensemble_mae = results['Ensemble']['MAE']

			lstm_rmse = results['LSTM']['RMSE']
			xgb_rmse = results['XGBoost']['RMSE']
			ensemble_rmse = results['Ensemble']['RMSE']

			lstm_r2 = results['LSTM']['R²']
			xgb_r2 = results['XGBoost']['R²']
			ensemble_r2 = results['Ensemble']['R²']

			improvement_mae_over_lstm = (lstm_mae - ensemble_mae) / lstm_mae * 100 if lstm_mae != 0 else 0
			improvement_mae_over_xgb = (xgb_mae - ensemble_mae) / xgb_mae * 100 if xgb_mae != 0 else 0

			improvement_rmse_over_lstm = (lstm_rmse - ensemble_rmse) / lstm_rmse * 100 if lstm_rmse != 0 else 0
			improvement_rmse_over_xgb = (xgb_rmse - ensemble_rmse) / xgb_rmse * 100 if xgb_rmse != 0 else 0

			improvement_r2_over_lstm = (ensemble_r2 - lstm_r2) / abs(lstm_r2) * 100 if lstm_r2 != 0 else 0
			improvement_r2_over_xgb = (ensemble_r2 - xgb_r2) / abs(xgb_r2) * 100 if xgb_r2 != 0 else 0

			# 查找最佳模型
			model_mae_pairs = [(model, results[model]['MAE']) for model in models if 'MAE' in results[model]]
			if model_mae_pairs:
				best_model, best_mae = min(model_mae_pairs, key=lambda x: x[1])
			else:
				best_model = "未知"
				best_mae = 0

			text_content = "集成模型性能改进分析\n\n"
			text_content += f"集成模型 vs LSTM:\n"
			text_content += f"  MAE改进: {improvement_mae_over_lstm:.2f}%\n"
			text_content += f"  RMSE改进: {improvement_rmse_over_lstm:.2f}%\n"
			text_content += f"  R²改进: {improvement_r2_over_lstm:.2f}%\n\n"

			text_content += f"集成模型 vs XGBoost:\n"
			text_content += f"  MAE改进: {improvement_mae_over_xgb:.2f}%\n"
			text_content += f"  RMSE改进: {improvement_rmse_over_xgb:.2f}%\n"
			text_content += f"  R²改进: {improvement_r2_over_xgb:.2f}%\n\n"

			text_content += f"最优模型: "
			text_content += f"{best_model} (MAE: {best_mae:.4f})"

			ax6.text(0.05, 0.95, text_content,
			         ha='left', va='top', transform=ax6.transAxes,
			         fontsize=12, bbox=dict(boxstyle="round,pad=0.5",
			                                facecolor="lightyellow",
			                                alpha=0.9),
			         linespacing=1.5)
		else:
			ax6.text(0.5, 0.5, '缺少必要的模型数据进行分析',
			         ha='center', va='center', fontsize=14)

		ax6.axis('off')
		plt.tight_layout()
		plt.savefig('improvement_analysis.png', dpi=150, bbox_inches='tight')
		plt.show()

		# 6. 预测误差分布图
		print("创建误差分布图...")

		# 获取测试数据
		X_seq_test, X_static_test, y_test = data['lstm']['test']
		X_xgb_test, _ = data['xgb']['test']

		# 确保y_test是一维数组
		y_test = y_test.ravel()
		print(f"y_test形状: {y_test.shape}")

		# 计算预测（使用分批处理）
		print("计算LSTM预测...")
		lstm_pred = self._predict_lstm_batch(X_seq_test, X_static_test, batch_size=512)
		print(f"lstm_pred原始形状: {lstm_pred.shape}")

		# 获取XGBoost预测
		print("计算XGBoost预测...")
		xgb_pred = self.models['xgb'].predict(X_xgb_test)
		print(f"xgb_pred形状: {xgb_pred.shape}")

		# 获取集成模型预测
		print("计算集成模型预测...")
		ensemble_pred = self.models['ensemble'].predict(
			(X_seq_test, X_static_test),
			X_xgb_test
		).flatten()
		print(f"ensemble_pred形状: {ensemble_pred.shape}")

		# 确保所有预测都是一维数组
		if len(lstm_pred.shape) > 1:
			lstm_pred = lstm_pred.ravel()
		if len(xgb_pred.shape) > 1:
			xgb_pred = xgb_pred.ravel()

		print(
			f"处理后形状 - y_test: {y_test.shape}, lstm_pred: {lstm_pred.shape}, xgb_pred: {xgb_pred.shape}, ensemble_pred: {ensemble_pred.shape}")

		# 检查形状是否匹配
		if y_test.shape[0] != lstm_pred.shape[0]:
			print(f"警告: y_test({y_test.shape})和lstm_pred({lstm_pred.shape})的形状不匹配")
			# 调整到相同长度
			min_len = min(y_test.shape[0], lstm_pred.shape[0])
			y_test = y_test[:min_len]
			lstm_pred = lstm_pred[:min_len]
			xgb_pred = xgb_pred[:min_len] if xgb_pred.shape[0] > min_len else xgb_pred
			ensemble_pred = ensemble_pred[:min_len] if ensemble_pred.shape[0] > min_len else ensemble_pred

		# 计算误差 - 使用安全的方式
		errors = {}

		try:
			print("计算LSTM误差...")
			errors['LSTM'] = y_test - lstm_pred
		except Exception as e:
			print(f"计算LSTM误差失败: {e}")
			# 创建安全的误差数组
			errors['LSTM'] = np.zeros_like(y_test)

		try:
			print("计算XGBoost误差...")
			errors['XGBoost'] = y_test - xgb_pred
		except Exception as e:
			print(f"计算XGBoost误差失败: {e}")
			errors['XGBoost'] = np.zeros_like(y_test)

		try:
			print("计算集成模型误差...")
			errors['Ensemble'] = y_test - ensemble_pred
		except Exception as e:
			print(f"计算集成模型误差失败: {e}")
			errors['Ensemble'] = np.zeros_like(y_test)

		# 绘制误差分布图
		fig3, axes3 = plt.subplots(1, 3, figsize=(18, 5))
		fig3.suptitle('模型预测误差分布 (抽样显示)', fontsize=16, fontweight='bold')

		# 限制样本数量以节省内存
		max_samples = 5000
		sample_indices = np.random.choice(len(y_test), min(max_samples, len(y_test)), replace=False)

		for idx, (model_name, error) in enumerate(errors.items()):
			if idx >= 3:  # 只显示前3个模型
				break

			ax = axes3[idx]

			# 使用抽样数据
			error_sample = error[sample_indices] if len(error) > max_samples else error

			# 限制误差范围以更好地显示
			error_clipped = np.clip(error_sample, -60, 60)

			# 误差分布
			ax.hist(error_clipped, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
			ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='零误差')
			ax.axvline(x=np.mean(error_clipped), color='green', linestyle='-', linewidth=2,
			           label=f'均值: {np.mean(error_clipped):.2f}')
			ax.axvline(x=np.median(error_clipped), color='orange', linestyle='-', linewidth=2,
			           label=f'中位数: {np.median(error_clipped):.2f}')

			ax.set_xlabel('预测误差 (分钟)', fontsize=12)
			ax.set_ylabel('频数', fontsize=12)
			ax.set_title(f'{model_name} - 误差分布', fontsize=14)
			ax.legend(fontsize=10)
			ax.grid(True, alpha=0.3)

		# 如果模型少于3个，隐藏多余的子图
		for idx in range(len(errors), 3):
			axes3[idx].axis('off')

		plt.tight_layout()
		plt.savefig('error_distributions.png', dpi=150, bbox_inches='tight')
		plt.show()

		# 7. 配对样本t检验结果可视化
		if 't_test_data' in self.results:
			print("创建配对样本t检验结果可视化...")
			self._visualize_t_test_results()
		else:
			print("没有可用的t检验数据")

		# 8. 按延误程度分析可视化
		if 'delay_severity_analysis' in self.results:
			print("创建按延误程度分析可视化...")
			self._visualize_delay_severity_analysis()

		# 9. 按不确定性分析可视化
		if 'uncertainty_analysis' in self.results:
			print("创建按不确定性分析可视化...")
			self._visualize_uncertainty_analysis()

		return results

	def _visualize_t_test_results(self):
		"""可视化配对样本t检验结果"""
		if 't_test_data' not in self.results:
			print("没有可用的t检验数据")
			return

		t_test_data = self.results['t_test_data']
		error_differences = t_test_data['error_differences']
		t_test_results = t_test_data['t_test_results']

		if len(error_differences) < 2:
			print("样本量太少，无法进行t检验可视化")
			return

		# 创建配对样本t检验结果可视化
		fig, axes = plt.subplots(2, 2, figsize=(16, 12))
		fig.suptitle('配对样本t检验分析：集成模型 vs XGBoost', fontsize=18, fontweight='bold')

		# 1. 误差差值分布图
		ax1 = axes[0, 0]
		ax1.hist(error_differences, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
		ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='无差异线')
		ax1.axvline(x=t_test_results['mean_difference'], color='green', linestyle='-', linewidth=2,
		            label=f'均值: {t_test_results["mean_difference"]:.4f}')
		ax1.axvline(x=t_test_results['ci_lower'], color='orange', linestyle=':', linewidth=2,
		            label=f'95% CI下限: {t_test_results["ci_lower"]:.4f}')
		ax1.axvline(x=t_test_results['ci_upper'], color='orange', linestyle=':', linewidth=2,
		            label=f'95% CI上限: {t_test_results["ci_upper"]:.4f}')

		ax1.set_xlabel('误差差值 (集成模型 - XGBoost)', fontsize=12)
		ax1.set_ylabel('频数', fontsize=12)
		ax1.set_title('误差差值分布', fontsize=14)
		ax1.legend(fontsize=10)
		ax1.grid(True, alpha=0.3)

		# 2. 误差差值Q-Q图（正态性检验）
		ax2 = axes[0, 1]
		stats.probplot(error_differences, dist="norm", plot=ax2)
		ax2.set_title('Q-Q图 (正态性检验)', fontsize=14)
		ax2.grid(True, alpha=0.3)

		# 添加正态性检验结果注释
		normality_text = f'Shapiro-Wilk检验:\n'
		normality_text += f'p值 = {t_test_results["normality_test_p"]:.6f}\n'
		if t_test_results["normality_test_p"] < 0.05:
			normality_text += '拒绝正态性假设'
		else:
			normality_text += '接受正态性假设'

		ax2.text(0.05, 0.95, normality_text,
		         transform=ax2.transAxes,
		         fontsize=10,
		         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9),
		         verticalalignment='top')

		# 3. 配对误差散点图
		ax3 = axes[1, 0]
		sample_size = min(1000, len(t_test_data['xgb_errors']))  # 限制样本数量
		sample_indices = np.random.choice(len(t_test_data['xgb_errors']), sample_size, replace=False)

		xgb_errors_sample = t_test_data['xgb_errors'][sample_indices]
		ensemble_errors_sample = t_test_data['ensemble_errors'][sample_indices]

		# 创建散点图
		scatter = ax3.scatter(xgb_errors_sample, ensemble_errors_sample,
		                      alpha=0.5, c='blue', edgecolors='black', linewidth=0.5)

		# 添加y=x参考线
		min_val = min(np.min(xgb_errors_sample), np.min(ensemble_errors_sample))
		max_val = max(np.max(xgb_errors_sample), np.max(ensemble_errors_sample))
		ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='y=x (无差异线)')

		ax3.set_xlabel('XGBoost预测误差', fontsize=12)
		ax3.set_ylabel('集成模型预测误差', fontsize=12)
		ax3.set_title('配对误差散点图', fontsize=14)
		ax3.legend(fontsize=10)
		ax3.grid(True, alpha=0.3)

		# 添加象限注释
		ax3.text(0.05, 0.95, '右下: XGBoost误差更大\n左上: 集成模型误差更大',
		         transform=ax3.transAxes,
		         fontsize=10,
		         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9),
		         verticalalignment='top')

		plt.tight_layout()
		plt.savefig('paired_t_test_results.png', dpi=150, bbox_inches='tight')
		plt.show()

	def _visualize_delay_severity_analysis(self):
		"""可视化按延误程度分析结果"""
		if 'delay_severity_analysis' not in self.results:
			print("没有可用的延误程度分析数据")
			return

		delay_analysis = self.results['delay_severity_analysis']
		analysis_results = delay_analysis['results']

		if not analysis_results:
			print("延误程度分析结果为空")
			return

		# 创建延误程度分析可视化
		fig, axes = plt.subplots(1, 2, figsize=(16, 6))
		fig.suptitle('按延误程度分析：集成模型性能改进', fontsize=18, fontweight='bold')

		# 1. 相对改进率柱状图
		ax1 = axes[0]

		groups = list(analysis_results.keys())
		improvement_rates = [analysis_results[group]['相对改进率 (%)'] for group in groups]

		colors = ['lightgreen', 'orange', 'red']
		bars = ax1.bar(range(len(groups)), improvement_rates, color=colors, edgecolor='black')

		ax1.set_xlabel('延误程度分组', fontsize=14)
		ax1.set_ylabel('相对改进率 (%)', fontsize=14)
		ax1.set_title('集成模型相对于XGBoost的MAE相对改进率', fontsize=16, fontweight='bold')
		ax1.set_xticks(range(len(groups)))
		ax1.set_xticklabels(groups, rotation=0, fontsize=12)
		ax1.grid(True, alpha=0.3, axis='y')

		# 在柱子上添加数值
		for bar, rate in zip(bars, improvement_rates):
			height = bar.get_height()
			ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
			         f'{rate:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

		# 添加显著性标记（严重延误组改进最明显）
		if len(groups) >= 3 and improvement_rates[2] > improvement_rates[0] and improvement_rates[2] > \
				improvement_rates[1]:
			ax1.text(2, improvement_rates[2] + 1, '改进最明显', ha='center', va='bottom',
			         fontsize=12, fontweight='bold', color='red')

		# 2. 各组的样本数量和MAE对比
		ax2 = axes[1]

		x = np.arange(len(groups))
		width = 0.35

		ensemble_mae = [analysis_results[group]['Ensemble MAE'] for group in groups]
		xgb_mae = [analysis_results[group]['XGBoost MAE'] for group in groups]
		sample_counts = [analysis_results[group]['样本数量'] for group in groups]

		bars1 = ax2.bar(x - width / 2, ensemble_mae, width, label='集成模型', color='steelblue', alpha=0.8)
		bars2 = ax2.bar(x + width / 2, xgb_mae, width, label='XGBoost', color='salmon', alpha=0.8)

		ax2.set_xlabel('延误程度分组', fontsize=14)
		ax2.set_ylabel('MAE', fontsize=14)
		ax2.set_title('各延误程度分组的MAE对比', fontsize=16, fontweight='bold')
		ax2.set_xticks(x)
		ax2.set_xticklabels(groups, rotation=0, fontsize=12)
		ax2.legend(fontsize=12)
		ax2.grid(True, alpha=0.3, axis='y')

		# 在柱子上添加数值
		for i, (bar1, bar2, count) in enumerate(zip(bars1, bars2, sample_counts)):
			height1 = bar1.get_height()
			height2 = bar2.get_height()
			ax2.text(bar1.get_x() + bar1.get_width() / 2., height1 + 0.001,
			         f'{height1:.2f}', ha='center', va='bottom', fontsize=10)
			ax2.text(bar2.get_x() + bar2.get_width() / 2., height2 + 0.001,
			         f'{height2:.2f}', ha='center', va='bottom', fontsize=10)

			# 在x轴下方添加样本数量
			ax2.text(i, -max(ensemble_mae + xgb_mae) * 0.05, f'n={count}',
			         ha='center', va='top', fontsize=10, color='gray')

		plt.tight_layout()
		plt.savefig('delay_severity_analysis.png', dpi=150, bbox_inches='tight')
		plt.show()

		# 3. 严重延误场景的详细分析
		if '严重延误 (>60分钟)' in analysis_results:
			print("\n严重延误场景详细分析:")
			severe_results = analysis_results['严重延误 (>60分钟)']
			print(f"  样本数量: {severe_results['样本数量']:,}")
			print(f"  集成模型MAE: {severe_results['Ensemble MAE']:.4f}")
			print(f"  XGBoost MAE: {severe_results['XGBoost MAE']:.4f}")
			print(f"  相对改进率: {severe_results['相对改进率 (%)']:.2f}%")

			if severe_results['相对改进率 (%)'] > 0:
				print(
					f"  结论: 在严重延误场景下，集成模型的改进最为明显（MAE降低约{severe_results['相对改进率 (%)']:.1f}%）。")
				print(f"        这具有重要的实际意义，因为准确预测严重延误对于运行控制中的应急决策价值最高。")

	def _visualize_uncertainty_analysis(self):
		"""可视化按不确定性分析结果"""
		if 'uncertainty_analysis' not in self.results:
			print("没有可用的不确定性分析数据")
			return

		uncertainty_analysis = self.results['uncertainty_analysis']
		analysis_results = uncertainty_analysis['results']

		if not analysis_results:
			print("不确定性分析结果为空")
			return

		# 创建不确定性分析可视化
		fig, axes = plt.subplots(1, 2, figsize=(16, 6))
		fig.suptitle('按不确定性分析：集成模型与简单平均法的性能对比', fontsize=18, fontweight='bold')

		# 1. 相对改进率柱状图
		ax1 = axes[0]

		groups = list(analysis_results.keys())
		improvement_rates = [analysis_results[group]['相对改进率 (%)'] for group in groups]
		mean_uncertainties = [analysis_results[group]['平均不确定性'] for group in groups]

		# 创建双轴图
		color = 'tab:blue'
		bars = ax1.bar(range(len(groups)), improvement_rates, color=color, alpha=0.7, edgecolor='black')
		ax1.set_xlabel('不确定性分组', fontsize=14)
		ax1.set_ylabel('相对改进率 (%)', fontsize=14, color=color)
		ax1.set_title('集成模型相对于简单平均法的MAE相对改进率', fontsize=16, fontweight='bold')
		ax1.set_xticks(range(len(groups)))
		ax1.set_xticklabels(groups, rotation=0, fontsize=12)
		ax1.tick_params(axis='y', labelcolor=color)
		ax1.grid(True, alpha=0.3, axis='y')

		# 在柱子上添加数值
		for bar, rate in zip(bars, improvement_rates):
			height = bar.get_height()
			ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
			         f'{rate:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

		# 添加不确定性折线图
		ax1b = ax1.twinx()
		color = 'tab:red'
		ax1b.plot(range(len(groups)), mean_uncertainties, color=color, marker='o', linewidth=2, markersize=8)
		ax1b.set_ylabel('平均不确定性', fontsize=14, color=color)
		ax1b.tick_params(axis='y', labelcolor=color)

		# 2. MAE对比和样本数量
		ax2 = axes[1]

		x = np.arange(len(groups))
		width = 0.35

		ensemble_mae = [analysis_results[group]['Ensemble MAE'] for group in groups]
		simple_avg_mae = [analysis_results[group]['简单平均法 MAE'] for group in groups]
		sample_counts = [analysis_results[group]['样本数量'] for group in groups]
		mae_diffs = [analysis_results[group]['MAE差值'] for group in groups]

		bars1 = ax2.bar(x - width / 2, ensemble_mae, width, label='集成模型', color='steelblue', alpha=0.8)
		bars2 = ax2.bar(x + width / 2, simple_avg_mae, width, label='简单平均法', color='salmon', alpha=0.8)

		ax2.set_xlabel('不确定性分组', fontsize=14)
		ax2.set_ylabel('MAE', fontsize=14)
		ax2.set_title('各不确定性分组的MAE对比', fontsize=16, fontweight='bold')
		ax2.set_xticks(x)
		ax2.set_xticklabels(groups, rotation=0, fontsize=12)
		ax2.legend(fontsize=12)
		ax2.grid(True, alpha=0.3, axis='y')

		# 在柱子上添加数值
		for i, (bar1, bar2, count, diff) in enumerate(zip(bars1, bars2, sample_counts, mae_diffs)):
			height1 = bar1.get_height()
			height2 = bar2.get_height()
			ax2.text(bar1.get_x() + bar1.get_width() / 2., height1 + 0.001,
			         f'{height1:.2f}', ha='center', va='bottom', fontsize=10)
			ax2.text(bar2.get_x() + bar2.get_width() / 2., height2 + 0.001,
			         f'{height2:.2f}', ha='center', va='bottom', fontsize=10)

			# 在x轴下方添加样本数量
			ax2.text(i, -max(ensemble_mae + simple_avg_mae) * 0.05, f'n={count}',
			         ha='center', va='top', fontsize=10, color='gray')

			# 添加MAE差值标记
			if diff > 0:
				ax2.text(i, max(height1, height2) + 0.005, f'Δ={diff:.3f}',
				         ha='center', va='bottom', fontsize=9, fontweight='bold', color='green')

		plt.tight_layout()
		plt.savefig('uncertainty_analysis.png', dpi=150, bbox_inches='tight')
		plt.show()

		# 3. 不确定性分析的详细解释
		if '高不确定性' in analysis_results:
			high_results = analysis_results['高不确定性']
			print("\n不确定性分析详细解释:")
			print(f"  在高不确定性样本组中:")
			print(f"  样本数量: {high_results['样本数量']:,}")
			print(f"  平均不确定性: {high_results['平均不确定性']:.4f}")
			print(f"  集成模型MAE: {high_results['Ensemble MAE']:.4f}")
			print(f"  简单平均法MAE: {high_results['简单平均法 MAE']:.4f}")
			print(f"  相对改进率: {high_results['相对改进率 (%)']:.2f}%")

	def save_feature_info_file(self, save_path='../saved_models/feature_info.pkl'):
		"""保存特征信息到文件"""
		feature_info = {
			'trainer_feature_info': self.feature_info,
			'scalers_info': {
				'seq_features': self.scalers.get('seq_features', []),
				'static_features': self.scalers.get('static_features', []),
				'seq_scaler_n_features_in': self.scalers.get('seq_scaler_n_features_in', 0),
				'static_scaler_n_features_in': self.scalers.get('static_scaler_n_features_in', 0)
			},
			'model_info': {
				'seq_length': self.seq_length,
				'lstm_input_sizes': {
					'seq_input_size': self.feature_info.get('seq_input_shape', (0, 0, 0))[
						2] if 'seq_input_shape' in self.feature_info else 0,
					'static_input_size': self.feature_info.get('static_input_shape', (0, 0))[
						1] if 'static_input_shape' in self.feature_info and self.feature_info[
						'static_input_shape'] is not None else 0
				}
			}
		}

		joblib.dump(feature_info, save_path)
		print(f"训练特征信息已保存到: {save_path}")

		# 打印详细特征信息
		print("\n" + "=" * 60)
		print("训练特征信息汇总:")
		print("=" * 60)
		print(f"时序特征数量: {self.feature_info.get('seq_features_count', 0)}")
		print(f"静态特征数量: {self.feature_info.get('static_features_count', 0)}")
		print(f"总数值特征数量: {self.feature_info.get('total_numeric_features', 0)}")

		seq_input_shape = self.feature_info.get('seq_input_shape', (0, 0, 0))
		static_input_shape = self.feature_info.get('static_input_shape', (0, 0))
		print(f"\n模型输入形状:")
		print(f"  时序特征: {seq_input_shape}")
		print(f"  静态特征: {static_input_shape}")

		print(f"\n标准化器特征数量:")
		print(f"  时序标准化器: {self.scalers.get('seq_scaler_n_features_in', 0)}")
		print(f"  静态标准化器: {self.scalers.get('static_scaler_n_features_in', 0)}")
		print("=" * 60)

		return feature_info


def main():
	"""主函数"""

	try:
		# 1. 数据加载和处理
		print("\n阶段1: 数据加载和处理")
		processor = EnhancedFlightDataProcessor(
			file_path='../data_set/flights_sample_3m.csv',
			sample_size=2500000
		)

		df = processor.load_data()
		df = processor.clean_data()
		df = processor.create_features()

		# 保存数据特征信息
		feature_info = processor.save_feature_info()

		# 2. 创建训练器
		print("\n阶段2: 创建训练器")
		trainer = HybridModelTrainer(processor, seq_length=10)

		# 3. 准备数据
		print("\n阶段3: 准备数据")
		data = trainer.prepare_data(test_size=0.2, val_size=0.1)

		# 4. 训练LSTM模型
		print("\n阶段4: 训练LSTM模型")
		lstm_model, lstm_history = trainer.train_lstm(data, epochs=15, patience=5)

		# 5. 训练XGBoost模型
		print("\n阶段5: 训练XGBoost模型")
		xgb_trainer = trainer.train_xgboost(data)

		# 6. 创建集成模型
		print("\n阶段6: 创建集成模型")
		ensemble_model = trainer.create_ensemble(data)

		# 7. 评估模型
		print("\n阶段7: 评估所有模型")
		results = trainer.evaluate_models(data)

		# 8. 可视化结果
		print("\n阶段8: 可视化结果")
		trainer.visualize_results(data)

		# 9. 保存模型
		print("\n阶段9: 保存模型")

		# 保存LSTM模型
		torch.save({
			'model_state_dict': lstm_model.state_dict(),
			'seq_length': trainer.seq_length,
			'seq_input_size': trainer.feature_info['seq_input_shape'][2],
			'static_input_size': trainer.feature_info['static_input_shape'][1] if trainer.feature_info[
				                                                                      'static_input_shape'] is not None else 0,
			'feature_info': trainer.feature_info
		}, '../saved_models/hybrid_lstm_model.pth')

		# 保存XGBoost模型 (xgb.Booster)
		xgb_trainer.model.save_model('../saved_models/optimized_xgboost_model.json')

		# 保存集成模型
		joblib.dump(ensemble_model, '../saved_models/ensemble_model.pkl')

		# 保存标准化器和特征信息
		joblib.dump(trainer.scalers, '../saved_models/scalers.pkl')

		# 保存完整的训练特征信息
		trainer.save_feature_info_file('../saved_models/feature_info.pkl')

		# 保存标签编码器
		joblib.dump(processor.label_encoders, '../saved_models/label_encoders.pkl')
		print(f"标签编码器已保存到: ../saved_models/label_encoders.pkl")
		print(f"包含的编码器: {list(processor.label_encoders.keys())}")

		# 保存结果
		results_df = pd.DataFrame.from_dict(results, orient='index')
		results_df.to_csv('../data_set/model_comparison_results.csv', encoding='utf8')

		# 保存t检验结果
		if 'T-Test' in results:
			t_test_df = pd.DataFrame([results['T-Test']])
			t_test_df.to_csv('../data_set/paired_t_test_results.csv', encoding='utf8')
			print("配对样本t检验结果已保存到 paired_t_test_results.csv")

		# 保存延误程度分析结果
		if 'Delay Severity Analysis' in results:
			delay_df = pd.DataFrame.from_dict(results['Delay Severity Analysis'], orient='index')
			delay_df.to_csv('../data_set/delay_severity_analysis.csv', encoding='utf8')
			print("延误程度分析结果已保存到 delay_severity_analysis.csv")

		# 保存不确定性分析结果
		if 'Uncertainty Analysis' in results:
			uncertainty_df = pd.DataFrame.from_dict(results['Uncertainty Analysis'], orient='index')
			uncertainty_df.to_csv('../data_set/uncertainty_analysis.csv', encoding='utf8')
			print("不确定性分析结果已保存到 uncertainty_analysis.csv")

		print("训练完成！")

		# 打印最终结果
		print("\n最终模型性能:")
		for model, metrics in results.items():
			if model in ['T-Test', 'Delay Severity Analysis', 'Uncertainty Analysis']:
				continue
			print(f"\n{model}:")
			for metric, value in metrics.items():
				print(f"  {metric}: {value:.4f}")

		# 检查集成模型是否优于单个模型
		if 'Ensemble' in results and 'LSTM' in results and 'XGBoost' in results:
			lstm_mae = results['LSTM']['MAE']
			xgb_mae = results['XGBoost']['MAE']
			ensemble_mae = results['Ensemble']['MAE']

			improvement_over_lstm = (lstm_mae - ensemble_mae) / lstm_mae * 100
			improvement_over_xgb = (xgb_mae - ensemble_mae) / xgb_mae * 100

			print(f"\n集成模型改进:")
			print(f"  相对于LSTM改进: {improvement_over_lstm:.2f}%")
			print(f"  相对于XGBoost改进: {improvement_over_xgb:.2f}%")

		# 打印t检验统计显著性
		if 'T-Test' in results:
			t_test = results['T-Test']
			print(f"\n配对样本t检验统计显著性:")
			if t_test['p_value'] < 0.05:
				print(f"  集成模型与XGBoost的预测误差有显著差异 (p = {t_test['p_value']:.6f})")
				if t_test['mean_difference'] < 0:
					print(f"  集成模型的预测误差显著小于XGBoost，性能提升具有统计显著性")
				else:
					print(f"  集成模型的预测误差显著大于XGBoost，性能下降具有统计显著性")
			else:
				print(f"  集成模型与XGBoost的预测误差无显著差异 (p = {t_test['p_value']:.6f})")

		# 打印延误程度分析总结
		if 'Delay Severity Analysis' in results:
			delay_analysis = results['Delay Severity Analysis']
			print(f"\n按延误程度分析总结:")
			for group, metrics in delay_analysis.items():
				print(f"  {group}:")
				print(f"    样本数量: {metrics['样本数量']}")
				print(f"    相对改进率: {metrics['相对改进率 (%)']:.2f}%")

		# 打印不确定性分析总结
		if 'Uncertainty Analysis' in results:
			uncertainty_analysis = results['Uncertainty Analysis']
			print(f"\n按不确定性分析总结:")
			for group, metrics in uncertainty_analysis.items():
				print(f"  {group}:")
				print(f"    样本数量: {metrics['样本数量']}")
				print(f"    相对改进率: {metrics['相对改进率 (%)']:.2f}%")

		return trainer, results

	except Exception as e:
		print(f"\n❌ 程序运行出错: {e}")
		import traceback
		traceback.print_exc()
		return None, None


if __name__ == "__main__":
	import os

	# 设置随机种子
	torch.manual_seed(42)
	np.random.seed(42)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(42)
		torch.cuda.manual_seed_all(42)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False

	trainer, results = main()