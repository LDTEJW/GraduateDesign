import torch
import joblib
import xgboost as xgb
import numpy as np


class HybridLSTM(torch.nn.Module):
    def __init__(self, seq_input_size, static_input_size=0, lstm_hidden_size=64, num_lstm_layers=1, dropout=0.2):
        super(HybridLSTM, self).__init__()
        self.lstm = torch.nn.LSTM(
            input_size=seq_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
            bidirectional=True
        )
        if static_input_size > 0:
            self.static_encoder = torch.nn.Sequential(
                torch.nn.Linear(static_input_size, 128),
                torch.nn.BatchNorm1d(128),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(128, 64),
                torch.nn.BatchNorm1d(64),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout)
            )
            static_output_size = 64
        else:
            self.static_encoder = None
            static_output_size = 0
        lstm_output_size = lstm_hidden_size * 2
        fusion_input_size = lstm_output_size + static_output_size
        self.fusion = torch.nn.Sequential(
            torch.nn.Linear(fusion_input_size, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(256, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(128, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout * 0.5)
        )
        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout * 0.3),
            torch.nn.Linear(32, 1)
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
                X_seq_batch = torch.FloatTensor(X_seq[i:end]).to(self.device)
                if X_static is not None:
                    X_static_batch = torch.FloatTensor(X_static[i:end]).to(self.device)
                    pred = self.lstm_model(X_seq_batch, X_static_batch).cpu().numpy()
                else:
                    pred = self.lstm_model(X_seq_batch, None).cpu().numpy()
            predictions.append(pred)
        return np.vstack(predictions)

    def _predict_xgb(self, X_xgb):
        dmatrix = xgb.DMatrix(X_xgb)
        return self.xgb_model.predict(dmatrix).reshape(-1, 1)


def load_all_models(model_dir='../train/saved_models'):
    # 加载特征信息
    feature_info = joblib.load(f'{model_dir}/feature_info.pkl')
    # 兼容嵌套结构
    if 'trainer_feature_info' in feature_info:
        feature_info = feature_info['trainer_feature_info']
    # 确保 seq_length 存在
    if 'seq_length' not in feature_info:
        feature_info['seq_length'] = 10

    scalers = joblib.load(f'{model_dir}/scalers.pkl')
    label_encoders = joblib.load(f'{model_dir}/label_encoders.pkl')

    # 加载LSTM
    lstm_checkpoint = torch.load(f'{model_dir}/hybrid_lstm_model.pth', map_location='cpu')
    seq_input_size = lstm_checkpoint.get('seq_input_size', 0)
    static_input_size = lstm_checkpoint.get('static_input_size', 0)
    lstm_model = HybridLSTM(
        seq_input_size=seq_input_size,
        static_input_size=static_input_size,
        lstm_hidden_size=64,
        num_lstm_layers=1,
        dropout=0.2
    )
    lstm_model.load_state_dict(lstm_checkpoint['model_state_dict'])
    lstm_model.eval()

    # 加载XGBoost
    xgb_model = xgb.Booster()
    xgb_model.load_model(f'{model_dir}/optimized_xgboost_model.json')

    # 加载集成元模型
    ensemble_meta = joblib.load(f'{model_dir}/ensemble_model.pkl')
    ensemble_wrapper = EnsembleModelWrapper(lstm_model, xgb_model, ensemble_meta.meta_model)

    return {
        'lstm': lstm_model,
        'xgb': xgb_model,
        'ensemble': ensemble_wrapper,
        'scalers': scalers,
        'label_encoders': label_encoders,
        'feature_info': feature_info
    }