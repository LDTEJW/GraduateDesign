import os
import io
import base64
import tempfile
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import warnings
warnings.filterwarnings('ignore')

from utils.predictor import FlightDelayPredictor,EnsembleModel,HybridLSTM

# 获取当前文件所在目录的绝对路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, 
            static_folder=os.path.join(BASE_DIR, 'static'),
            template_folder=os.path.join(BASE_DIR, 'templates'))
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024 * 1024  # 1GB
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'csv'}

predictor = None
# 缓存 FontProperties；
_CHART_FP_UNSET = object()
_chart_fontproperties_cache = _CHART_FP_UNSET


def _json_float(x, ndigits=2):
    """JSON 不支持 NaN/Inf，统一转为有限 float 或 None。"""
    try:
        v = float(x)
        if not np.isfinite(v):
            return None
        return round(v, ndigits) if ndigits is not None else float(v)
    except (TypeError, ValueError):
        return None


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init_predictor():
    global predictor
    try:
        predictor = FlightDelayPredictor(model_dir='../train/saved_models')
        print("预测器初始化成功")
        return True
    except Exception as e:
        print(f"预测器初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False

@app.errorhandler(Exception)
def handle_exception(e):
    import traceback
    traceback.print_exc()
    return jsonify({'error': f'服务器错误：{str(e)}'}), 500

@app.errorhandler(404)
def not_found(e):
    """处理 404 错误，返回空响应避免浏览器重试"""
    return '', 204

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': '文件过大，请上传小于 1GB 的 CSV 文件'}), 413


def _get_chart_fontproperties():
    """
    返回绑定到具体字体文件的 FontProperties。
    """
    global _chart_fontproperties_cache
    if _chart_fontproperties_cache is not _CHART_FP_UNSET:
        return _chart_fontproperties_cache

    from matplotlib import font_manager as fm

    plt.rcParams['axes.unicode_minus'] = False

    candidate_paths = []
    env_font = os.environ.get('FLIGHT_DELAY_CHART_FONT', '').strip()
    if env_font and os.path.isfile(env_font):
        candidate_paths.append(env_font)

    web_root = os.path.dirname(os.path.abspath(__file__))
    for rel in (
        'static/fonts/NotoSansSC-Regular.otf',
        'static/fonts/NotoSansSC-VF.ttf',
        'static/fonts/SourceHanSansCN-Regular.otf',
    ):
        p = os.path.join(web_root, rel)
        if os.path.isfile(p):
            candidate_paths.append(p)

    if os.name == 'nt':
        fd = os.path.join(os.environ.get('WINDIR', r'C:\Windows'), 'Fonts')
        for fname in (
            'msyh.ttc', 'msyhbd.ttc', 'simhei.ttf', 'simsun.ttc',
            'simkai.ttf', 'msyh.ttf', 'STXIHEI.TTF', 'msjhl.ttc',
        ):
            p = os.path.join(fd, fname)
            if os.path.isfile(p):
                candidate_paths.append(p)

    for linux in (
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
    ):
        if os.path.isfile(linux):
            candidate_paths.append(linux)

    for path in candidate_paths:
        try:
            fp = fm.FontProperties(fname=path)
            _ = fp.get_name()
            _chart_fontproperties_cache = fp
            return fp
        except Exception:
            continue

    for family in (
        'Microsoft YaHei', 'Microsoft YaHei UI', 'SimHei', 'SimSun',
        'PingFang SC', 'Heiti SC', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC',
    ):
        path = fm.findfont(fm.FontProperties(family=family))
        if path and 'dejavu' not in path.lower():
            try:
                fp = fm.FontProperties(fname=path)
                _chart_fontproperties_cache = fp
                return fp
            except Exception:
                continue

    _chart_fontproperties_cache = None
    return None


def _apply_text_font(ax, fp, title, xlabel, ylabel, legend=True):
    """为当前坐标系所有中文文本绑定字体。"""
    if fp is not None:
        ax.set_title(title, fontproperties=fp)
        ax.set_xlabel(xlabel, fontproperties=fp)
        ax.set_ylabel(ylabel, fontproperties=fp)
        plt.setp(ax.get_xticklabels(), fontproperties=fp)
        plt.setp(ax.get_yticklabels(), fontproperties=fp)
        if legend:
            leg = ax.get_legend()
            if leg is not None:
                for t in leg.get_texts():
                    t.set_fontproperties(fp)
    else:
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)


def generate_charts(y_true, y_pred, model_name='集成模型'):
    charts = {}
    fp = _get_chart_fontproperties()
    if fp is not None:
        plt.rcParams['font.sans-serif'] = [fp.get_name()] + list(plt.rcParams['font.sans-serif'])
    sns.set_style('whitegrid')
    _save_kw = dict(format='png', dpi=100, bbox_inches='tight', pad_inches=0.45)

    # 散点图
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, alpha=0.5, c='steelblue', edgecolors='white', linewidth=0.5)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='完美预测')
    if fp is not None:
        ax.legend(prop=fp)
    else:
        ax.legend()
    _apply_text_font(
        ax, fp,
        f'{model_name} - 预测 vs 实际',
        '实际延误 (分钟)',
        '预测延误 (分钟)',
        legend=False,
    )
    if fp is not None:
        leg = ax.get_legend()
        if leg is not None:
            for t in leg.get_texts():
                t.set_fontproperties(fp)
    buf = io.BytesIO()
    plt.savefig(buf, **_save_kw)
    buf.seek(0)
    charts['scatter'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()

    # 误差分布
    errors = y_true - y_pred
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(errors, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='零误差')
    ax.axvline(x=np.mean(errors), color='green', linestyle='-', linewidth=2, label=f'均值: {np.mean(errors):.2f}')
    if fp is not None:
        ax.legend(prop=fp)
    else:
        ax.legend()
    _apply_text_font(ax, fp, f'{model_name} - 误差分布', '预测误差 (分钟)', '频数', legend=False)
    if fp is not None:
        leg = ax.get_legend()
        if leg is not None:
            for t in leg.get_texts():
                t.set_fontproperties(fp)
    buf = io.BytesIO()
    plt.savefig(buf, **_save_kw)
    buf.seek(0)
    charts['error_dist'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()

    # 箱线图
    groups = []
    errors_by_group = []
    for name, mask in [('轻微/提前 (-15~15)', (y_true >= -15) & (y_true <= 15)),
                       ('中等延误 (15~60)', (y_true > 15) & (y_true <= 60)),
                       ('严重延误 (>60)', y_true > 60)]:
        if mask.sum() > 0:
            groups.append(name)
            errors_by_group.append(errors[mask])
    if errors_by_group:
        fig, ax = plt.subplots(figsize=(10, 6))
        bp = ax.boxplot(errors_by_group, labels=groups, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
        _apply_text_font(ax, fp, '不同延误程度下的预测误差', '延误程度分组', '预测误差 (分钟)', legend=False)
        if fp is not None:
            plt.setp(ax.get_xticklabels(), fontproperties=fp, rotation=10)
        buf = io.BytesIO()
        plt.savefig(buf, **_save_kw)
        buf.seek(0)
        charts['error_box'] = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
    return charts

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    """返回空的 favicon，避免 404 错误"""
    return '', 204

@app.route('/predict', methods=['POST'])
def predict():
    if predictor is None:
        return jsonify({'error': '预测器未初始化，请检查服务端日志'}), 500
    if 'file' not in request.files:
        return jsonify({'error': '请上传文件'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '请选择文件'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': '仅支持CSV文件'}), 400

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        df = pd.read_csv(filepath, low_memory=False)
        os.remove(filepath)

        required = ['FL_DATE', 'AIRLINE_CODE', 'ORIGIN', 'DEST', 'CRS_DEP_TIME', 'DEP_DELAY']
        missing = [c for c in required if c not in df.columns]
        if missing:
            return jsonify({'error': f'缺少必要列: {missing}'}), 400

        # 获取预测值（predictor 已对无窗口样本做逐行回退；非有限值不参与指标）
        y_pred = np.asarray(predictor.predict_dataframe(df), dtype=np.float64)

        # 获取真实值（如果存在）
        y_true = None
        if 'ARR_DELAY' in df.columns:
            y_true = pd.to_numeric(df['ARR_DELAY'], errors='coerce').values[:len(y_pred)]
        elif 'arrival_delay' in df.columns:
            y_true = pd.to_numeric(df['arrival_delay'], errors='coerce').values[:len(y_pred)]

        # 计算指标（过滤 NaN）
        metrics = {}
        if y_true is not None and len(y_true) == len(y_pred):
            # 有效样本：真实值与预测值均为有限数（避免将旧逻辑中 NaN→0 与真实延误对比拉低 R²）
            valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
            y_true_clean = y_true[valid_mask]
            y_pred_clean = y_pred[valid_mask]
            if len(y_true_clean) > 0:
                from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
                mae = mean_absolute_error(y_true_clean, y_pred_clean)
                rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
                r2 = r2_score(y_true_clean, y_pred_clean)
                metrics = {
                    'MAE': _json_float(mae),
                    'RMSE': _json_float(rmse),
                    'R2': _json_float(r2),
                    '样本数': int(len(y_true_clean)),
                }
            else:
                metrics = {'预测样本数': len(y_pred), '说明': '有效样本数为0，无法计算指标'}
        else:
            metrics = {'预测样本数': len(y_pred), '说明': '数据中未包含ARR_DELAY列或长度不匹配'}

        # 生成图表（同样过滤 NaN）
        charts = {}
        if y_true is not None and len(y_true) > 0:
            valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
            y_true_clean = y_true[valid_mask]
            y_pred_clean = y_pred[valid_mask]
            if len(y_true_clean) > 0:
                charts = generate_charts(y_true_clean, y_pred_clean, '集成模型')

        # 准备表格数据（前50条）
        table_data = []
        for i in range(min(50, len(y_pred))):
            row = {'序号': i + 1, '预测延误(分钟)': _json_float(y_pred[i])}
            if y_true is not None and i < len(y_true) and np.isfinite(y_true[i]):
                row['实际延误(分钟)'] = _json_float(y_true[i])
                err = float(y_true[i]) - float(y_pred[i])
                row['误差(分钟)'] = _json_float(err)
            table_data.append(row)

        # 摘要统计（对全为 NaN 等情况使用 nanmean/nanmin 仍可能得到 nan，需兜底）
        finite_pred = y_pred[np.isfinite(y_pred)]
        if len(finite_pred) > 0:
            summary = {
                'total_predictions': int(len(y_pred)),
                'avg_pred_delay': _json_float(np.mean(finite_pred)),
                'std_pred_delay': _json_float(np.std(finite_pred)),
                'min_pred_delay': _json_float(np.min(finite_pred)),
                'max_pred_delay': _json_float(np.max(finite_pred)),
            }
        else:
            summary = {
                'total_predictions': int(len(y_pred)),
                'avg_pred_delay': None,
                'std_pred_delay': None,
                'min_pred_delay': None,
                'max_pred_delay': None,
            }

        return jsonify({
            'success': True,
            'metrics': metrics,
            'charts': charts,
            'table_data': table_data,
            'summary': summary
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'预测失败: {str(e)}'}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'predictor_ready': predictor is not None})

if __name__ == '__main__':
    if init_predictor():
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        print("预测器初始化失败，请检查模型文件")