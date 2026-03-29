# test_api.py
import requests
import json

# API地址
BASE_URL = "http://localhost:5000"


def test_health():
    """测试健康检查"""
    response = requests.get(f"{BASE_URL}/health")
    print("健康检查:", response.json())
    return response.json()


def test_model_info():
    """测试模型信息"""
    response = requests.get(f"{BASE_URL}/model_info")
    print("模型信息:", response.json())
    return response.json()


def test_predict():
    """测试单条预测"""

    flight_data = {
        "flight_date": "2024-01-15",
        "airline_code": "AA",
        "flight_number": "1234",
        "origin": "JFK",
        "destination": "LAX",
        "scheduled_departure_time": 830,
        "scheduled_arrival_time": 1130,
        "distance": 2475
    }

    response = requests.post(
        f"{BASE_URL}/predict",
        json=flight_data
    )

    print("\n单条预测结果:")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    return response.json()


def test_batch_predict():
    """测试批量预测"""

    batch_data = {
        "flights": [
            {
                "flight_date": "2024-01-15",
                "airline_code": "AA",
                "flight_number": "1234",
                "origin": "JFK",
                "destination": "LAX",
                "scheduled_departure_time": 830,
                "scheduled_arrival_time": 1130,
                "distance": 2475
            },
            {
                "flight_date": "2024-01-15",
                "airline_code": "UA",
                "flight_number": "5678",
                "origin": "SFO",
                "destination": "ORD",
                "scheduled_departure_time": 930,
                "scheduled_arrival_time": 1530,
                "distance": 1846
            }
        ]
    }

    response = requests.post(
        f"{BASE_URL}/predict_batch",
        json=batch_data
    )

    print("\n批量预测结果:")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    return response.json()


def test_predict_example():
    """测试示例预测"""
    response = requests.get(f"{BASE_URL}/predict_example")
    print("\n示例预测:")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    return response.json()


if __name__ == '__main__':
    print("开始测试API...")

    # 测试所有接口
    test_health()
    test_model_info()
    test_predict()
    test_batch_predict()
    test_predict_example()

    print("\n测试完成!")