"""
Test script for Wine Rating Prediction API
==========================================
"""

import requests
import json



def test_health_check(BASE_URL):
    """Test the health check endpoint"""
    print("\n🔍 Testing health check...")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_single_prediction(BASE_URL):
    """Test single wine prediction"""
    print("\n🍷 Testing single prediction...")

    # Example wine (typical Bordeaux characteristics)
    wine_data = {
        "vintage_year": 2018,
        "structure_acidity": 3.5,
        "structure_tannin": 3.2,
        "region": "Bordeaux"
    }

    response = requests.post(f"{BASE_URL}/predict", json=wine_data)

    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"\n✨ Prediction Results:")
        print(f"   Predicted Rating: {result['predicted_rating']}")
        print(f"   Category: {result['rating_class']}")
    else:
        print(f"Error: {response.text}")


def test_batch_prediction(BASE_URL):
    """Test batch prediction with multiple wines"""
    print("\n🍷🍷 Testing batch prediction...")

    wines = [
        {
            "vintage_year": 2018,
            "structure_acidity": 3.5,
            "structure_tannin": 3.2,
            "region": "vin-de-pays-vignobles-de-france"
        },
        {
            "vintage_year": 2020,
            "structure_acidity": 1.2,
            "structure_tannin": 2.8,
            "region": "haut-medoc"
        },
        {
            "vintage_year": 1983,
            "structure_acidity": 1.2,
            "structure_tannin": 3.8,
            "region": "pauillac"
        }
    ]

    response = requests.post(f"{BASE_URL}/predict_batch", json=wines)

    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        results = response.json()
        print(f"\n✨ Batch Predictions ({len(results)} wines):")
        for i, result in enumerate(results, 1):
            print(f"\n   Wine {i}:")
            print(f"      Rating: {result['predicted_rating']}")
            print(f"      Category: {result['rating_class']}")
    else:
        print(f"Error: {response.text}")


def test_invalid_input(BASE_URL):
    """Test API with invalid input"""
    print("\n❌ Testing invalid input handling...")

    # Missing required field
    invalid_wine = {
        "vintage_year": 2018,
        "structure_acidity": 3.5
        # Missing structure_tannin and region
    }

    response = requests.post(f"{BASE_URL}/predict", json=invalid_wine)
    print(f"Status: {response.status_code}")
    print("Expected 422 (Validation Error)")
    print(f"Response: {response.text}")


if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Wine Rating API Test Suite")
    print("NOTE YOU NEED TO RUN first Lot4_websevice_predict.py ")
    print("Or 3_serve.py or have a docker running or an api endpoint ect   ")
    print("=" * 60)

    # API endpoint
    BASE_URL = "http://0.0.0.0:7860"

    try:
        # Run all tests
        test_health_check(BASE_URL)
        test_single_prediction(BASE_URL)
        test_batch_prediction(BASE_URL)
        test_invalid_input(BASE_URL)

        print("\n" + "=" * 60)
        print("✅ All tests completed!")
        print("=" * 60)

    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to API")
        print(f"Make sure the API is running on {BASE_URL}")
        print("Run: python wine_rating_api.py")
