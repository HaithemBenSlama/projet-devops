from django.test import TestCase

from django.test import TestCase, Client
from django.urls import reverse
import json

class BasicAIAPITestCase(TestCase):
    def setUp(self):
        self.client = Client()

    def test_api_returns_empty_list(self):
        
        response = self.client.get("http://127.0.0.1:8000/api/models/")

        # Assert status code is 200 (OK)
        self.assertEqual(response.status_code, 200)

        # Parse response content as JSON
        data = json.loads(response.content)

        # Assert that the response contains an empty list
        self.assertEqual(data, [])
        
from .models import AIModel

class ModelCreationTestCase(TestCase):
    def test_ai_model_creation(self):
        ai_model = AIModel.objects.create(
            name="Test Model",
            description="A test model description",
            num_classes=10,
            accuracy=99.99,
            macro_avg=95.00,
            wieghted_avg=96.00,
            architecture={"layers": 5, "type": "CNN"},
            dataset="MNIST"
        )
        self.assertEqual(ai_model.name, "Test Model")
        self.assertEqual(ai_model.num_classes, 10)
        self.assertEqual(float(ai_model.accuracy), 99.99)

class AIModelAPITestCase(TestCase):
    def setUp(self):
        self.ai_model = AIModel.objects.create(name="Test Model 2", num_classes=3)

    def test_model_list(self):
        response = self.client.get(reverse('model-list'))
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]['name'], "Test Model 2")

class ModelFieldValidationTestCase(TestCase):
    def test_invalid_accuracy(self):
        with self.assertRaises(ValueError):
            AIModel.objects.create(name="Invalid Model", accuracy=101.00)
