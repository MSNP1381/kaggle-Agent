import unittest
from unittest.mock import MagicMock
from data_utils import DataUtils
from states.main import KaggleProblemState
from langchain_openai import ChatOpenAI

class TestDataUtils(unittest.TestCase):
    def setUp(self):
        # Mock the LLM
        self.mock_llm = MagicMock(spec=ChatOpenAI)
        # Mock the response from the LLM
        self.mock_llm.invoke.return_value = {
            "quantitative_analysis": "Quantitative analysis mock result.",
            "qualitative_analysis": "Qualitative analysis mock result.",
            "feature_recommendations": ["Feature1", "Feature2"]
        }
        # Initialize DataUtils with mocked LLM
        self.data_utils = DataUtils(config={}, proxy=None, llm=self.mock_llm)
    
    def test_analyze_dataset(self):
        # Create a sample dataset
        sample_data = pd.DataFrame({
            'LotArea': [8450, 9600, 11250],
            'OverallQual': [7, 6, 7],
            'YearBuilt': [2003, 1976, 2001]
        })
        dataset_path = "dummy_path.csv"
        sample_data.to_csv(dataset_path, index=False)
        
        # Create a KaggleProblemState instance
        state = KaggleProblemState(
            index=-1,
            problem_description="Predict house prices.",
            dataset_path=dataset_path,
            evaluation_metric="RMSE"
        )
        
        # Invoke DataUtils
        result = self.data_utils(state)
        
        # Assertions
        self.assertIn("dataset_info", result)
        dataset_info = json.loads(result["dataset_info"])
        self.assertEqual(dataset_info["quantitative_analysis"], "Quantitative analysis mock result.")
        self.assertEqual(dataset_info["qualitative_analysis"], "Qualitative analysis mock result.")
        self.assertEqual(dataset_info["feature_recommendations"], ["Feature1", "Feature2"])
        
        # Clean up
        os.remove(dataset_path)

if __name__ == "__main__":
    unittest.main()