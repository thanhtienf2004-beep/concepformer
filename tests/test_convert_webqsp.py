import unittest
from src.GraphAligner.BigGraphAligner import ConceptFormerModel

class TestConceptFormerModel(unittest.TestCase):
    def setUp(self):
        self.model = ConceptFormerModel()

    def test_train_with_matching_batch_sizes(self):
        train_data = ...  # Provide appropriate mock data
        optimizer_cf = ...  # Provide appropriate mock optimizer
        device = ...  # Specify the device
        embeddings = ...  # Provide appropriate mock embeddings
        
        outputs, labels = self.model.train(train_data, optimizer_cf, device, embeddings)
        self.assertEqual(outputs.size(0), labels.size(0))

    def test_train_with_empty_input(self):
        train_data = []  # Empty input
        optimizer_cf = ...  # Provide appropriate mock optimizer
        device = ...  # Specify the device
        embeddings = ...  # Provide appropriate mock embeddings
        
        with self.assertRaises(ValueError):
            self.model.train(train_data, optimizer_cf, device, embeddings)

if __name__ == '__main__':
    unittest.main()