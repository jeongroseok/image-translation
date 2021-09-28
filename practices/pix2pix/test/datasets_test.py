import datasets
import pl_examples
import unittest


class Datasets(unittest.TestCase):
    def __init__(self, methodName: str) -> None:
        super().__init__(methodName=methodName)
        self.trainset = datasets.Facades(pl_examples._DATASETS_PATH, 'train')
        self.valset = datasets.Facades(pl_examples._DATASETS_PATH, 'val')
        self.testset = datasets.Facades(pl_examples._DATASETS_PATH, 'test')

    def test_trainset(self):
        self.assertEqual(len(self.trainset), 400)

    def test_valset(self):
        self.assertEqual(len(self.valset), 100)

    def test_testset(self):
        self.assertEqual(len(self.testset), 106)

    def test_item(self):
        input, target = self.trainset[0]
        self.assertEqual(input.shape, target.shape)
