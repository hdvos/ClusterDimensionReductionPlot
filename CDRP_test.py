import unittest
import CDRP

class TestInit(unittest.TestCase):
    def test_init(self):
        with self.assertRaises(ValueError) as context:
            CDRP.ClusterDimRedPLot(dtm_type='xxx')

        self.assertTrue('Unknown dtm type' in context.exception)
        # self.assertRaises(ValueError, )
        
        self.assertTrue(CDRP.ClusterDimRedPLot(dtm_type='count'))

if __name__ == "__main__":
    unittest.main()