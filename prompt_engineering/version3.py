# Example output from Final Zero-Shot Prompt (Version 3)
# API: tf.keras.layers.Dense
# System: You are a unit test suite generator for tensorflow library.
# User: Generate a python unit test case to test the functionality of tf.keras.layers.Dense API in tensorflow library with maximum coverage. Only create new tests if they cover new lines of code. Generate test suite using unittest library so it can be directly runnable (with the necessary imports and a main function).

import unittest
import tensorflow as tf
import numpy as np

class TestDense(unittest.TestCase):
    
    def test_dense_layer_creation(self):
        """Test basic Dense layer creation"""
        layer = tf.keras.layers.Dense(10)
        self.assertEqual(layer.units, 10)
        self.assertTrue(layer.use_bias)
    
    def test_dense_layer_call(self):
        """Test Dense layer forward pass"""
        layer = tf.keras.layers.Dense(5)
        input_data = tf.random.normal((2, 3))
        output = layer(input_data)
        self.assertEqual(output.shape, (2, 5))
    
    def test_dense_layer_with_activation(self):
        """Test Dense layer with activation function"""
        layer = tf.keras.layers.Dense(4, activation='relu')
        input_data = tf.random.normal((1, 3))
        output = layer(input_data)
        self.assertEqual(output.shape, (1, 4))
    
    def test_dense_layer_without_bias(self):
        """Test Dense layer without bias"""
        layer = tf.keras.layers.Dense(3, use_bias=False)
        self.assertFalse(layer.use_bias)
    
    def test_dense_layer_with_different_units(self):
        """Test Dense layer with different number of units"""
        for units in [1, 10, 100]:
            layer = tf.keras.layers.Dense(units)
            self.assertEqual(layer.units, units)
    
    def test_dense_layer_error_handling(self):
        """Test error cases"""
        with self.assertRaises(ValueError):
            tf.keras.layers.Dense(0)

if __name__ == '__main__':
    unittest.main()

# Final improvements in Version 3:
# - Added "maximum coverage" focus
# - Included "only create new tests if they cover new lines" constraint
# - Added numpy import for completeness
# - More comprehensive test scenarios
# - Better docstrings
# - Added error handling test
# - Emphasized "directly runnable" requirement
# 
# This is the final zero-shot baseline prompt used in the research