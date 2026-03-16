# Example output from Improved Structure Prompt (Version 2)
# API: tf.keras.layers.Dense
# System: You are a unit test generator for tensorflow library.
# User: Generate a python unit test case to test tf.keras.layers.Dense API. Use unittest library.

import unittest
import tensorflow as tf

class TestDense(unittest.TestCase):
    
    def test_dense_creation(self):
        layer = tf.keras.layers.Dense(10)
        self.assertEqual(layer.units, 10)
    
    def test_dense_activation(self):
        layer = tf.keras.layers.Dense(5, activation='relu')
        self.assertEqual(layer.activation, tf.keras.activations.relu)
    
    def test_dense_bias(self):
        layer = tf.keras.layers.Dense(3, use_bias=False)
        self.assertFalse(layer.use_bias)

if __name__ == '__main__':
    unittest.main()

# Improvements in Version 2:
# - Added unittest framework structure
# - Proper test class with inheritance
# - Better assertions using unittest methods
# - Main function for execution
# - More organized structure
# 
# Remaining issues:
# - No coverage guidance
# - Limited test scenarios
# - Missing "maximum coverage" instruction