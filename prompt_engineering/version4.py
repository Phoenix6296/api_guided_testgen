# Example output from RAG-Enhanced Prompt (Version 4)
# API: tf.keras.layers.Dense
# System: You are a unit test suite generator for tensorflow library.
# User: Generate a python unit test case to test the functionality of tf.keras.layers.Dense API in tensorflow library with maximum coverage. Use the following documents (surrounded by @@@) to make the test case more compilable, and passable, and cover more lines. Only create new tests if they cover new lines of code. Generate test suite using unittest library so it can be directly runnable (with the necessary imports and a main function).
# @@@ Doc_1: {API documentation} @@@ Doc_2: {GitHub issues} @@@ Doc_3: {StackOverflow discussions} @@@

import unittest
import tensorflow as tf
import numpy as np

class TestDense(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        tf.random.set_seed(42)
    
    def test_dense_layer_creation(self):
        """Test basic Dense layer creation"""
        layer = tf.keras.layers.Dense(10)
        self.assertEqual(layer.units, 10)
        self.assertTrue(layer.use_bias)
        self.assertEqual(layer.activation, tf.keras.activations.linear)
    
    def test_dense_layer_call(self):
        """Test Dense layer forward pass"""
        layer = tf.keras.layers.Dense(5)
        input_data = tf.random.normal((2, 3))
        output = layer(input_data)
        
        self.assertEqual(output.shape, (2, 5))
        self.assertEqual(output.dtype, tf.float32)
        self.assertTrue(tf.reduce_all(tf.math.is_finite(output)))
    
    def test_dense_layer_with_activations(self):
        """Test Dense layer with different activation functions"""
        activations = ['relu', 'tanh', 'sigmoid', 'softmax']
        for activation in activations:
            layer = tf.keras.layers.Dense(3, activation=activation)
            input_data = tf.random.normal((1, 4))
            output = layer(input_data)
            
            self.assertEqual(output.shape, (1, 3))
            if activation == 'softmax':
                # Softmax outputs should sum to 1
                self.assertAlmostEqual(tf.reduce_sum(output).numpy(), 1.0, places=5)
    
    def test_dense_layer_without_bias(self):
        """Test Dense layer without bias"""
        layer = tf.keras.layers.Dense(3, use_bias=False)
        input_data = tf.random.normal((1, 2))
        output = layer(input_data)
        
        self.assertFalse(layer.use_bias)
        self.assertEqual(output.shape, (1, 3))
    
    def test_dense_layer_weight_initialization(self):
        """Test Dense layer with custom weight initialization"""
        layer = tf.keras.layers.Dense(
            4, 
            kernel_initializer='he_normal',
            bias_initializer='zeros'
        )
        input_data = tf.random.normal((1, 3))
        _ = layer(input_data)  # Build layer
        
        self.assertEqual(layer.kernel.shape, (3, 4))
        self.assertTrue(tf.reduce_all(tf.equal(layer.bias, 0.0)))
    
    def test_dense_layer_regularization(self):
        """Test Dense layer with regularization"""
        layer = tf.keras.layers.Dense(
            3,
            kernel_regularizer=tf.keras.regularizers.l2(0.01),
            bias_regularizer=tf.keras.regularizers.l1(0.01)
        )
        input_data = tf.random.normal((1, 2))
        output = layer(input_data)
        
        self.assertEqual(output.shape, (1, 3))
        self.assertGreater(len(layer.losses), 0)  # Should have regularization losses
    
    def test_dense_layer_batch_sizes(self):
        """Test Dense layer with different batch sizes"""
        layer = tf.keras.layers.Dense(4)
        
        batch_sizes = [1, 16, 32, 128]
        for batch_size in batch_sizes:
            input_data = tf.random.normal((batch_size, 5))
            output = layer(input_data)
            self.assertEqual(output.shape, (batch_size, 4))
    
    def test_dense_layer_error_conditions(self):
        """Test Dense layer error handling"""
        # Test invalid units
        with self.assertRaises(ValueError):
            tf.keras.layers.Dense(0)
        
        with self.assertRaises(ValueError):
            tf.keras.layers.Dense(-1)
    
    def test_dense_layer_serialization(self):
        """Test Dense layer serialization"""
        layer = tf.keras.layers.Dense(5, activation='relu')
        config = layer.get_config()
        
        # Verify config contains expected keys
        self.assertIn('units', config)
        self.assertIn('activation', config)
        self.assertEqual(config['units'], 5)
        
        # Test reconstruction
        new_layer = tf.keras.layers.Dense.from_config(config)
        self.assertEqual(new_layer.units, 5)

if __name__ == '__main__':
    unittest.main()

# RAG-Enhanced improvements in Version 4:
# - More comprehensive test coverage from retrieved documentation
# - Advanced features testing (regularization, weight initialization)
# - Better error handling and edge cases from GitHub issues
# - Real-world usage patterns from StackOverflow discussions
# - Context-informed parameter choices
# - More sophisticated assertions and validations
# - setUp method for reproducible testing
# - Serialization testing from documentation examples