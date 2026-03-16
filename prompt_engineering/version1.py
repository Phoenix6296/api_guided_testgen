# Example output from Initial Basic Prompt (Version 1)
# API: tf.keras.layers.Dense
# System: You are a helpful AI assistant.
# User: Generate unit tests for tf.keras.layers.Dense

import tensorflow as tf

def test_dense():
    layer = tf.keras.layers.Dense(10)
    assert layer.units == 10

def test_dense_activation():
    layer = tf.keras.layers.Dense(5, activation='relu')
    assert layer.activation == tf.keras.activations.relu

# Run tests
test_dense()
test_dense_activation()

# Issues with Version 1:
# - No proper test structure (missing unittest framework)
# - No main function for execution
# - Basic assertions only
# - Limited test coverage
# - No imports organization
# - Direct function calls instead of test runner