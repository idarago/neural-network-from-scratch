import tensorflow as tf
import numpy as np
import pandas as pd

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train , x_test = x_train/255.0 , x_test/255.0

NUMBER_OF_TRAINING_DATA = len(x_train)
NUMBER_OF_TEST_DATA = len(x_test)

def generate_x_train():
    cpp_readable_format = [str(NUMBER_OF_TRAINING_DATA), "\n"]
    for k in range(NUMBER_OF_TRAINING_DATA):
        for i in range(28):
            for j in range(28):
               cpp_readable_format.append(str(x_train[k][i][j]))
               cpp_readable_format.append("\n")

    text_file = open("mnist_train.txt", "w")
    text_file.write("".join(cpp_readable_format[:-1]))
    text_file.close()

def generate_y_train():
    cpp_readable_format = [str(NUMBER_OF_TRAINING_DATA), "\n"]
    for k in range(NUMBER_OF_TRAINING_DATA):
        cpp_readable_format.append(str(y_train[k]))
        cpp_readable_format.append("\n")

    text_file = open("mnist_train_labels.txt", "w")
    text_file.write("".join(cpp_readable_format[:-1]))
    text_file.close()

def generate_x_test():
    cpp_readable_format = [str(NUMBER_OF_TEST_DATA), "\n"]
    for k in range(NUMBER_OF_TRAINING_DATA):
        for i in range(28):
            for j in range(28):
               cpp_readable_format.append(str(x_train[k][i][j]))
               cpp_readable_format.append("\n")

    text_file = open("mnist_test.txt", "w")
    text_file.write("".join(cpp_readable_format[:-1]))
    text_file.close()

def generate_y_test():
    cpp_readable_format = [str(NUMBER_OF_TEST_DATA), "\n"]
    for k in range(NUMBER_OF_TRAINING_DATA):
        cpp_readable_format.append(str(y_train[k]))
        cpp_readable_format.append("\n")

    text_file = open("mnist_test_labels.txt", "w")
    text_file.write("".join(cpp_readable_format[:-1]))
    text_file.close()

generate_x_train()
generate_y_train()
generate_x_test()
generate_y_test()