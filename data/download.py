import os

import numpy as np
import tensorflow as tf


def save_dataset_as_pgm_p5(
    x_data: np.ndarray, y_data: np.ndarray, output_folder: str
) -> None:
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    labels: list[str] = []

    for i in range(len(x_data)):
        filename: str = f"{i:05d}.pgm"
        filepath: str = os.path.join(output_folder, filename)

        with open(filepath, "wb") as f:
            header: str = "P5\n28 28\n255\n"
            _ = f.write(header.encode("ascii"))
            _ = f.write(x_data[i].tobytes())

        labels.append(str(y_data[i]))

    with open(os.path.join(output_folder, "labels.txt"), "w") as f:
        _ = f.write(",".join(labels))


x_train: np.ndarray
y_train: np.ndarray
x_test: np.ndarray
y_test: np.ndarray

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

save_dataset_as_pgm_p5(x_train, y_train, "train")
save_dataset_as_pgm_p5(x_test, y_test, "test")

print("Files saved in 'train' and 'test'.")
