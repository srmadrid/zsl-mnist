import os

import tensorflow as tf


def save_dataset_as_pgm_p5(x_data, y_data, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    labels = []

    for i in range(len(x_data)):
        filename = f"{i:05d}.pgm"
        filepath = os.path.join(output_folder, filename)

        with open(filepath, "wb") as f:
            header = f"P5\n28 28\n255\n"
            f.write(header.encode("ascii"))
            f.write(x_data[i].tobytes())

        labels.append(str(y_data[i]))

    with open(os.path.join(output_folder, "labels.txt"), "w") as f:
        f.write(",".join(labels))


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
save_dataset_as_pgm_p5(x_train, y_train, "train")
save_dataset_as_pgm_p5(x_test, y_test, "test")

print("Files saved in 'train' and 'test'.")
