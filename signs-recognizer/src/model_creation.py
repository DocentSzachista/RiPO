from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import tensorflow as tf
import pathlib

# for test purpose only 
import matplotlib.pyplot as plt

class RecognizerModel :
    def __init__(
    self, 
    image_width: int | None = 200, 
    image_height: int | None = 200, 
    batch_size : int | None = 32
    )->None:
        self._image_width = image_width
        self._image_height = image_height
        self._batch_size = batch_size
        self.AUTOTUNE = tf.data.AUTOTUNE # https://www.tensorflow.org/tutorials/images/classification#configure_the_dataset_for_performance
        self.normalization_layer = layers.Rescaling(1./255)

    def setup_datasets(self, show_ex_data : bool | None=False)->None:
        # 1. retrievwe data
        self.train_dataset = tf.keras.utils.image_dataset_from_directory(
            "../dataset",
            validation_split = 0.2,
            subset = "training",
            seed = 123,
            image_size=(self._image_height, self._image_width),
            batch_size = self._batch_size
        )
        self.test_dataset = tf.keras.utils.image_dataset_from_directory(
            "../dataset",
            validation_split = 0.2,
            subset = "validation",
            seed = 123,
            image_size=(self._image_height, self._image_width),
            batch_size = self._batch_size
        )
        # 2 preproccess data 
        # configure data efficiency
        train_ds = self.train_dataset.cache().shuffle(100).prefetch(buffer_size=self.AUTOTUNE)
        val_ds = self.test_dataset.cache().prefetch(buffer_size=self.AUTOTUNE)
        
        # normalize data so its easier to process data
        normalized_ds = train_ds.map(lambda x, y: (self.normalization_layer(x), y))
        image_batch, labels_batch = next(iter(normalized_ds))
        # 3 create model
        base_model = 

        # 4 train it 
        # 5 check it 

        # Test check if dataset has loaded correctly
        if show_ex_data:
            print(self.train_dataset.class_names)
            plt.figure(figsize=(10, 10))
            for images, labels in self.train_dataset.take(1):
                for i in range(9):
                    ax = plt.subplot(3, 3, i + 1)
                    plt.imshow(images[i].numpy().astype("uint8"))
                    # plt.title(class_names[labels[i]])
                    plt.axis("off")
            plt.show()

if __name__ == "__main__":
    model = RecognizerModel()
    model.setup_datasets()