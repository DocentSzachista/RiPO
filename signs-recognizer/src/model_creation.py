from operator import mod
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import tensorflow_hub as hub
import tensorflow as tf
import pathlib
import numpy as np
# for test purpose only 
import matplotlib.pyplot as plt

class RecognizerModel :
    def __init__(
    self, 
    image_width: int | None = 200, 
    image_height: int | None = 200, 
    batch_size : int | None = 32,
    load_model : bool | None = False
    )->None:
        self._image_width = image_width
        self._image_height = image_height
        self._batch_size = batch_size
        self.AUTOTUNE = tf.data.AUTOTUNE # https://www.tensorflow.org/tutorials/images/classification#configure_the_dataset_for_performance
        self.normalization_layer = layers.Rescaling(1./255)
        if load_model :
            self.model = tf.keras.models.load_model('./src/model/test_model')
            # Check its architecture
            self.model.summary()
        


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
        test_ds = self.test_dataset.cache().prefetch(buffer_size=self.AUTOTUNE)
        
        # normalize data so its easier to process data
        normalized_ds = train_ds.map(lambda x, y: (self.normalization_layer(x), y))
        image_batch, labels_batch = next(iter(normalized_ds))
        
        num_classes = len(self.train_dataset.class_names)

        model = Sequential([
        layers.Rescaling(1./255, input_shape=(self._image_height, self._image_width, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
        ])

        model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

        model.summary()

        epochs=20
        history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=epochs
        )

        model.save("model/test_model")

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(epochs)


        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()



        # link do headless modelu
        #  DUPA DEBBUGING xD 
        # feature_url = "https://tfhub.dev/google/efficientnet/b0/feature-vector/1"
        # extract_layer = hub.KerasLayer(feature_url, input_shape =(self._image_height, self._image_width))
        # extract_layer.trainable = False
        # # 3 create model
        # model = tf.keras.Sequential([
        #     extract_layer, 
        #     layers.Dense(len(self.train_dataset.class_names), activation="softmax")
        # ]) 
        # # 4 train it
        # model.compile(optimizer = tf.keras.optimizers.Adam(), loss="categorical_crossentropy", metrics=['acc'])
        # steps_per_epoch = np.ceil(train_ds.samples/train_ds.batch_size)
        # history = model.fit(train_ds, epochs =2, steps_per_epoch=steps_per_epoch)
        # class_names = sorted(test_ds.class_indices.items(), key = lambda pair:pair[1])
        # class_names = np.array([key.title() for key, _ in class_names])

        # # 5 check it 
        # for img_batch, label_batch in test_ds:
        #     predicted_batch = model.predict(img_batch)
        #     predicted_id = np.argmax(predicted_batch, axis =-1)
        #     predicted_label_batch = class_names[predicted_id]
        
        # model.evaluate(test_ds)



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
    def predict(self, image):


        
        img_array = tf.keras.utils.img_to_array(image)
        img_array = tf.expand_dims(img_array, 0) # Create a batch
        predictions = self.model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(self.train_dataset.class_names[np.argmax(score)], 100 * np.max(score))
        )

        pass


if __name__ == "__main__":
    model = RecognizerModel()
    model.setup_datasets()