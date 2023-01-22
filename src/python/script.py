import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
import pytesseract
from PIL import Image
import os


def models():
    if os.path.exists("models/model_cropped") and os.path.exists("models/model_text"):
        return tf.saved_model.load("models/model_cropped"), tf.saved_model.load("models/model_text")

    raw_images = tf.keras.preprocessing.image.load_img("trainingdata/raw/")
    cropped_images = tf.keras.preprocessing.image.load_img("trainingdata/cropped/")
    strings = []
    for files in os.listdir("trainingdata/strings/"):
        with open(os.path.join("trainingdata/strings/", files), "r") as f:
            strings.append(f.read())

    # Define the input shape for the model
    input_shape = (256, 256, 3)

    # Create the input layer
    inputs = Input(shape=input_shape)

    # Create the first convolutional layer
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = MaxPooling2D((2, 2), padding="same")(x)

    # Create the second convolutional layer
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2), padding="same")(x)

    # Create the third convolutional layer
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2), padding="same")(x)

    # Create the fourth convolutional layer
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = UpSampling2D((2, 2))(x)

    # Create the fifth convolutional layer
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = UpSampling2D((2, 2))(x)

    # Create the sixth convolutional layer
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = UpSampling2D((2, 2))(x)

    # Create the output layer
    outputs = Conv2D(3, (3, 3), activation="sigmoid", padding="same")(x)

    # Create the model
    crop_model = keras.Model(inputs, outputs)

    # Compile the model
    crop_model.compile(optimizer="adam", loss="binary_crossentropy")

    # Train the model using the training data
    crop_model.fit(
        x=raw_images,
        y=cropped_images,
        batch_size=32,
        epochs=10,
    )

    model_cropped = crop_model.predict(raw_images)
    tf.saved_model.save(model_cropped, "models/model_cropped")

    # Define a function to process the images and extract text using pytesseract
    def process_image(image):
        # Crop the image using the cropping model
        cropped_image = crop_model.predict(image)

        # Convert the cropped image to a PIL image
        image = Image.fromarray(cropped_image)

        # Use pytesseract to extract text from the image
        text = pytesseract.image_to_string(image)

        return text

    # Process the training images and extract text
    train_text = [process_image(image) for image in model_cropped]

    # Create a text recognition model using TensorFlow
    text_model = keras.Sequential()
    text_model.add(keras.layers.Dense(128, input_shape=(None,), activation="relu"))
    text_model.add(keras.layers.Dense(64, activation="relu"))
    text_model.add(keras.layers.Dense(len(strings[0]), activation="softmax"))
    text_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    # Train the text recognition model
    text_model.fit(train_text, strings, epochs=10)

    tf.saved_model.save(text_model, "models/text_model")

    return crop_model, text_model


def main():
    crop_model, text_model = models()

    # Load the image to be processed
    image = Image.open("test.jpg")

    # Process the image and extract text
    text = text_model.predict(crop_model.predict(image))

    # Print the extracted text
    print(text)


if __name__ == "__main__":
    main()
