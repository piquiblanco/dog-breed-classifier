import pickle

from keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling2D,
    MaxPooling2D,
)
from keras.models import Sequential


def extract_Resnet50(tensor):
    #  function derived from dog_app.ipynb file
    from keras.applications.resnet50 import ResNet50, preprocess_input

    return ResNet50(weights="imagenet", include_top=False).predict(
        preprocess_input(tensor)
    )


transfer_model = Sequential()
transfer_model.add(GlobalAveragePooling2D(input_shape=train_Resnet50.shape[1:]))
transfer_model.add(Dense(133, activation="softmax"))
transfer_model.summary()
transfer_model.load_weights("weights.best.transfer.hdf5")

bottleneck_features = np.load("../bottleneck_features/DogResnet50Data.npz")


def Resnet50_predict_breed(img_path):
    #  function derived from dog_app.ipynb file
    # extract bottleneck features
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = transfer_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]


def dog_breed_predict(img_path):
    """
    Function detecting dogs or humans in picture and predicting a dog breed based on image path.
    Input:
    img_path : str
        Path to image.
    Output:
    detected : str
        Label 'human', 'dog', or None.
    dog_breed_label : str
        Predicted dog breed.
    """
    plt.imshow(cv2.imread(img_path))
    detected = None
    if dog_detector(img_path) == True:
        detected = "dog"
        print("Dog detected in image")
        dog_breed_label = Resnet50_predict_breed(img_path)
    else:
        if face_detector(img_path) == True:
            detected = "human"
            print("Human detected in image")
            dog_breed_label = Resnet50_predict_breed(img_path)
        else:
            print("No dog or human detected in image")
            dog_breed_label = None
    return detected, dog_breed_label

