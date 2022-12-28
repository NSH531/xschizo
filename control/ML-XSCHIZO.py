import tensorflow as tf
from tensorflow.keras import layers

def create_cnn_model():
    model = tf.keras.Sequential()

    # first convolutional layer
    model.add(layers.Conv3D(filters=16, kernel_size=(3,3,3), activation='relu', input_shape=(256,256,256,1)))
    model.add(layers.MaxPool3D(pool_size=(2,2,2)))

    # second convolutional layer
    model.add(layers.Conv3D(filters=32, kernel_size=(3,3,3), activation='relu'))
    model.add(layers.MaxPool3D(pool_size=(2,2,2)))

    # third convolutional layer
    model.add(layers.Conv3D(filters=64, kernel_size=(3,3,3), activation='relu'))
    model.add(layers.MaxPool3D(pool_size=(2,2,2)))

    # flatten output and add dense layers
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    return model

model = create_cnn_model()
model.summary()




import glob
import nibabel as nib

get all fMRI NIFTI files in current directory
fmri_files = glob.glob('*.nii')

for fmri_file in fmri_files:
    # load fMRI data
    fmri_data = nib.load(fmri_file).get_data()
    # reshape data for CNN input
    fmri_data = fmri_data.reshape(fmri_data.shape[0], fmri_data.shape[1], fmri_data.shape[2], 1)

    # run fMRI data through CNN model
    prediction = model.predict(fmri_data)

    print("Prediction for", fmri_file, ":", prediction)

