import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Input

print(f"TensorFlow Version: {tf.__version__}")

try:
    print("\nAttempting to build the model...")
    input_tensor = Input(shape=(224, 224, 3))
    model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_tensor=input_tensor
    )
    print("\n✅ ✅ ✅ SUCCESS! Model built without errors. ✅ ✅ ✅")
    print("This means your TensorFlow installation is working correctly.")
    model.summary()

except Exception as e:
    print("\n❌ ❌ ❌ FAILURE! The model could not be built. ❌ ❌ ❌")
    print("This confirms the issue is with the TensorFlow/Keras installation itself.")
    print("\n--- ERROR ---")
    print(e)
    print("\n--- END ERROR ---")
