"# raahulbale_dremalscan" 
# DermalScan-AI

DermalScan-AI is a small Python/Flask project for classifying common facial skin conditions using convolutional neural networks. It includes trained model files, scripts used for training, and a simple web frontend for uploading images and getting a diagnosis.

## Project structure

- `app.py` - Flask web app (serves `index.html` and exposes endpoints for image upload and inference).
- `diagnose.py` - Image preprocessing and model inference helper (used by the web app).
- `train_resnet_model.py` - Training script for fine-tuning a ResNet model on the dataset.
- `dermalscan_model.h5`, `dermalscan_resnet_model.h5`, `dermalscan_resnet_finetuned_model.h5` - Pretrained model files used for inference.
- `index.html` - Simple frontend for uploading images and displaying results.
- `dataset/` - Labeled image folders used for training (e.g. `clear face/`, `darkspots/`, `puffy eyes/`, `wrinkles/`).
- `models/` - Additional model files (face detection and age models used for preprocessing).
- `uploads/` - Uploaded images used by the webapp.

## Quick setup (Windows - cmd)

1. Install Python 3.10+ and create a virtual environment (recommended):

   python -m venv venv
   venv\Scripts\activate

2. Upgrade pip and install dependencies. There is no `requirements.txt` in this repo by default; install the common packages used by the project:

   python -m pip install --upgrade pip
   python -m pip install flask tensorflow keras opencv-python pillow numpy

3. Place the pretrained model files in the project root (they are already included in the repo):

   - `dermalscan_model.h5`
   - `dermalscan_resnet_model.h5`
   - `dermalscan_resnet_finetuned_model.h5`

4. Start the web app:

   set FLASK_APP=app.py
   python app.py

   Then open http://127.0.0.1:5000/ in your browser.

Note: If `app.py` uses a different run pattern (for example directly calling `app.run()`), run that file directly as shown.

## Usage

1. Open the web UI at `index.html` served by the Flask app.
2. Upload a face image using the provided form.
3. The server will preprocess the image, run the model, and return the predicted skin condition and confidence.

If you prefer command-line inference, use `diagnose.py` (example usage depends on the script's interface):

   python diagnose.py --image uploads/small.jpg --model dermalscan_resnet_finetuned_model.h5

Adjust arguments as needed based on the script implementation.

## Training

The `train_resnet_model.py` script contains the training loop used to fine-tune a ResNet-based classifier. Typical steps:

1. Prepare the `dataset/` directory with labeled subfolders for each class.
2. Edit `train_resnet_model.py` to point to your dataset path, desired image size, batch size, and training hyperparameters.
3. Run training (requires GPU for practical speed):

   python train_resnet_model.py

Checkpoints (saved model weights) will be produced in the script's configured output path.

## Notes and assumptions

- This README infers typical usage based on file names. If your `app.py` or other scripts expect different CLI flags or environment variables, update the commands accordingly.
- The repository does not include a `requirements.txt` or Dockerfile; consider adding one for reproducible setup.

