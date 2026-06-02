Deepfake Detection System 

A Deepfake Detection System built using AI/ML and Deep Learning in Google Colab.
This project demonstrates how machine learning can be used to classify images or video frames as Real or Fake based on deepfake patterns.

This notebook was created as part of a learning project to explore AI, neural networks, and computer vision techniques.

📌 Overview

Deepfakes are AI-generated/manipulated images or videos that look extremely realistic. Detecting them requires deep learning models, especially CNNs, which can learn subtle pixel-level inconsistencies.

This project (via Deepfake.ipynb) performs:

Data preprocessing

Image/frame extraction

Model training

Feature analysis

Deepfake classification

Accuracy & metric evaluation

🧠 Technologies & Libraries Used

Your Deepfake Detection System uses Artificial Intelligence & Machine Learning, specifically:

🔥 Deep Learning Framework

TensorFlow / Keras
Used for building, training, and validating the deepfake detection neural network.

🖼 Image & Video Processing

OpenCV (cv2)
Used for reading images, resizing, extracting frames, and basic preprocessing.

🔢 Numeric Computing

NumPy
For handling image arrays and numerical operations.

📊 Machine Learning Utilities

scikit-learn (optional but commonly used)
Used for:

Train-Test Split

Accuracy, Precision, Recall

Confusion Matrix

📈 Visualization

Matplotlib

Seaborn
Used to visualize training graphs and model performance.

☁ Environment

Google Colab
Notebook-based environment used to run all code, train models, and access GPU/TPU if needed.
(colab.research.google.com
)

📁 Repository Structure

```text
Deepfake-detection-System/
│
├── static/                # Web UI assets
│   ├── style.css          # Glassmorphic layout styles
│   └── app.js             # Drag-and-drop & Chart.js client-side scripts
├── templates/             # HTML Templates
│   └── index.html         # Analysis HUD, Analytics tab, and Dataset onboarding
├── app.py                 # Flask backend API server (processes requests)
├── download_dataset.py    # Automated dataset utility (Kaggle download & format)
├── Deepfake.ipynb        # Jupyter/Colab notebook for model training
├── README.md             # Project documentation
└── LICENSE               # MIT License
```

🚀 Getting Started

Follow these steps to expand the dataset, run the training pipeline, and deploy the visual frontend interface.

### 1️⃣ Download Expanded Dataset (Local Workspace)
Before training on more data, download the **140k Real and Fake Faces** dataset:
```bash
python download_dataset.py
```
This utility will authenticate with your Kaggle API, download the 1.2 GB dataset, and organize it into the standard `Dataset/` folder structure.

### 2️⃣ Train on Google Colab (GPU Acceleration)
1. Compress your new `Dataset` folder into a ZIP file named `Dataset.zip`.
2. Upload `Dataset.zip` to your main Google Drive directory (`My Drive`).
3. Upload `Deepfake.ipynb` to [Google Colab](https://colab.research.google.com/).
4. Follow the notebook steps. It will mount Google Drive, unzip `Dataset.zip`, and train the CNN model.

### 3️⃣ Save and Export Your Trained Model
At the end of your training Colab session, run the following code cell to download your model:
```python
# Save the model
model.save("deepfake_model.h5")

# Download model locally
from google.colab import files
files.download("deepfake_model.h5")
```
Place the downloaded `deepfake_model.h5` file in the root folder of this project.

### 4️⃣ Launch the Aesthetic Web Frontend
Run the Flask server locally:
```bash
pip install flask opencv-python tensorflow numpy pandas openpyxl
python app.py
```
Open your browser and navigate to:
👉 **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

*Note: If no `deepfake_model.h5` model is found, the server launches in **Demonstration Mode**, providing simulated scans and onboarding prompts so you can check out the frontend visuals immediately.*

📊 UI/UX Features
- **DeepScan HUD**: Interactive drop zone, dynamic scanner laser animation, progress arc metrics, and diagnostic indicators.
- **Model Analytics**: Visualizes your model performance data (Accuracy, Loss, Confusion Matrix) using interactive Chart.js widgets.
- **Dataset Hub**: Walkthrough onboarding system explaining how to link Google Drive, run Colab notebooks, and compile `.h5` files.

📄 License

This project is licensed under the MIT License.
