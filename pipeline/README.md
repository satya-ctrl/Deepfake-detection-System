🧠 Deepfake Detection for Images & Text 🕵️‍♂️

🚀 A Python project to detect deepfake images and optionally analyze associated text or metadata using AI/ML — built for learning, exploration, and real-world practice.

This project demonstrates how to build a Deepfake Detection System that processes images (and text data if available) to classify them as Real or Fake using machine learning techniques like convolutional neural networks (CNNs) and image preprocessing. 
GitHub

📌 Features

✨ Image preprocessing (convert videos to frames)
✨ Face cropping for focused analysis with MT-CNN or Azure Vision API
✨ Dataset preparation of real vs fake images
✨ CNN training & evaluation for classification
✨ Automated script runner (run_all.py)
✨ Image & text (optional) support to explore multimodal detection

📂 Repository Structure
Deepfake-detection-for-Images-and-text/
│
├── 00-convert_video_to_image.py         # Converts videos into frames
├── 01a-crop_faces_with_mtcnn.py         # Crops faces using MT-CNN
├── 01b-crop_faces_with_azure-vision-api.py  # Uses Azure Vision for cropping
├── 02-prepare_fake_real_dataset.py      # Prepares datasets for training
├── 03-train_cnn.py                      # Trains the CNN model
├── run_all.py                           # Runs the pipeline end-to-end
├── requirements.txt                     # Dependencies 📦
├── sample_dataset.png                   # Example data preview 🖼
├── dfdetect-home.png                    # Project screenshot 📸
└── README.md                            # This file 📝


(This structure is based on your repository files.) 
GitHub

🧠 Technologies Used

This project uses AI/ML and deep learning to detect deepfake media:

🧠 TensorFlow / Keras — Neural networks for deepfake classification
📸 OpenCV — Image frame extraction & processing
🤖 NumPy & Pandas — Data handling and preprocessing
🔍 scikit-learn — Evaluation metrics & dataset splitting
📊 Matplotlib / Seaborn — Visualizing loss & accuracy
🎯 MTCNN / Azure Vision API — Face detection & cropping

These tools power the detection workflow and help you build an accurate classifier. 
GitHub

📦 Setup & Installation

Clone the repository:

git clone https://github.com/satya-ctrl/Deepfake-detection-for-Images-and-text.git
cd Deepfake-detection-for-Images-and-text


Create a Python environment:

python -m venv venv
source venv/bin/activate   # macOS / Linux
venv\Scripts\activate      # Windows


Install dependencies:

pip install -r requirements.txt

🏁 How to Use
🛠 Run the Full Pipeline
python run_all.py


This script automatically runs conversion, cropping, dataset preparation, and model training.

🧪 What the Scripts Do 🎯
Script	Purpose
00-convert_video_to_image.py	Convert video inputs into image frames
01a-crop_faces_with_mtcnn.py	Crop faces using MTCNN
01b-crop_faces_with_azure-vision-api.py	Crop using Azure Vision API
02-prepare_fake_real_dataset.py	Build training/testing folders
03-train_cnn.py	Train CNN model for real/fake classification
run_all.py	Full pipeline runner

Each step builds toward a better deepfake classifier with improved accuracy and performance. 
GitHub

📈 Evaluation & Metrics

After training, your model performance is measured using:

✔ Accuracy
✔ Loss curve visualization
✔ Confusion matrix
✔ Precision & Recall

These metrics help you understand real vs fake classification quality. 
GitHub

🔥 Future Enhancements

🎯 Add support for text analysis alongside image detection
🧠 Use pre-trained deep models (ResNet, EfficientNet) for better accuracy 
GitHub

📊 Build a web interface for user uploads
🚀 Integrate real-time detection from webcam or video streams

🤝 Contributing

Want to help this project grow?

⭐ Star the repo

🍴 Fork it

✨ Create a branch (feature/awesome-feature)

📌 Commit your changes

📬 Open a Pull Request

📝 License

This project is open-source under the MIT License — free for learning and non-commercial use.

🙌 Thank You!

Thanks for exploring this project!
If you found it helpful, consider ⭐ starring the repo — and happy coding! 🚀✨
