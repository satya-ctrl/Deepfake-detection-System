import subprocess

scripts = [
    "00-convert_video_to_image.py",
    "01a-crop_faces_with_mtcnn.py",  # or use 01b if you want Azure version
    "02-prepare_fake_real_dataset.py",
    "03-train_cnn.py"
]

for script in scripts:
    print(f"\n📁 Running: {script}")
    subprocess.run(["python", script], check=True)
