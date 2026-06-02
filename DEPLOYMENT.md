# Deploying Your Deepfake Detection HUD

This guide explains how to initialize Git, push your project to GitHub, and deploy it to a free hosting service like Render so you can share it with others.

---

## 🐙 Step 1: Push to GitHub

1. **Install Git** (if not already installed) from [git-scm.com](https://git-scm.com/).
2. **Open Git Bash or Command Prompt** in your project folder:
   `c:\Users\Satya\OneDrive\Desktop\Deepfake-detection-System-main`
3. **Initialize Git** and commit your files:
   ```bash
   git init
   git add .
   git commit -m "Initialize DeepSense web application and dataset downloader"
   ```
4. **Create a new Repository** on your GitHub account:
   - Go to [github.com/new](https://github.com/new).
   - Name your repository (e.g. `Deepfake-Detection-HUD`).
   - Do **NOT** initialize it with a README, `.gitignore`, or License (we already have them).
   - Click **Create repository**.
5. **Link and Push** your code:
   ```bash
   # Replace with your actual GitHub URL
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   git push -u origin main
   ```

---

## 🚀 Step 2: Deploy to Render (Free Hosting)

Render is a modern cloud hosting platform with a generous free tier that automatically connects to your GitHub repository.

1. **Create an account** at [render.com](https://render.com/) (log in with your GitHub account).
2. Click **New +** in the dashboard and select **Web Service**.
3. **Connect your Repository**: Select the GitHub repository you just pushed.
4. **Configure Web Service Settings**:
   - **Name**: `deepsense-deepfake-hud` (or any name you like)
   - **Region**: Select the closest region (e.g. Oregon or Singapore)
   - **Branch**: `main`
   - **Runtime**: `Python`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
5. **Select Plan**: Select the **Free** instance type.
6. Click **Deploy Web Service** at the bottom.

---

## 🧠 Step 3: Managing the AI Model (`deepfake_model.h5`) on Cloud

Your trained weights file (`deepfake_model.h5`) can be around 100MB–200MB. GitHub has a limit of 100MB per file. 

If your file is larger than 100MB, you have two great options:

### Option A: Direct Upload (Easiest)
If your model is under 100MB, you can commit it to Git and push it:
1. Copy `deepfake_model.h5` into this folder.
2. In git, commit it:
   ```bash
   git add deepfake_model.h5
   git commit -m "Add trained deepfake model"
   git push origin main
   ```
Render will automatically build it, and it will deploy in **Live AI Mode**!

### Option B: Cloud Storage Link (Best for large models)
If your model is too large for GitHub, you can host the file on Google Drive or Dropbox and modify `app.py` to download it automatically on startup.
Alternatively, you can host your application as a **Hugging Face Space** using Git LFS, which supports model sizes up to 10GB for free! (Instructions below).

---

## 🤗 Alternative: Hugging Face Spaces (Highly Recommended for ML Apps)

Hugging Face Spaces is designed specifically for Machine Learning web apps, supports Python/Flask, and includes Git LFS (Large File Storage) for free.

1. Create a free account at [huggingface.co](https://huggingface.co/).
2. Click **New** -> **Space**.
3. Configure Space:
   - **Owner / Space Name**: Your choice.
   - **SDK**: Select **Docker** -> **Blank** (or create a simple Gradio/Streamlit space, but Docker allows you to run Flask perfectly).
   - **Space Hardware**: Free CPU basic.
   - **Public** or **Private**: Your choice.
4. Clone the space locally and copy these project files into it, commit, and push. Hugging Face will build your Flask site using Docker and deploy it immediately!
