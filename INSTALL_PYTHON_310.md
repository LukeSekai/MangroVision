# Install Python 3.10 for Detectree2

## Step 1: Download Python 3.10

1. Go to: https://www.python.org/downloads/release/python-31014/
2. Scroll down to "Files"
3. Download: **Windows installer (64-bit)**

## Step 2: Install Python 3.10

1. Run the installer
2. ✅ **IMPORTANT**: Check "Add Python 3.10 to PATH"
3. Click "Install Now"
4. Wait for installation to complete

## Step 3: Verify Installation

Open a NEW PowerShell window and run:

```powershell
py -3.10 --version
```

Should show: `Python 3.10.14`

## Step 4: Create New Virtual Environment with Python 3.10

```powershell
cd C:\Users\Asus-Pc\Desktop\MangroVision

# Create new venv with Python 3.10
py -3.10 -m venv venv_py310

# Activate it
venv_py310\Scripts\activate

# Verify you're using Python 3.10
python --version
```

## Step 5: Install All Dependencies

```powershell
# Install PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install Detectron2
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch2.0/index.html

# Install Detectree2
pip install detectree2

# Install other requirements
pip install -r requirements.txt
```

## Step 6: Test Detectree2

```powershell
python -c "import torch; import detectron2; import detectree2; print('✅ All AI libraries installed successfully!')"
```

## Step 7: Run MangroVision with Python 3.10

```powershell
# Make sure you're in the MangroVision folder with venv_py310 activated
venv_py310\Scripts\python.exe -m streamlit run app.py
```

---

## Alternative: Use Your Current Python 3.12 with Pre-built Wheel

If Python 3.10 installation fails, try this community-built wheel for Python 3.12:

```powershell
# Activate your current venv
venv\Scripts\activate

# Try installing unofficial detectron2 build
pip install detectron2 @ https://github.com/MaureenZOU/detectron2-windows/releases/download/v0.6/detectron2-0.6-cp312-cp312-win_amd64.whl

# Then install detectree2
pip install detectree2
```

---

## Which Method to Use?

**RECOMMENDED: Install Python 3.10** (most reliable)
- Official support
- Proven compatibility
- 20 minutes setup time

**ALTERNATIVE: Try unofficial wheel** (faster but risky)
- May work immediately
- Not officially supported
- May have bugs

---

## After Installation

Update your batch files to use the correct Python:

### START_MANGROVISION.bat
```batch
@echo off
echo Starting MangroVision...
start cmd /k "cd /d C:\Users\Asus-Pc\Desktop\MangroVision && venv_py310\Scripts\python.exe start_tile_server.py"
timeout /t 3
start cmd /k "cd /d C:\Users\Asus-Pc\Desktop\MangroVision && venv_py310\Scripts\python.exe -m streamlit run app.py"
```
