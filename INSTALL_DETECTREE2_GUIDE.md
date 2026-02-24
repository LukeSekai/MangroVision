# Installing Detectree2 AI on Windows

## Prerequisites

### Step 1: Install Visual Studio Build Tools

1. Download **Visual Studio Build Tools 2022**:
   - https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022

2. Run the installer and select:
   - ✅ **Desktop development with C++**
   - ✅ **MSVC v143 - VS 2022 C++ x64/x86 build tools**
   - ✅ **Windows 10/11 SDK**

3. This download is ~6-8GB and takes 20-40 minutes

### Step 2: Install Detectron2

```powershell
# Activate your virtual environment
cd C:\Users\Asus-Pc\Desktop\MangroVision
venv\Scripts\activate

# Install detectron2 from source
pip install git+https://github.com/facebookresearch/detectron2.git
```

### Step 3: Install Detectree2

```powershell
pip install detectree2
```

### Step 4: Test Installation

```powershell
python -c "import detectron2; import detectree2; print('✓ Success!')"
```

---

## Alternative: Use Python 3.10 or 3.11

Detectron2 has better support for older Python versions:

### 1. Install Python 3.10

Download from: https://www.python.org/downloads/release/python-31014/

### 2. Create New Virtual Environment

```powershell
cd C:\Users\Asus-Pc\Desktop\MangroVision
python3.10 -m venv venv_py310
venv_py310\Scripts\activate
```

### 3. Install Dependencies

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch2.0/index.html
pip install detectree2
pip install -r requirements.txt
```

---

## Alternative: Use Windows Subsystem for Linux (WSL)

This is the EASIEST method if you're comfortable with Linux:

### 1. Install WSL (PowerShell as Admin)

```powershell
wsl --install
```

Restart your computer.

### 2. Open Ubuntu and Install Everything

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install python3-pip python3-venv git -y

# Clone your project (or access it via /mnt/c/Users/...)
cd ~
git clone <your-repo> MangroVision
cd MangroVision

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch
pip install torch torchvision

# Install Detectron2 (much easier on Linux!)
pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Install Detectree2
pip install detectree2

# Install other dependencies
pip install -r requirements.txt
```

### 3. Run Your App from WSL

```bash
python -m streamlit run app.py
```

Your browser will still work from Windows!

---

## Quick Comparison

| Method | Difficulty | Time | Best For |
|--------|-----------|------|----------|
| **HSV Detection (Current)** | ✅ Already working | 0 min | Quick demos, proof of concept |
| **VS Build Tools** | 🟡 Medium | 1-2 hours | Full Windows setup |
| **Python 3.10** | 🟡 Medium | 30-60 min | If VS Build Tools fail |
| **WSL** | 🟢 Easy | 30-45 min | Best Linux-like experience |
| **Docker** | 🔴 Hard | 1-2 hours | Production deployment |

---

## My Recommendation

**For your thesis defense:**
1. **Use HSV detection** (already working) for initial demos
2. **Install WSL** if you want AI detection - it's the most reliable method
3. Show both methods in your defense to demonstrate different approaches

**HSV Detection is Actually Good Enough:**
- ✅ Works immediately
- ✅ Fast processing
- ✅ Reliable for green vegetation
- ✅ No complex dependencies
- ✅ Easier to explain in defense

**AI Detection Advantages:**
- Better accuracy with overlapping trees
- Handles shadows better
- More professional/impressive
- Published research method

---

## Test Current System

Your system is currently using HSV detection. Try it first:

```powershell
venv\Scripts\python.exe -m streamlit run app.py
```

Upload a drone image and see if the results are acceptable for your needs!

---

## Need Help?

If you decide to install detectron2, let me know which method you choose and I can guide you through any errors.
