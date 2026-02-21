# Team Setup Guide

## For New Team Members

### 1. Clone the Repository

```bash
git clone https://github.com/LukeSekai/MangroVision.git
cd MangroVision
```

### 2. Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Get the Data Files

⚠️ **Important:** The large data files are NOT in this repository.

Ask the project owner for:
- `dataset_frames/` - Training/test image frames
- `drone_images/` - Drone imagery
- `flight_videos/` - Original flight videos

Place these folders in the `MangroVision/` directory.

### 5. Test Installation

```bash
python canopy_detection/test_installation.py
```

---

## Daily Workflow

### Get Latest Changes
Before starting work:
```bash
git pull
```

### Make Changes
1. Edit files as needed
2. Test your changes

### Upload Your Changes
```bash
git add .
git commit -m "Brief description of what you changed"
git push
```

### If Push Fails (Someone Else Pushed First)
```bash
git pull
# Resolve any conflicts if needed
git push
```

---

## Collaboration Tips

- **Pull before you push** - Always get latest changes first
- **Commit often** - Small, frequent commits are better
- **Write clear commit messages** - Help your team understand changes
- **Test before pushing** - Make sure your code works
- **Communicate** - Let team know about major changes

---

## Useful Commands

```bash
git status              # Check what files changed
git log --oneline -5    # See recent commits
git diff                # See what changed in files
git branch              # See current branch
```

---

## Getting Help

- Check the main [README.md](README.md) for project overview
- See [HOW_TO_USE.md](HOW_TO_USE.md) for usage instructions
- Ask team members if stuck!
