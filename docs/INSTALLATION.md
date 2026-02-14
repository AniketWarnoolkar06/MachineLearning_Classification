# Installation Guide

## Prerequisites

- **Python**: 3.10 recommended (matches `runtime.txt` / `.python-version`)
- **pip**: Python package installer
- **Git**: For cloning the repository

## Step 1: Clone the Repository

```bash
git clone <repository-url>
cd MachineLearning_Classification
```

If you've already cloned the repo, just `cd` into the project root.

## Step 2: Create a Virtual Environment

It's recommended to use a virtual environment to avoid dependency conflicts.

### On macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

## Step 3: Install Dependencies

Install all required packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Required Packages:
- `streamlit` - Web app framework
- `pandas`, `numpy` - Data handling
- `scikit-learn`, `scipy` - ML and scientific computing
- `xgboost` - Gradient boosting model
- `matplotlib` - Plots in the Streamlit app
- `joblib` - Model loading

## Step 4: Verify Installation

Check if all packages are installed correctly:

```bash
pip list
```

Or run a quick test:

```bash
python -c "import pandas, numpy, sklearn, xgboost, streamlit; print('All packages installed successfully!')"
```

Optionally verify Streamlit starts:

```bash
streamlit --version
```

## Step 5: Set Up Jupyter (Optional)

If you want to run the notebooks, install Jupyter:

```bash
pip install jupyter notebook
```

Or use JupyterLab:

```bash
pip install jupyterlab
```

## Troubleshooting

### Issue: `pip` command not found
- Make sure Python is installed and added to your system PATH
- Try using `pip3` instead of `pip`

### Issue: Permission denied during installation
- Use `pip install --user -r requirements.txt`
- Or use `sudo pip install -r requirements.txt` (not recommended)

### Issue: XGBoost installation fails
- On macOS: Install with `brew install libomp` first
- On Windows: May require Microsoft Visual C++ Build Tools

### Issue: Streamlit app can't find model files
Run Streamlit from the project root (the folder that contains `app.py`, `models/`, and `data/`):

```bash
streamlit run app.py
```

## Next Steps

Once installation is complete, refer to [USAGE.md](USAGE.md) for instructions on running the notebooks and Streamlit app.
