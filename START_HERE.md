# 🚀 START HERE - Quick Setup Guide

Welcome to your Sales Forecasting Dashboard! Follow these simple steps to get started.

## ⚡ 3-Minute Quick Start

### Step 1: Copy Your Data (30 seconds)
```bash
# Copy your all_data2.csv file to the data folder
cp /path/to/your/all_data2.csv data/
```

> **Note**: Your data path is: `/Users/abhinav/Documents/Jupyter/Tata/all_data2.csv`

### Step 2: Install Dependencies (90 seconds)
```bash
pip install -r requirements.txt
```

Or use the automated script:
```bash
./setup.sh
```

### Step 3: Verify Setup (30 seconds)
```bash
python3 verify_setup.py
```

This will check if everything is configured correctly.

### Step 4: Launch App (30 seconds)
```bash
streamlit run app.py
```

The dashboard will open in your browser at: `http://localhost:8501`

---

## 📖 What to Read Next

**For first-time users:**
1. ✅ You are here: `START_HERE.md`
2. 📊 Next: `DEMO_GUIDE.md` - Learn how to present to VCs
3. 📚 Then: `README.md` - Complete documentation

**For developers:**
1. 🔧 `PROJECT_SUMMARY.md` - Technical overview
2. 🚀 `DEPLOYMENT.md` - Production deployment
3. ⚙️ `config.py` - Configuration options

---

## 🎯 Quick Demo Flow

Once the app is running:

1. **Load Data** (sidebar)
   - Click "Load Data" button
   - Wait for ✅ success message

2. **Set Inputs** (main area)
   - Move "Discount % Change" slider (try +20%)
   - Move "Budget % Change" slider (try +50%)

3. **Generate Forecast**
   - Click "Generate November Forecast" button
   - See results appear

4. **Explore Results** (tabs)
   - 📍 City Analysis
   - 📦 SKU Analysis
   - 🎯 Key Drivers
   - 📈 Detailed Data

---

## 📁 Project Structure

```
sales_forecasting_app/
├── START_HERE.md          ← You are here!
├── README.md              ← Full documentation
├── DEMO_GUIDE.md          ← Presentation guide
├── PROJECT_SUMMARY.md     ← Technical overview
├── DEPLOYMENT.md          ← Production deployment
├── app.py                 ← Main application
├── utils.py               ← Core logic
├── config.py              ← Settings
├── verify_setup.py        ← Setup checker
├── models/
│   ├── xgb_model.pkl     ← Trained model
│   └── scaler.pkl        ← Feature scaler
└── data/
    ├── all_data2.csv     ← YOUR DATA (copy here)
    ├── y_pred.csv        ← Predictions
    └── city_item_sales.csv ← Weights
```

---

## ❓ Troubleshooting

**Problem: "ModuleNotFoundError"**
```bash
# Solution: Install dependencies
pip install -r requirements.txt
```

**Problem: "FileNotFoundError: data/all_data2.csv"**
```bash
# Solution: Copy your data file
cp /Users/abhinav/Documents/Jupyter/Tata/all_data2.csv data/
```

**Problem: App won't start**
```bash
# Check Python version (need 3.8+)
python3 --version

# Verify setup
python3 verify_setup.py
```

**Problem: Port already in use**
```bash
# Run on different port
streamlit run app.py --server.port=8502
```

---

## 💡 Pro Tips

✨ **Keyboard Shortcuts in App:**
- `R` - Rerun the app
- `C` - Clear cache
- `?` - Show help

✨ **For VC Demo:**
- Read `DEMO_GUIDE.md` first
- Practice 2-3 scenarios beforehand
- Have backup screenshots ready

✨ **For Development:**
- Edit `config.py` to customize settings
- Check `utils.py` for logic changes
- Modify `app.py` for UI updates

---

## 🎓 Learning Path

**Beginner** → Run the app, explore features
**Intermediate** → Modify colors/layouts in config.py
**Advanced** → Add custom features to utils.py and app.py

---

## 📞 Quick Reference

| Command | Purpose |
|---------|---------|
| `streamlit run app.py` | Start the app |
| `python3 verify_setup.py` | Check setup |
| `./setup.sh` | Automated setup |
| `streamlit cache clear` | Clear cache |
| `streamlit --version` | Check version |

---

## ✅ Pre-Launch Checklist

Before presenting or deploying:

- [ ] Data file copied to `data/all_data2.csv`
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Setup verified (`python3 verify_setup.py`)
- [ ] App starts successfully (`streamlit run app.py`)
- [ ] Can load data in sidebar
- [ ] Can generate forecast
- [ ] All tabs display correctly
- [ ] Read `DEMO_GUIDE.md`
- [ ] Practiced demo scenarios

---

## 🎉 Ready to Go!

You have everything you need:
- ✅ Complete working application
- ✅ Trained ML model
- ✅ Comprehensive documentation
- ✅ Demo presentation guide
- ✅ Deployment instructions

**Now run:**
```bash
streamlit run app.py
```

**And impress your VCs! 🚀**

---

**Questions?** Check the README.md or other documentation files.

**Issues?** Run `python3 verify_setup.py` to diagnose.

**Ready to deploy?** Read `DEPLOYMENT.md`.

---

*Happy Forecasting! 📊*
