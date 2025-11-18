#!/usr/bin/env python3
"""
Quick verification script to check if everything is set up correctly
Run this before starting the Streamlit app
"""

import sys
import os

def check_python_version():
    """Check Python version"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor} - Need Python 3.8+")
        return False

def check_dependencies():
    """Check if all required packages are installed"""
    print("\nChecking dependencies...")
    required = {
        'streamlit': 'Streamlit',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'plotly': 'Plotly',
        'xgboost': 'XGBoost',
        'sklearn': 'Scikit-learn',
        'joblib': 'Joblib'
    }
    
    all_ok = True
    for package, name in required.items():
        try:
            __import__(package)
            print(f"✅ {name} - OK")
        except ImportError:
            print(f"❌ {name} - NOT INSTALLED")
            all_ok = False
    
    return all_ok

def check_files():
    """Check if all required files exist"""
    print("\nChecking required files...")
    
    files_to_check = [
        ('app.py', 'Main application'),
        ('utils.py', 'Utility functions'),
        ('requirements.txt', 'Dependencies list'),
        ('models/xgb_model.pkl', 'XGBoost model'),
        ('models/scaler.pkl', 'Scaler object'),
        ('data/y_pred.csv', 'Predictions data'),
        ('data/city_item_sales.csv', 'Sales weights'),
    ]
    
    all_ok = True
    for filepath, description in files_to_check:
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"✅ {description} - OK ({size:,} bytes)")
        else:
            print(f"❌ {description} - NOT FOUND")
            all_ok = False
    
    return all_ok

def check_data_file():
    """Check if user's data file exists"""
    print("\nChecking data file...")
    
    if os.path.exists('data/all_data2.csv'):
        size = os.path.getsize('data/all_data2.csv')
        print(f"✅ all_data2.csv - OK ({size:,} bytes)")
        return True
    else:
        print("⚠️  all_data2.csv - NOT FOUND")
        print("   This is expected if you haven't copied your data yet.")
        print("   Copy your data file to: data/all_data2.csv")
        return None  # Not an error, just a warning

def main():
    """Run all checks"""
    print("=" * 60)
    print("Sales Forecasting App - Setup Verification")
    print("=" * 60)
    print()
    
    results = []
    
    # Run checks
    results.append(check_python_version())
    results.append(check_dependencies())
    results.append(check_files())
    data_check = check_data_file()
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    if all(results):
        print("✅ All checks passed!")
        
        if data_check:
            print("\n🚀 Ready to run the app!")
            print("\nTo start the dashboard, run:")
            print("   streamlit run app.py")
        else:
            print("\n⚠️  Almost ready! Just copy your data file:")
            print("   cp /path/to/your/all_data2.csv data/")
            print("\nThen run:")
            print("   streamlit run app.py")
    else:
        print("\n❌ Some checks failed. Please fix the issues above.")
        print("\nTo install dependencies, run:")
        print("   pip install -r requirements.txt")
    
    print()

if __name__ == "__main__":
    main()
