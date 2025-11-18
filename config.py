# Configuration File for Sales Forecasting App
# Modify these settings to customize the dashboard behavior

# ======================
# DATA PATHS
# ======================
DATA_PATH = "data/all_data2.csv"
Y_PRED_PATH = "data/y_pred.csv"
SALES_WEIGHTS_PATH = "data/city_item_sales.csv"
MODEL_PATH = "models/xgb_model.pkl"
SCALER_PATH = "models/scaler.pkl"

# ======================
# FORECAST SETTINGS
# ======================
# Month to forecast
FORECAST_MONTH = "2025-11-01"

# Baseline month (for comparison)
BASELINE_MONTH = "2025-10-01"

# Training cutoff (months before this are used for training)
TRAIN_CUTOFF = "2025-10-01"

# ======================
# UI SETTINGS
# ======================
# Slider ranges
DISCOUNT_MIN = -50
DISCOUNT_MAX = 100
DISCOUNT_DEFAULT = 0
DISCOUNT_STEP = 5

BUDGET_MIN = -50
BUDGET_MAX = 200
BUDGET_DEFAULT = 0
BUDGET_STEP = 10

# Number of top items to display in charts
TOP_N_CITIES = 10
TOP_N_SKUS = 10

# ======================
# FEATURE SETTINGS
# ======================
# Features to drop before modeling
DROP_COLS = ['monthly_qty_sold', 'month', 'item_name', 'city_norm', 'item_id']

# Marketing columns prefix (for preprocessing)
MARKETING_PREFIXES = ('own_', 'comp_', 'bgr_')

# Sparsity threshold for dropping columns
SPARSITY_THRESHOLD = 0.90

# Large column threshold for log transformation
LARGE_COL_THRESHOLD = 1e3

# Outlier clipping quantiles
CLIP_LOWER_QUANTILE = 0.005
CLIP_UPPER_QUANTILE = 0.995

# ======================
# CONFIDENCE INTERVALS
# ======================
# Confidence level multipliers (based on standard error)
CI_50_MULTIPLIER = 0.67
CI_75_MULTIPLIER = 1.15
CI_95_MULTIPLIER = 1.96

# ======================
# VISUALIZATION SETTINGS
# ======================
# Color schemes
COLOR_BASELINE = "lightblue"
COLOR_FORECAST = "darkblue"
COLOR_SKU_BASELINE = "lightgreen"
COLOR_SKU_FORECAST = "darkgreen"

# Chart heights
CHART_HEIGHT_CITIES = 500
CHART_HEIGHT_SKUS = 600
CHART_HEIGHT_IMPORTANCE = 600

# ======================
# ADVANCED FEATURES
# ======================
# Enable/disable advanced city x SKU controls
ENABLE_ADVANCED_MODE = True

# Enable/disable feature importance tab
ENABLE_FEATURE_IMPORTANCE = True

# Enable/disable data download
ENABLE_DOWNLOAD = True

# ======================
# PERFORMANCE SETTINGS
# ======================
# Number of rows to display in tables
TABLE_HEIGHT = 400

# Cache TTL (seconds) for data loading
CACHE_TTL = 3600

# ======================
# BUSINESS SETTINGS
# ======================
# Company/Brand name for display
BRAND_NAME = "Tata"

# Competitor reference name
COMPETITOR_NAME = "Competitors"

# Currency symbol
CURRENCY_SYMBOL = "₹"

# Quantity unit
QUANTITY_UNIT = "units"
