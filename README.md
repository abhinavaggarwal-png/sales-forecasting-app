# Sales Forecasting Dashboard

A Streamlit-based interactive dashboard for forecasting SKU x City level sales with scenario planning capabilities.

## Features

- 📊 **Interactive Forecasting**: Adjust discount and marketing budget to see impact on November 2025 sales
- 📍 **Multi-level Analysis**: View forecasts at city level, SKU level, or detailed city x SKU combinations
- 🎯 **Key Driver Analysis**: Understand which features drive your forecast most
- 📈 **Confidence Intervals**: Get 50%, 75%, and 95% confidence bands for all predictions
- 🔄 **Baseline Comparison**: Compare your scenarios against a baseline (no-change) forecast
- 💾 **Export Capability**: Download detailed forecast data as CSV

## Project Structure

```
sales_forecasting_app/
├── app.py                      # Main Streamlit application
├── utils.py                    # Utility functions for preprocessing and prediction
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── models/
│   ├── xgb_model.pkl          # Trained XGBoost model
│   └── scaler.pkl             # Fitted StandardScaler
└── data/
    ├── all_data2.csv          # Historical data (you provide this)
    ├── y_pred.csv             # October predictions for error calculation
    └── city_item_sales.csv    # City x Item sales weights
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Steps

1. **Clone or download this project**

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Place your data file**
   - Copy your `all_data2.csv` to the `data/` folder
   - OR use the upload feature in the app
   - Default path expected: `data/all_data2.csv`

## Usage

### Running the App

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

### Using the Dashboard

#### 1. Load Data
- Use the sidebar to load your data
- Choose between default path or upload CSV
- Click "Load Data" button
- Wait for confirmation that data is loaded

#### 2. Set Input Parameters
- **Discount % Change**: Adjust the discount strategy (-50% to +100%)
  - Positive values = increase discounts
  - Negative values = decrease discounts
- **Budget % Change**: Adjust marketing spend (-50% to +200%)
  - Positive values = increase budget
  - Negative values = decrease budget

#### 3. Generate Forecast
- Click "Generate November Forecast" button
- The app will:
  - Update features based on your inputs
  - Generate predictions with confidence intervals
  - Compare against baseline scenario

#### 4. Analyze Results
Navigate through the tabs to explore:

**📍 City Analysis**
- Top 10 cities by forecast volume
- City-wise comparison of baseline vs updated forecast
- Detailed table with all cities

**📦 SKU Analysis**
- Top 10 SKUs by forecast volume
- SKU-wise comparison of baseline vs updated forecast
- Detailed table with all SKUs

**🎯 Key Drivers**
- Top 20 feature importances
- Impact summary of your input changes

**📈 Detailed Data**
- City x SKU level detailed forecast
- Filter by city or SKU
- Download complete forecast as CSV

## How It Works

### Data Flow

1. **Baseline**: September 2025 data is used as the conceptual baseline
2. **Update**: October 2025 features are adjusted based on user inputs
3. **Prepare**: November 2025 features are created:
   - Lag features updated with October actuals
   - Time features updated for November
   - User-adjusted discount and budget propagated
4. **Predict**: XGBoost model generates predictions
5. **Confidence**: Historical errors used to calculate confidence intervals

### Feature Updates

When you change inputs, the following features are automatically updated:

**Discount Changes** affect:
- `own_discount_avg`
- `own_discount_max`
- `discount_index_avg` (recalculated)
- `discount_index_sales_wt` (recalculated)
- `num_tata_discount_days` (scaled)

**Budget Changes** affect:
- `own_estimated_budget_consumed`
- `own_total_impressions`
- `own_ad_impressions`
- `own_organic_impressions`
- `own_direct_atc`, `own_indirect_atc`
- `own_direct_qty_sold`, `own_indirect_qty_sold`
- `own_direct_sales`, `own_indirect_sales`

### Model Details

- **Algorithm**: XGBoost Regressor
- **Features**: 127 features across 4 categories:
  - Historical sales (lags, rolling statistics)
  - Pricing & discounts (own and competitor)
  - On-shelf availability (OSA)
  - Marketing (impressions, budget, rankings)
- **Target**: Monthly quantity sold at city x SKU level
- **Preprocessing**: Standardization using fitted scaler

### Confidence Intervals

Confidence intervals are calculated based on October 2025 prediction errors:
- **50% CI**: ± 0.67 × std_error
- **75% CI**: ± 1.15 × std_error
- **95% CI**: ± 1.96 × std_error

All predictions and bounds are constrained to be non-negative.

## Data Requirements

### Input Data Format (all_data2.csv)

Required columns (132 total):
- **Identifiers**: `item_id`, `item_name`, `city_norm`, `month`
- **Target**: `monthly_qty_sold`
- **Sales Features**: Historical lags, rolling statistics, trends
- **Pricing Features**: Own and competitor prices, discounts
- **OSA Features**: On-shelf availability metrics
- **Marketing Features**: Impressions, budget, rankings

### Temporal Coverage
The data should include at least:
- August 2025
- September 2025
- October 2025

## Customization

### Changing Default Paths

Edit `app.py` and modify:
```python
data_path = "your/custom/path/all_data2.csv"
```

### Adjusting Slider Ranges

Edit `app.py` and modify the slider parameters:
```python
discount_change = st.slider(
    "Discount % Change",
    min_value=-50,    # Adjust minimum
    max_value=100,    # Adjust maximum
    value=0,          # Default value
    step=5            # Step size
)
```

### Adding Custom Visualizations

Add your charts in the appropriate tab sections in `app.py`:
```python
with tab1:  # City Analysis
    # Add your custom city-level charts here
    
with tab2:  # SKU Analysis
    # Add your custom SKU-level charts here
```

## Troubleshooting

### Common Issues

**"ModuleNotFoundError: No module named 'xgboost'"**
- Solution: Run `pip install -r requirements.txt`

**"FileNotFoundError: data/all_data2.csv not found"**
- Solution: Place your data file in the `data/` folder or use the upload feature

**"Predictions seem incorrect"**
- Check that your data format matches the expected schema
- Ensure all 132 columns are present
- Verify date formats are consistent (YYYY-MM-DD)

**"Model loading error"**
- Ensure `xgb_model.pkl` and `scaler.pkl` are in the `models/` folder
- Check that XGBoost version matches the one used for training

### Performance Tips

- For faster loading, filter data to only recent months if file is very large
- Close unused browser tabs when running locally
- Consider using uploaded file cache for repeated analyses

## Model Performance

Historical performance (October 2025):
- Check the sidebar after loading data for WAPE metric
- Typical accuracy varies by city and SKU
- Confidence intervals provide uncertainty quantification

## Future Enhancements

Potential improvements for future versions:
- [ ] Advanced mode: City x SKU level individual controls (currently placeholder)
- [ ] Multiple scenario comparison (side-by-side)
- [ ] What-if analysis with optimization suggestions
- [ ] Time series visualization of historical + forecast
- [ ] ROI calculator for discount and budget decisions
- [ ] Export to PowerPoint with auto-generated slides
- [ ] Integration with live data sources

## Support

For questions or issues:
1. Check this README
2. Review error messages in the Streamlit interface
3. Examine the console output for detailed logs

## License

Internal use only. Confidential and proprietary.

---

**Version**: 1.0  
**Last Updated**: November 2025  
**Built with**: Streamlit, XGBoost, Plotly
