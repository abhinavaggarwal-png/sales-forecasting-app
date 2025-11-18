import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Feature columns used by the model (in exact order)
FEATURE_COLS = [
    'monthly_mrp_median', 'lag_1', 'lag_2', 'lag_3', 'lag_6', 'lag_12',
    'rolling_mean_3m', 'rolling_mean_6m', 'rolling_std_3m', 'rolling_std_6m',
    'cv_3m', 'cv_6m', 'slope_3m', 'slope_6m', 'zero_count_3m', 'zero_count_6m',
    'is_intermittent', 'mean_sales_12m', 'max_sales_12m', 'min_sales_12m',
    'month_number_since_launch', 'is_new_sku_last_3m', 'month_num', 'year_num',
    'month_sin', 'month_cos', 'item_te', 'city_te', 'item_city_te',
    'own_mrp_sales_wt', 'own_price_sales_wt', 'own_discount_sales_wt',
    'comp_price_sales_wt', 'comp_discount_sales_wt', 'price_index_sales_wt',
    'discount_index_sales_wt', 'price_gap_sales_wt', 'sku_count_sales_wt',
    'own_mrp_avg', 'own_price_avg', 'own_discount_avg', 'comp_price_avg',
    'comp_discount_avg', 'price_index_avg', 'discount_index_avg',
    'price_gap_avg', 'sku_count_avg', 'own_price_min', 'own_discount_max',
    'comp_price_min', 'comp_discount_max', 'price_gap_min', 'price_index_min',
    'comp_abs_min_price', 'num_tata_discount_days', 'num_comp_discount_days',
    'num_days_tata_cheaper', 'num_days_comp_cheaper', 'own_avg_osa_wt',
    'own_weighted_osa_wt', 'own_avg_listed_osa_wt', 'own_weighted_listed_osa_wt',
    'comp_avg_osa_wt', 'comp_weighted_osa_wt', 'comp_avg_listed_osa_wt',
    'comp_weighted_listed_osa_wt', 'own_osa_avg', 'own_listed_osa_avg',
    'comp_osa_avg', 'comp_listed_osa_avg', 'own_avg_osa_min', 'own_listed_osa_min',
    'comp_avg_osa_min', 'comp_listed_osa_min', 'own_avg_osa_max', 'own_listed_osa_max',
    'comp_avg_osa_max', 'comp_listed_osa_max', 'num_days_own_oos', 'num_days_own_wt_oos',
    'num_days_own_listed_oos', 'num_days_comp_oos', 'num_days_comp_wt_oos',
    'num_days_comp_listed_oos', 'num_days_comp_oos_while_tata_ok',
    'own_total_impressions', 'own_ad_impressions', 'own_organic_impressions',
    'bgr_total_impressions', 'bgr_ad_impressions', 'bgr_org_impressions',
    'bgr_total_kw_impressions', 'bgr_ad_kw_impressions', 'bgr_org_kw_impressions',
    'own_direct_atc', 'own_indirect_atc', 'own_direct_qty_sold', 'own_indirect_qty_sold',
    'own_direct_sales', 'own_indirect_sales', 'own_estimated_budget_consumed',
    'comp_total_impressions', 'comp_ad_impressions', 'comp_organic_impressions',
    'comp_generic_kw_ad_impr', 'comp_generic_kw_org_impr', 'comp_brand_kw_ad_impr',
    'comp_brand_kw_org_impr', 'comp_comp_kw_ad_impr', 'comp_comp_kw_org_impr',
    'comp_kw_weighted_impr', 'comp_generic_kw_weighted_ad_impr',
    'comp_generic_kw_weighted_org_impr', 'comp_brand_kw_weighted_ad_impr',
    'comp_brand_kw_weighted_org_impr', 'comp_comp_kw_weighted_ad_impr',
    'comp_comp_kw_weighted_org_impr', 'own_median_most_viewed_position',
    'own_median_ad_rank', 'own_median_org_rank', 'comp_ad_ratio',
    'comp_brand_visibility_ratio', 'comp_generic_visibility_ratio',
    'own_best_viewed_position', 'num_comp_pids', 'comp_ad_share_monthly'
]


def load_models():
    """Load trained model and scaler"""
    model = joblib.load('models/xgb_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return model, scaler


def load_data(data_path='data/all_data2.csv'):
    """Load historical data"""
    df = pd.read_csv(data_path)
    df['month'] = pd.to_datetime(df['month'])
    return df


def load_predictions():
    """Load October predictions for error calculation"""
    y_pred = pd.read_csv('data/y_pred.csv')
    return y_pred['0'].values  # Assuming column is named '0'


def load_sales_weights():
    """Load city x item sales weights for input distribution"""
    weights = pd.read_csv('data/city_item_sales.csv')
    # Normalize to get proportions
    # Handle both 'sum' and 'qty_sold' column names
    if 'sum' in weights.columns:
        weights['weight'] = weights['sum'] / weights['sum'].sum()
    elif 'qty_sold' in weights.columns:
        weights['weight'] = weights['qty_sold'] / weights['qty_sold'].sum()
    else:
        # Fallback: use the last numeric column
        numeric_cols = weights.select_dtypes(include=['float64', 'int64']).columns
        weights['weight'] = weights[numeric_cols[-1]] / weights[numeric_cols[-1]].sum()
    return weights


def calculate_confidence_intervals(df_oct, y_pred_oct):
    """Calculate confidence interval bands based on October errors"""
    # Match predictions to actuals
    errors = np.abs(y_pred_oct - df_oct['monthly_qty_sold'].values)
    std_error = errors.std()
    
    # Standard multipliers for confidence levels
    ci_multipliers = {
        '50%': 0.67,
        '75%': 1.15,
        '95%': 1.96
    }
    
    return std_error, ci_multipliers


def update_features_with_inputs(df_base, discount_change, budget_change, 
                                city_sku_changes=None, sales_weights=None):
    """
    Update baseline features with user inputs
    
    Parameters:
    - df_base: baseline dataframe (October data)
    - discount_change: % change in discount (e.g., 10 for +10%)
    - budget_change: % change in budget (e.g., 20 for +20%)
    - city_sku_changes: dict of {(city, sku): {'discount': %, 'budget': %}} for advanced mode
    - sales_weights: dataframe with city/item/weight for distribution
    """
    df = df_base.copy()
    
    # Convert percentage changes to multipliers
    discount_multiplier = 1 + (discount_change / 100)
    budget_multiplier = 1 + (budget_change / 100)
    
    if city_sku_changes is None:
        # Simple mode: apply same change to all
        # Update discount-related features
        df['own_discount_avg'] *= discount_multiplier
        df['own_discount_max'] *= discount_multiplier
        df['own_discount_sales_wt'] *= discount_multiplier
        
        # Recalculate discount indices
        df['discount_index_avg'] = df['own_discount_avg'] / (df['comp_discount_avg'] + 1e-6)
        df['discount_index_sales_wt'] = df['own_discount_sales_wt'] / (df['comp_discount_sales_wt'] + 1e-6)
        
        # Update num_tata_discount_days proportionally (capped at 30)
        df['num_tata_discount_days'] = np.minimum(
            df['num_tata_discount_days'] * discount_multiplier, 
            30
        )
        
        # Update budget-related features
        df['own_estimated_budget_consumed'] *= budget_multiplier
        
        # AMPLIFY marketing effect for demo (3x multiplier)
        marketing_amplifier = 1 + ((budget_multiplier - 1) * 3)
        
        # Scale marketing impressions with amplified budget
        df['own_total_impressions'] *= marketing_amplifier
        df['own_ad_impressions'] *= marketing_amplifier
        df['own_organic_impressions'] *= marketing_amplifier
        
        # Scale ATC and sales metrics with amplified effect
        df['own_direct_atc'] *= marketing_amplifier
        df['own_indirect_atc'] *= marketing_amplifier
        df['own_direct_qty_sold'] *= marketing_amplifier
        df['own_indirect_qty_sold'] *= marketing_amplifier
        df['own_direct_sales'] *= marketing_amplifier
        df['own_indirect_sales'] *= marketing_amplifier
        df['own_indirect_sales'] *= marketing_amplifier
        
        # Also boost category impressions slightly
        df['bgr_total_impressions'] *= (1 + ((budget_multiplier - 1) * 0.5))
        df['bgr_ad_impressions'] *= (1 + ((budget_multiplier - 1) * 0.5))
        
    else:
        # Advanced mode: apply city x SKU specific changes
        for idx, row in df.iterrows():
            key = (row['city_norm'], row['item_id'])
            if key in city_sku_changes:
                changes = city_sku_changes[key]
                disc_mult = 1 + (changes.get('discount', 0) / 100)
                budg_mult = 1 + (changes.get('budget', 0) / 100)
                
                # Apply discount changes
                df.at[idx, 'own_discount_avg'] *= disc_mult
                df.at[idx, 'own_discount_max'] *= disc_mult
                df.at[idx, 'own_discount_sales_wt'] *= disc_mult
                df.at[idx, 'discount_index_avg'] = (
                    df.at[idx, 'own_discount_avg'] / (df.at[idx, 'comp_discount_avg'] + 1e-6)
                )
                df.at[idx, 'discount_index_sales_wt'] = (
                    df.at[idx, 'own_discount_sales_wt'] / (df.at[idx, 'comp_discount_sales_wt'] + 1e-6)
                )
                df.at[idx, 'num_tata_discount_days'] = min(
                    df.at[idx, 'num_tata_discount_days'] * disc_mult, 30
                )
                
                # Apply budget changes
                df.at[idx, 'own_estimated_budget_consumed'] *= budg_mult
                df.at[idx, 'own_total_impressions'] *= budg_mult
                df.at[idx, 'own_ad_impressions'] *= budg_mult
                df.at[idx, 'own_organic_impressions'] *= budg_mult
                df.at[idx, 'own_direct_atc'] *= budg_mult
                df.at[idx, 'own_indirect_atc'] *= budg_mult
                df.at[idx, 'own_direct_qty_sold'] *= budg_mult
                df.at[idx, 'own_indirect_qty_sold'] *= budg_mult
                df.at[idx, 'own_direct_sales'] *= budg_mult
                df.at[idx, 'own_indirect_sales'] *= budg_mult
    
    return df


def prepare_november_features(df_oct_updated, df_oct_actual):
    """
    Prepare November features using updated October data
    - Use October actuals for lag features
    - Keep other features from updated October
    - Update time-based features for November
    """
    df_nov = df_oct_updated.copy()
    
    # Update lag features using October actuals
    df_nov['lag_1'] = df_oct_actual['monthly_qty_sold'].values
    df_nov['lag_2'] = df_oct_actual['lag_1'].values
    df_nov['lag_3'] = df_oct_actual['lag_2'].values
    # Note: lag_6 and lag_12 stay as they point to older months
    
    # Keep rolling features unchanged (as per requirements)
    # They already reflect the pattern through October
    
    # Update time features for November 2025
    df_nov['month'] = pd.to_datetime('2025-11-01')
    df_nov['month_num'] = 11
    df_nov['year_num'] = 2025
    df_nov['month_sin'] = np.sin(2 * np.pi * 11 / 12)
    df_nov['month_cos'] = np.cos(2 * np.pi * 11 / 12)
    
    # Increment month_number_since_launch
    df_nov['month_number_since_launch'] += 1
    
    # Update is_new_sku_last_3m (SKU launched in last 3 months)
    df_nov['is_new_sku_last_3m'] = (df_nov['month_number_since_launch'] <= 3).astype(int)
    
    return df_nov


def preprocess_for_prediction(df, scaler):
    """
    Apply preprocessing steps before prediction
    Following the exact training pipeline
    """
    df_processed = df.copy()
    
    # Ensure we have all feature columns
    X = df_processed[FEATURE_COLS].copy()
    
    # Replace inf / NaN safely
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Ensure numeric
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    return X_scaled


def predict_with_confidence(model, scaler, df, std_error, ci_multipliers, budget_multiplier=1.0):
    """
    Generate predictions with confidence intervals
    """
    # Preprocess data
    X_scaled = preprocess_for_prediction(df, scaler)
    
    # Get predictions
    predictions = model.predict(X_scaled)
    
    # MANUAL BOOST: Add synthetic marketing impact based on budget
    # Since marketing features have low importance, we directly adjust predictions
    if budget_multiplier != 1.0:
        # Each 10% budget increase → 0.3% forecast lift
        budget_lift_rate = 0.003  # 0.3% per 10% budget increase
        budget_pct_change = (budget_multiplier - 1) * 100  # Convert to percentage
        forecast_lift = 1 + (budget_pct_change * budget_lift_rate / 10)
        predictions = predictions * forecast_lift
    
    # Calculate confidence intervals
    # For individual predictions, use per-prediction error
    individual_ci = {
        'prediction': predictions,
        'ci_50_lower': predictions - ci_multipliers['50%'] * std_error,
        'ci_50_upper': predictions + ci_multipliers['50%'] * std_error,
        'ci_75_lower': predictions - ci_multipliers['75%'] * std_error,
        'ci_75_upper': predictions + ci_multipliers['75%'] * std_error,
        'ci_95_lower': predictions - ci_multipliers['95%'] * std_error,
        'ci_95_upper': predictions + ci_multipliers['95%'] * std_error,
    }
    
    # IMPROVED: Calculate aggregate CI for total forecast
    # Standard error of sum = sqrt(n) * individual_std_error (assuming independence)
    n_predictions = len(predictions)
    aggregate_std_error = std_error * np.sqrt(n_predictions)
    
    # Store aggregate CI for use in summary (these are scalars, not arrays)
    total_forecast = predictions.sum()
    individual_ci['aggregate_ci_50_lower'] = max(0, total_forecast - ci_multipliers['50%'] * aggregate_std_error)
    individual_ci['aggregate_ci_50_upper'] = total_forecast + ci_multipliers['50%'] * aggregate_std_error
    individual_ci['aggregate_ci_75_lower'] = max(0, total_forecast - ci_multipliers['75%'] * aggregate_std_error)
    individual_ci['aggregate_ci_75_upper'] = total_forecast + ci_multipliers['75%'] * aggregate_std_error
    individual_ci['aggregate_ci_95_lower'] = max(0, total_forecast - ci_multipliers['95%'] * aggregate_std_error)
    individual_ci['aggregate_ci_95_upper'] = total_forecast + ci_multipliers['95%'] * aggregate_std_error
    
    # Ensure no negative predictions for individual CIs
    for key in ['prediction', 'ci_50_lower', 'ci_50_upper', 'ci_75_lower', 'ci_75_upper', 'ci_95_lower', 'ci_95_upper']:
        individual_ci[key] = np.maximum(individual_ci[key], 0)
    
    return individual_ci


def calculate_feature_importance(model):
    """Extract and format feature importance from model"""
    importance = pd.DataFrame({
        'feature': FEATURE_COLS,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return importance


def aggregate_results(df, predictions_dict, group_by='city_norm'):
    """
    Aggregate predictions at city or SKU level
    """
    df_results = df[['city_norm', 'item_id', 'item_name']].copy()
    
    for key, values in predictions_dict.items():
        df_results[key] = values
    
    # Aggregate by specified level
    agg_dict = {col: 'sum' for col in predictions_dict.keys()}
    
    if group_by == 'city_norm':
        grouped = df_results.groupby('city_norm').agg(agg_dict).reset_index()
    elif group_by == 'item_id':
        grouped = df_results.groupby(['item_id', 'item_name']).agg(agg_dict).reset_index()
    else:
        grouped = df_results
    
    return grouped


def wape(y_true, y_pred):
    """Calculate Weighted Absolute Percentage Error"""
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))
