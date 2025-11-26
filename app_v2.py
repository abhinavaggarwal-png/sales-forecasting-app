import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# LOAD MODEL & DATA
# ============================================================================

@st.cache_resource
def load_model_artifacts():
    """Load all model artifacts"""
    try:
        # M+1 MODEL (October forecast)
        # Load model
        with open('models/xgboost_model_final.pkl', 'rb') as f:
            model_m1 = pickle.load(f)
        
        with open('models/scaler_final.pkl', 'rb') as f:
            scaler_m1 = pickle.load(f)
        
        with open('models/feature_list.json', 'r') as f:
            features_m1 = json.load(f)
        
        with open('models/label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        
        with open('models/model_metadata.json', 'r') as f:
            metadata_m1 = json.load(f)
        
        # Load feature importance
        importance = pd.read_csv('models/feature_importance.csv')
        
        # M+2 MODEL (November forecast)
        # ====================================================================
        with open('models/xgboost_model_m2.pkl', 'rb') as f:
            model_m2 = pickle.load(f)
        
        with open('models/scaler_m2.pkl', 'rb') as f:
            scaler_m2 = pickle.load(f)
        
        with open('models/feature_list_m2.json', 'r') as f:
            features_m2 = json.load(f)
        
        with open('models/model_metadata_m2.json', 'r') as f:
            metadata_m2 = json.load(f)

        # M+3 MODEL (December forecast)
        with open('models/xgboost_model_m3.pkl', 'rb') as f:
            model_m3 = pickle.load(f)

        with open('models/scaler_m3.pkl', 'rb') as f:
            scaler_m3 = pickle.load(f)

        with open('models/feature_list_m3.json', 'r') as f:
            features_m3 = json.load(f)

        with open('models/model_metadata_m3.json', 'r') as f:
            metadata_m3 = json.load(f)
        
        return {
            'model_m1': model_m1,
            'scaler_m1': scaler_m1,
            'features_m1': features_m1,
            'metadata_m1': metadata_m1,
            'model_m2': model_m2,
            'scaler_m2': scaler_m2,
            'features_m2': features_m2,
            'metadata_m2': metadata_m2,
            'model_m3': model_m3,
            'scaler_m3': scaler_m3,
            'features_m3': features_m3,
            'metadata_m3': metadata_m3,
            'label_encoder': label_encoder,
            'importance': importance
        }
    
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        return None

@st.cache_data
@st.cache_data
def load_data():
    """Load all data files"""
    try:
        # M+1 October validation
        oct_validation = pd.read_csv('models/october_with_features.csv')
        oct_validation_tcpl = pd.read_csv('models/october_validation_tcpl.csv')
        
        # M+2 October validation (for showing expected accuracy)
        oct_validation_m2 = pd.read_csv('models/october_validation_m2.csv')
        oct_validation_tcpl_m2 = pd.read_csv('models/october_validation_tcpl_m2.csv')
        
        # M+2 November forecast (NEW!)
        nov_forecast_m2 = pd.read_csv('models/november_forecast_m2.csv')
        nov_forecast_tcpl_m2 = pd.read_csv('models/november_forecast_tcpl_m2.csv')

        # M+3 December forecast (NEW!)
        dec_forecast_m3 = pd.read_csv('models/december_forecast_m3.csv')
        dec_forecast_tcpl_m3 = pd.read_csv('models/december_forecast_tcpl_m3.csv')

        # M+3 October validation
        oct_validation_m3 = pd.read_csv('models/october_validation_m3.csv')
        oct_validation_tcpl_m3 = pd.read_csv('models/october_validation_tcpl_m3.csv')
        
        # Other data
        oct_actuals = pd.read_csv('models/october_actuals.csv')
        sept_actuals = pd.read_csv('models/september_actuals.csv')
        growth_features = pd.read_csv('models/growth_features_oct_nov.csv')
        growth_features['month'] = pd.to_datetime(growth_features['month'])
        
        sales_features = pd.read_csv('data/sales_features_oct.csv')
        pricing_features = pd.read_csv('data/pricing_features_oct.csv')
        osa_features = pd.read_csv('data/osa_features_oct.csv')
        marketing_features = pd.read_csv('data/marketing_features_oct.csv')
        
        city_tcpl_mapping = pd.read_csv('data/city_tcpl_mapping.csv')
        item_list = pd.read_csv('data/item_list.csv')
        city_list = pd.read_csv('data/city_list.csv')

        # November/December with features (for slider adjustments)
        nov_with_features = pd.read_csv('models/november_with_features.csv')
        dec_with_features = pd.read_csv('models/december_with_features.csv')
        
        return {
            'oct_validation': oct_validation,
            'oct_validation_tcpl': oct_validation_tcpl,
            'oct_validation_m2': oct_validation_m2,
            'oct_validation_tcpl_m2': oct_validation_tcpl_m2,
            'nov_forecast_m2': nov_forecast_m2,  # NEW
            'nov_forecast_tcpl_m2': nov_forecast_tcpl_m2,  # NEW
            'dec_forecast_m3': dec_forecast_m3,
            'dec_forecast_tcpl_m3': dec_forecast_tcpl_m3,
            'oct_validation_m3': oct_validation_m3,
            'oct_validation_tcpl_m3': oct_validation_tcpl_m3,
            'oct_actuals': oct_actuals,
            'sept_actuals': sept_actuals,
            'growth_features': growth_features,
            'sales_features': sales_features,
            'pricing_features': pricing_features,
            'osa_features': osa_features,
            'marketing_features': marketing_features,
            'city_tcpl_mapping': city_tcpl_mapping,
            'nov_with_features': nov_with_features,
            'dec_with_features': dec_with_features,
            'item_list': item_list,
            'city_list': city_list
        }
    
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None
    
# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_wape(actual, predicted):
    """Calculate WAPE"""
    return np.sum(np.abs(actual - predicted)) / np.sum(np.abs(actual)) * 100

def generate_forecast(data, model, scaler, features, month_name, 
                     discount_change, marketing_change, label_encoder):
    """
    Generate forecast for a given month with slider adjustments
    
    Args:
        data: Dictionary with all data
        model: XGBoost model
        scaler: Fitted scaler
        features: List of feature names
        month_name: 'November' or 'December'
        discount_change: % change in discount (e.g., 0.1 for +10%)
        marketing_change: % change in marketing (e.g., 0.2 for +20%)
        label_encoder: For gc_campaign_type
    
    Returns:
        DataFrame with city × item × forecasted_qty
    """
    
    # This is a simplified version - you'll need to implement the full logic
    # For now, returning October validation as placeholder
    
    st.warning(f"⚠️ {month_name} forecast generation not yet implemented. Showing October data as example.")
    
    # Placeholder: Return October validation data
    return data['oct_validation'].copy()


# ============================================================================
# SLIDER ADJUSTMENT FUNCTION
# ============================================================================

def adjust_features_and_predict(base_data, model, scaler, feature_cols, 
                                discount_change, marketing_change):
    """
    Adjust features based on slider values and generate new predictions
    
    MARKETING APPROACH:
    - Model prediction may not capture marketing impact well (budget alone insufficient)
    - Apply real-world elasticity (0.2423) from historical MoM analysis
    - Real data: 56.8% of budget increases led to sales increases, avg +21.04%
    
    PRICING APPROACH:
    - Model captures price elasticity through price features
    - Direct feature adjustment works well
    
    Args:
        oct_data: Original October data with all features
        model: Trained XGBoost model
        scaler: Fitted StandardScaler
        feature_cols: List of feature column names
        discount_change: Discount % change (e.g., 10 for +10%)
        marketing_change: Marketing % change (e.g., 20 for +20%)
    
    Returns:
        DataFrame with updated forecasts
    """
    
    adjusted_data = base_data.copy()
    X = adjusted_data[feature_cols].copy()
    
    # ========================================================================
    # APPLY DISCOUNT ADJUSTMENTS
    # ========================================================================
    if discount_change != 0:
        discount_multiplier = 1 - (discount_change / 100)
        
        # Adjust price features
        if 'own_price_avg' in X.columns:
            X['own_price_avg'] = X['own_price_avg'] * discount_multiplier
        
        if 'own_price_sales_wt' in X.columns:
            X['own_price_sales_wt'] = X['own_price_sales_wt'] * discount_multiplier
        
        # own_mrp_avg stays constant (not adjusted)
    
    # ========================================================================
    # APPLY MARKETING ADJUSTMENTS (Feature Level)
    # ========================================================================
    marketing_boost = 0
    
    if marketing_change != 0:
        # Adjust budget feature (model will see the change)
        if 'own_estimated_budget_consumed_v2' in X.columns:
            X['own_estimated_budget_consumed_v2'] *= (1 + marketing_change / 100)
        
        # Calculate expected boost from real-world historical data
        # Real elasticity: 0.2423 (from MoM analysis)
        # Interpretation: +10% budget → +2.423% sales
        marketing_boost = marketing_change * 0.2423 / 100
    
    # ========================================================================
    # HANDLE MISSING VALUES & ENSURE NUMERIC
    # ========================================================================
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    
    # ========================================================================
    # SCALE & PREDICT
    # ========================================================================
    X_scaled = scaler.transform(X)
    new_predictions = model.predict(X_scaled)
    new_predictions = np.maximum(new_predictions, 0)
    
    # ========================================================================
    # APPLY REAL-WORLD MARKETING BOOST (Post-Prediction Adjustment)
    # ========================================================================
    if marketing_change != 0:
        # Apply historical elasticity to predictions
        new_predictions = new_predictions * (1 + marketing_boost)
        
        # Show transparency message
        expected_change = marketing_boost * 100
        st.info(f"""
        📊 **Marketing Impact Applied**  
        Historical elasticity: **0.2423** (from month-over-month analysis)  
        Expected sales change: **{expected_change:+.2f}%** for {marketing_change:+d}% budget change  
        
        *Note: Model predictions adjusted using real-world historical patterns where 56.8% of budget increases led to sales growth.*
        """)
    
    adjusted_data['forecasted_qty'] = new_predictions
    
    return adjusted_data

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Load artifacts
    artifacts = load_model_artifacts()
    data = load_data()
    
    if artifacts is None or data is None:
        st.error("Failed to load model or data. Please check file paths.")
        return
    
    # Unpack artifacts
    model_m1 = artifacts['model_m1']
    scaler_m1 = artifacts['scaler_m1']
    features_m1 = artifacts['features_m1']
    model_m2 = artifacts['model_m2']
    scaler_m2 = artifacts['scaler_m2']
    features_m2 = artifacts['features_m2']
    label_encoder = artifacts['label_encoder']
    metadata_m1 = artifacts['metadata_m1']
    metadata_m2 = artifacts['metadata_m2']
    importance = artifacts['importance']
    model_m3 = artifacts['model_m3']      # ADD THIS
    scaler_m3 = artifacts['scaler_m3']    # ADD THIS
    features_m3 = artifacts['features_m3']  # ADD THIS
    metadata_m3 = artifacts['metadata_m3']  # ADD THIS
    
    # ========================================================================
    # SIDEBAR
    # ========================================================================
    
    st.sidebar.title("⚙️ Configuration")
    
    # Tab selection
    forecast_tab = st.sidebar.radio(
        "Select Forecast Period:",
        ["October 2025 (Validation)", "November 2025 (M+2)", "December 2025 (M+3)"]
    )
    
    st.sidebar.markdown("---")
    
    # ========================================================================
    # INPUT CONTROLS
    # ========================================================================
    
    st.sidebar.subheader("🎮 Input Controls")
    
    # Discount slider
    st.sidebar.markdown("### 🏷️ Discount Strategy")
    discount_change = st.sidebar.slider(
        "Discount % Change (from October baseline)",
        min_value=-50,
        max_value=50,
        value=0,
        step=5,
        help="Adjust your pricing discount strategy"
    )
    st.sidebar.caption(f"Change: {discount_change:+d}%")
    
    # Marketing slider
    st.sidebar.markdown("### 💰 Marketing Budget")
    marketing_change = st.sidebar.slider(
        "Budget % Change (from October baseline)",
        min_value=-50,
        max_value=100,
        value=0,
        step=10,
        help="Adjust your marketing spend"
    )
    st.sidebar.caption(f"Change: {marketing_change:+d}%")
    
    # ========================================================================
    # MAIN CONTENT
    # ========================================================================
    
    st.title("📊 Sales Forecasting Dashboard")
    st.markdown(f"### {forecast_tab}")
    
    # ========================================================================
    # OCTOBER VALIDATION TAB
    # ========================================================================

    if forecast_tab == "October 2025 (Validation)":

        oct_data = data['oct_validation'].copy()
        
        # ========================================================================
        # APPLY SLIDER ADJUSTMENTS
        # ========================================================================
        if discount_change != 0 or marketing_change != 0:
            st.info(f"🎯 Applying adjustments: Discount {discount_change:+d}%, Marketing {marketing_change:+d}%")
            
            # Adjust features and get new predictions
            oct_data = adjust_features_and_predict(
                base_data=oct_data,
                model=model_m1,
                scaler=scaler_m1,
                feature_cols=features_m1,
                discount_change=discount_change,
                marketing_change=marketing_change
            )
            
            st.success("✅ Forecast updated with your strategy adjustments!")
        
        # ========================================================================
        # LOAD TCPL DATA
        # ========================================================================
        try:
            oct_data_tcpl = data['oct_validation_tcpl'].copy()
            
            # Also apply adjustments to TCPL data if sliders moved
            if discount_change != 0 or marketing_change != 0:
                # Need to reconstruct TCPL from adjusted city data
                # Add TCPL mapping
                city_tcpl = data['city_tcpl_mapping']
                oct_data_with_tcpl = oct_data.merge(
                    city_tcpl[['city_norm', 'tcpl_plant_code', 'fe_city_name']],
                    on='city_norm',
                    how='left'
                )
                
                # Aggregate to TCPL × SKU
                tcpl_forecast = oct_data_with_tcpl.groupby(['tcpl_plant_code', 'item_id']).agg({
                    'forecasted_qty': 'sum',
                    'actual_qty': 'sum'
                }).reset_index()
                
                # Handle NCR split (50-50 between 1214 and 1476)
                ncr_mask = oct_data_with_tcpl['fe_city_name'] == 'NCR'
                if ncr_mask.sum() > 0:
                    # Get NCR TCPL code
                    ncr_tcpl_current = oct_data_with_tcpl[ncr_mask]['tcpl_plant_code'].iloc[0]
                    ncr_rows = tcpl_forecast[tcpl_forecast['tcpl_plant_code'] == ncr_tcpl_current].copy()
                    tcpl_forecast_non_ncr = tcpl_forecast[tcpl_forecast['tcpl_plant_code'] != ncr_tcpl_current].copy()
                    
                    # Split NCR
                    ncr_1214 = ncr_rows.copy()
                    ncr_1214['tcpl_plant_code'] = 1214
                    ncr_1214['forecasted_qty'] = ncr_1214['forecasted_qty'] * 0.5
                    ncr_1214['actual_qty'] = ncr_1214['actual_qty'] * 0.5
                    
                    ncr_1476 = ncr_rows.copy()
                    ncr_1476['tcpl_plant_code'] = 1476
                    ncr_1476['forecasted_qty'] = ncr_1476['forecasted_qty'] * 0.5
                    ncr_1476['actual_qty'] = ncr_1476['actual_qty'] * 0.5
                    
                    # Combine
                    tcpl_forecast = pd.concat([tcpl_forecast_non_ncr, ncr_1214, ncr_1476], ignore_index=True)
                    
                    # Re-aggregate
                    tcpl_forecast = tcpl_forecast.groupby(['tcpl_plant_code', 'item_id']).agg({
                        'forecasted_qty': 'sum',
                        'actual_qty': 'sum'
                    }).reset_index()
                
                oct_data_tcpl = tcpl_forecast
            
            tcpl_available = True
            
        except Exception as e:
            st.warning(f"⚠️ TCPL data not available: {e}")
            tcpl_available = False

        # ========================================================================
        # CALCULATE METRICS - City × SKU Level
        # ========================================================================
        total_actual = oct_data['actual_qty'].sum()
        total_forecast = oct_data['forecasted_qty'].sum()
        wape_city_sku = calculate_wape(oct_data['actual_qty'], oct_data['forecasted_qty'])

        # ========================================================================
        # CALCULATE METRICS - TCPL × SKU Level
        # ========================================================================
        if tcpl_available:
            tcpl_total_actual = oct_data_tcpl['actual_qty'].sum()
            tcpl_total_forecast = oct_data_tcpl['forecasted_qty'].sum()
            wape_tcpl_sku = calculate_wape(oct_data_tcpl['actual_qty'], oct_data_tcpl['forecasted_qty'])
            tcpl_accuracy = 100 - wape_tcpl_sku
        
        # ========================================================================
        # DISPLAY TOP METRICS - Two Rows
        # ========================================================================
        
        st.markdown("### 📊 Forecast Accuracy Metrics")
        
        # Row 1: City × SKU Level
        st.markdown("**City × SKU Level**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Actual Sales",
                f"{total_actual:,.0f}",
                help="Actual sales in October 2025"
            )
        
        with col2:
            st.metric(
                "Total Forecast",
                f"{total_forecast:,.0f}",
                delta=f"{total_forecast - total_actual:,.0f}",
                help="Forecasted sales"
            )
        
        with col3:
            st.metric(
                "WAPE",
                f"{wape_city_sku:.2f}%",
                help="City × SKU level WAPE"
            )
        
        with col4:
            accuracy = 100 - wape_city_sku
            st.metric(
                "Accuracy",
                f"{accuracy:.2f}%",
                help="Forecast accuracy"
            )
        
        # Row 2: TCPL × SKU Level (if available)
        if tcpl_available:
            st.markdown("**TCPL × SKU Level**")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Actual Sales",
                    f"{tcpl_total_actual:,.0f}",
                    help="Actual sales at TCPL × SKU level"
                )
            
            with col2:
                st.metric(
                    "Total Forecast",
                    f"{tcpl_total_forecast:,.0f}",
                    delta=f"{tcpl_total_forecast - tcpl_total_actual:,.0f}",
                    help="Forecasted sales at TCPL × SKU level"
                )
            
            with col3:
                st.metric(
                    "WAPE",
                    f"{wape_tcpl_sku:.2f}%",
                    delta=f"{wape_tcpl_sku - wape_city_sku:+.2f}pp",
                    delta_color="inverse",
                    help="TCPL × SKU level WAPE"
                )
            
            with col4:
                st.metric(
                    "Accuracy",
                    f"{tcpl_accuracy:.2f}%",
                    delta=f"{tcpl_accuracy - (100 - wape_city_sku):+.2f}pp",
                    help="Forecast accuracy at TCPL level"
                )
        
        # ========================================================================
        # CONFIDENCE INTERVALS - Two Rows
        # ========================================================================
        
        st.markdown("---")
        st.markdown("### 📊 Confidence Intervals")
        
        # Calculate standard error for City × SKU
        oct_data['error'] = oct_data['forecasted_qty'] - oct_data['actual_qty']
        std_error_total_city = np.sqrt((oct_data['error']**2).sum())

        # Calculate WAPEs
        #wape_city = calculate_wape(oct_data['actual_qty'], oct_data['forecasted_qty'])
        #wape_tcpl = calculate_wape(oct_data_tcpl['actual_qty'], oct_data_tcpl['forecasted_qty'])

        # Scale TCPL by WAPE ratio
        #wape_ratio = wape_tcpl / wape_city
        #std_error_tcpl = std_error_base * wape_ratio
        
        # Row 1: City × SKU CI
        st.markdown("**City × SKU Level**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            lower = total_forecast - 0.67 * std_error_total_city
            upper = total_forecast + 0.67 * std_error_total_city
            st.info(f"**50% CI:** {lower:,.0f} - {upper:,.0f}")
        
        with col2:
            lower = total_forecast - 1.15 * std_error_total_city
            upper = total_forecast + 1.15 * std_error_total_city
            st.info(f"**75% CI:** {lower:,.0f} - {upper:,.0f}")
        
        with col3:
            lower = total_forecast - 1.96 * std_error_total_city
            upper = total_forecast + 1.96 * std_error_total_city
            st.info(f"**95% CI:** {lower:,.0f} - {upper:,.0f}")
        
        # Row 2: TCPL × SKU CI (if available)
        if tcpl_available:
            st.markdown("**TCPL × SKU Level**")
            
            # Calculate standard error for TCPL × SKU
            oct_data_tcpl['error'] = oct_data_tcpl['forecasted_qty'] - oct_data_tcpl['actual_qty']
            std_error_total_tcpl = np.sqrt((oct_data_tcpl['error']**2).sum())
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                lower = tcpl_total_forecast - 0.67 * std_error_total_tcpl
                upper = tcpl_total_forecast + 0.67 * std_error_total_tcpl
                st.info(f"**50% CI:** {lower:,.0f} - {upper:,.0f}")
            
            with col2:
                lower = tcpl_total_forecast - 1.15 * std_error_total_tcpl
                upper = tcpl_total_forecast + 1.15 * std_error_total_tcpl
                st.info(f"**75% CI:** {lower:,.0f} - {upper:,.0f}")
            
            with col3:
                lower = tcpl_total_forecast - 1.96 * std_error_total_tcpl
                upper = tcpl_total_forecast + 1.96 * std_error_total_tcpl
                st.info(f"**95% CI:** {lower:,.0f} - {upper:,.0f}")
        
        # ========================================================================
        # VISUALIZATION TABS
        # ========================================================================
        
        st.markdown("---")
        viz_tab1, viz_tab2, viz_tab3, viz_tab4, viz_tab5 = st.tabs(["🏭 TCPL Analysis", "📍 City Analysis", "📦 SKU Analysis", "🔑 Key Drivers", "📋 Detailed Data"])
    
        # ... rest of your viz tabs code ...  
        with viz_tab1:
            st.subheader("TCPL Warehouse-Level Forecast")
            
            if tcpl_available:
                # Aggregate TCPL data for display
                tcpl_summary = oct_data_tcpl.groupby('tcpl_plant_code').agg({
                    'forecasted_qty': 'sum',
                    'actual_qty': 'sum'
                }).reset_index()
                
                # Calculate metrics
                tcpl_summary['error'] = tcpl_summary['forecasted_qty'] - tcpl_summary['actual_qty']
                tcpl_summary['wape'] = (abs(tcpl_summary['error']) / tcpl_summary['actual_qty'] * 100).round(2)
                tcpl_summary = tcpl_summary.sort_values('forecasted_qty', ascending=False)
                
                # Format TCPL code as integer
                tcpl_summary['tcpl_plant_code'] = tcpl_summary['tcpl_plant_code'].astype(int)
                
                # Top 10 TCPL Chart
                top_tcpl = tcpl_summary.head(10)
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=top_tcpl['tcpl_plant_code'].astype(str),
                    y=top_tcpl['actual_qty'],
                    name='Actual',
                    marker_color='lightcoral'
                ))
                fig.add_trace(go.Bar(
                    x=top_tcpl['tcpl_plant_code'].astype(str),
                    y=top_tcpl['forecasted_qty'],
                    name='Forecast',
                    marker_color='darkred'
                ))
                
                fig.update_layout(
                    barmode='group',
                    xaxis_title="TCPL Plant Code",
                    yaxis_title="Sales Volume",
                    height=400,
                    title="Top 10 TCPL Plants by Forecast Volume",
                    xaxis=dict(
                        type='category',
                        categoryorder='total descending'
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Key metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    total_tcpls = tcpl_summary['tcpl_plant_code'].nunique()
                    st.metric("Total TCPL Plants", total_tcpls)
                
                with col2:
                    st.metric("Total TCPL Forecast", f"{tcpl_total_forecast:,.0f}")
                
                with col3:
                    st.metric("TCPL-Level WAPE", f"{wape_tcpl_sku:.2f}%")
                
                # Detailed TCPL table
                st.markdown("---")
                st.markdown("### 📊 Detailed TCPL Data")
                
                display_tcpl = tcpl_summary[['tcpl_plant_code', 'forecasted_qty', 'actual_qty', 'error', 'wape']].copy()
                display_tcpl.columns = ['TCPL Code', 'Forecast', 'Actual', 'Error', 'WAPE %']
                
                st.dataframe(
                    display_tcpl.style.format({
                        'TCPL Code': '{:,}',
                        'Forecast': '{:,.0f}',
                        'Actual': '{:,.0f}',
                        'Error': '{:+,.0f}',
                        'WAPE %': '{:.2f}%'
                    }),
                    use_container_width=True,
                    height=400,
                    hide_index=True
                )
            else:
                st.warning("⚠️ TCPL data not available")

        with viz_tab2:
            st.subheader("Top 10 Cities by Forecast Volume")
            
            city_summary = oct_data.groupby('city_norm').agg({
                'forecasted_qty': 'sum',
                'actual_qty': 'sum'
            }).reset_index().sort_values('forecasted_qty', ascending=False).head(10)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=city_summary['city_norm'],
                y=city_summary['actual_qty'],
                name='Actual',
                marker_color='lightblue'
            ))
            fig.add_trace(go.Bar(
                x=city_summary['city_norm'],
                y=city_summary['forecasted_qty'],
                name='Forecast',
                marker_color='darkblue'
            ))
            
            fig.update_layout(
                barmode='group',
                xaxis_title="City",
                yaxis_title="Sales Volume",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show detailed city table (OPEN BY DEFAULT)
            st.markdown("---")
            st.markdown("### 📊 Detailed City Data")

            # Get all cities (not just top 10)
            city_detail = oct_data.groupby('city_norm').agg({
                'forecasted_qty': 'sum',
                'actual_qty': 'sum'
            }).reset_index().sort_values('forecasted_qty', ascending=False)

            city_detail['error'] = city_detail['forecasted_qty'] - city_detail['actual_qty']
            city_detail['error_pct'] = (city_detail['error'] / city_detail['actual_qty'] * 100).round(2)
            city_detail['wape'] = (abs(city_detail['error']) / city_detail['actual_qty'] * 100).round(2)
            city_detail.columns = ['City', 'Forecast', 'Actual', 'Error', 'Error %', 'WAPE %']

            st.dataframe(
                city_detail.style.format({
                    'Forecast': '{:,.0f}',
                    'Actual': '{:,.0f}',
                    'Error': '{:+,.0f}',
                    'Error %': '{:+.2f}%',
                    'WAPE %': '{:.2f}%'
                }),
                use_container_width=True,
                height=400,
                hide_index=True  # ADD THIS LINE
            )
        
        with viz_tab3:
            st.subheader("Top 10 SKUs by Forecast Volume")
            
            # Load SKU names
            try:
                sku_names = pd.read_csv('data/sku_names.csv')
            except:
                st.warning("⚠️ SKU names file not found. Using item IDs.")
                sku_names = pd.DataFrame({
                    'item_id': oct_data['item_id'].unique(),
                    'item_name': oct_data['item_id'].unique().astype(str)
                })
            
            # Aggregate by item_id
            sku_summary = oct_data.groupby('item_id').agg({
                'forecasted_qty': 'sum',
                'actual_qty': 'sum'
            }).reset_index()
            
            # Add SKU names
            sku_summary = sku_summary.merge(sku_names, on='item_id', how='left')
            sku_summary['item_name'] = sku_summary['item_name'].fillna('Unknown SKU')
            
            # Get top 10
            sku_summary_top10 = sku_summary.sort_values('forecasted_qty', ascending=False).head(10)
            
            # Create horizontal bar chart
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                y=sku_summary_top10['item_name'],
                x=sku_summary_top10['actual_qty'],
                name='Baseline',
                orientation='h',
                marker_color='#90EE90',
                text=sku_summary_top10['actual_qty'].apply(lambda x: f'{x/1000:.0f}k' if x >= 1000 else f'{x:.0f}'),
                textposition='outside',
                textfont=dict(size=10)
            ))
            
            fig.add_trace(go.Bar(
                y=sku_summary_top10['item_name'],
                x=sku_summary_top10['forecasted_qty'],
                name='Updated Forecast',
                orientation='h',
                marker_color='#006400',
                text=sku_summary_top10['forecasted_qty'].apply(lambda x: f'{x/1000:.0f}k' if x >= 1000 else f'{x:.0f}'),
                textposition='outside',
                textfont=dict(size=10)
            ))
            
            fig.update_layout(
                barmode='group',
                xaxis_title="Forecasted Quantity",
                yaxis_title="SKU",
                height=500,
                yaxis={'categoryorder': 'total ascending'},
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                xaxis=dict(
                    tickformat=',',
                    showgrid=True,
                    gridcolor='lightgray'
                ),
                plot_bgcolor='white',
                margin=dict(l=200)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show detailed table (OPEN BY DEFAULT)
            st.markdown("---")
            st.markdown("### 📊 Detailed SKU Data")
            
            display_df = sku_summary[['item_name', 'actual_qty', 'forecasted_qty']].copy()
            display_df['error'] = display_df['forecasted_qty'] - display_df['actual_qty']
            display_df['error_pct'] = (display_df['error'] / display_df['actual_qty'] * 100).round(2)
            display_df = display_df.sort_values('forecasted_qty', ascending=False)
            display_df.columns = ['SKU Name', 'Actual', 'Forecast', 'Error', 'Error %']
            
            st.dataframe(
                display_df.style.format({
                    'Actual': '{:,.0f}',
                    'Forecast': '{:,.0f}',
                    'Error': '{:+,.0f}',
                    'Error %': '{:+.2f}%'
                }),
                use_container_width=True,
                height=400,
                hide_index=True  # ADD THIS LINE

            )
                    
        
        with viz_tab4:
            st.subheader("Top 20 Feature Importances")
            
            top_features = importance.head(20)
            
            fig = px.bar(
                top_features,
                x='importance',
                y='feature',
                orientation='h',
                title="Feature Importance in Sales Forecasting"
            )
            
            fig.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Controllable features highlight
            # Calculate from feature importance
            pricing_features = ['own_price_avg', 'own_mrp_avg', 'own_price_sales_wt']
            marketing_features = ['own_estimated_budget_consumed_v2', 'gc_campaign_type', 'comp_total_impressions', 'comp_ad_impressions', 'comp_ad_ratio']
            osa_features = ['own_wt_osa_avg']

            importance_df = artifacts['importance']

            pricing_imp = importance_df[importance_df['feature'].isin(pricing_features)]['importance'].sum() * 100
            marketing_imp = importance_df[importance_df['feature'].isin(marketing_features)]['importance'].sum() * 100
            osa_imp = importance_df[importance_df['feature'].isin(osa_features)]['importance'].sum() * 100
            total_controllable = pricing_imp + marketing_imp + osa_imp

            st.info(f"""
            **Controllable Features Impact:**  
            - Pricing: {pricing_imp:.2f}%  
            - Marketing: {marketing_imp:.2f}%  
            - OSA: {osa_imp:.2f}%  
            - **Total: {total_controllable:.2f}%**
            """)


        with viz_tab5:
            st.subheader("📋 Detailed Forecast Data - October 2025")
            
            # Prepare data with TCPL mapping
            oct_detail = oct_data[['city_norm', 'item_id', 'forecasted_qty', 'actual_qty']].copy()
            oct_detail = oct_detail.merge(
                data['city_tcpl_mapping'][['city_norm', 'tcpl_plant_code', 'fe_city_name']],
                on='city_norm',
                how='left'
            )
            
            # Reorder columns
            oct_detail = oct_detail[['tcpl_plant_code', 'fe_city_name', 'city_norm', 'item_id', 'actual_qty', 'forecasted_qty']]
            oct_detail = oct_detail.sort_values(['tcpl_plant_code', 'city_norm', 'item_id'])
            
            # Filters
            st.markdown("#### 🔍 Filters")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                tcpl_opts = ['All'] + sorted(oct_detail['tcpl_plant_code'].dropna().unique().astype(int).tolist())
                sel_tcpl = st.selectbox("TCPL Plant", tcpl_opts, key='oct_detail_tcpl')
            
            with col2:
                city_opts = ['All'] + sorted(oct_detail['city_norm'].dropna().unique().tolist())
                sel_city = st.selectbox("City", city_opts, key='oct_detail_city')
            
            with col3:
                sku_opts = ['All'] + sorted(oct_detail['item_id'].dropna().unique().tolist())
                sel_sku = st.selectbox("SKU", sku_opts, key='oct_detail_sku')
            
            # Apply filters
            filtered = oct_detail.copy()
            if sel_tcpl != 'All':
                filtered = filtered[filtered['tcpl_plant_code'] == sel_tcpl]
            if sel_city != 'All':
                filtered = filtered[filtered['city_norm'] == sel_city]
            if sel_sku != 'All':
                filtered = filtered[filtered['item_id'] == sel_sku]
            
            st.markdown(f"**Showing {len(filtered):,} records**")
            
            # Display
            display_df = filtered.copy()
            display_df.columns = ['TCPL Code', 'City Name', 'City Norm', 'Item ID', 'Actual', 'Forecast']
            
            st.dataframe(display_df, use_container_width=True, height=400, hide_index=True)
            
            # Download
            csv = filtered.to_csv(index=False)
            st.download_button(
                label="📥 Download October Data (CSV)",
                data=csv,
                file_name="october_2025_forecast_detail.csv",
                mime="text/csv",
                key='download_oct_detail'
            )
    
    # ========================================================================
    # NOVEMBER FORECAST TAB
    # ========================================================================
    
    # ========================================================================
    # NOVEMBER 2025 TAB (M+2 FORECAST)
    # ========================================================================

    elif forecast_tab == "November 2025 (M+2)":
        
        # Load November forecast
        #nov_data = data['nov_forecast_m2'].copy()
        nov_data = data['nov_with_features'].copy()

        
         #Apply slider adjustments using MODEL
        if discount_change != 0 or marketing_change != 0:
            st.info(f"🎯 Applying adjustments: Discount {discount_change:+d}%, Marketing {marketing_change:+d}%")
            
            # Use model to re-predict with adjusted features
            nov_data = adjust_features_and_predict(
                base_data=nov_data,
                model=model_m2,
                scaler=scaler_m2,
                feature_cols=features_m2,
                discount_change=discount_change,
                marketing_change=marketing_change
            )
            
            if marketing_change != 0:
                st.info(f"""
                📊 **Marketing Impact Applied**  
                Historical elasticity: **0.2423**  
                Expected sales change: **{marketing_change * 0.2423 / 100 * 100:+.2f}%**
                """)
            
            st.success("✅ Forecast updated using M+2 model with adjusted features!")
    
        # Calculate totals
        total_forecast = nov_data['forecasted_qty'].sum()
        
        # Recalculate TCPL from adjusted city data
        try:
            # Add TCPL mapping
            city_tcpl = data['city_tcpl_mapping']
            nov_data_with_tcpl = nov_data.merge(
                city_tcpl[['city_norm', 'tcpl_plant_code', 'fe_city_name']],
                on='city_norm',
                how='left'
            )
            
            # Aggregate to TCPL × SKU
            nov_data_tcpl = nov_data_with_tcpl.groupby(['tcpl_plant_code', 'item_id']).agg({
                'forecasted_qty': 'sum'
            }).reset_index()
            
            # Handle NCR split (50-50 between 1214 and 1476)
            ncr_mask = nov_data_with_tcpl['fe_city_name'] == 'NCR'
            if ncr_mask.sum() > 0:
                ncr_tcpl_current = nov_data_with_tcpl[ncr_mask]['tcpl_plant_code'].iloc[0]
                ncr_rows = nov_data_tcpl[nov_data_tcpl['tcpl_plant_code'] == ncr_tcpl_current].copy()
                nov_data_tcpl_non_ncr = nov_data_tcpl[nov_data_tcpl['tcpl_plant_code'] != ncr_tcpl_current].copy()
                
                # Split NCR
                ncr_1214 = ncr_rows.copy()
                ncr_1214['tcpl_plant_code'] = 1214
                ncr_1214['forecasted_qty'] = ncr_1214['forecasted_qty'] * 0.5
                
                ncr_1476 = ncr_rows.copy()
                ncr_1476['tcpl_plant_code'] = 1476
                ncr_1476['forecasted_qty'] = ncr_1476['forecasted_qty'] * 0.5
                
                # Combine
                nov_data_tcpl = pd.concat([nov_data_tcpl_non_ncr, ncr_1214, ncr_1476], ignore_index=True)
                
                # Re-aggregate
                nov_data_tcpl = nov_data_tcpl.groupby(['tcpl_plant_code', 'item_id']).agg({
                    'forecasted_qty': 'sum'
                }).reset_index()
            
            tcpl_available = True
        except Exception as e:
            st.warning(f"⚠️ TCPL aggregation error: {e}")
            tcpl_available = False
        
        # Calculate totals
        total_forecast = nov_data['forecasted_qty'].sum()
        
        if tcpl_available:
            tcpl_total_forecast = nov_data_tcpl['forecasted_qty'].sum()
        
        # ========================================================================
        # DISPLAY METRICS
        # ========================================================================
        
        st.markdown("### 📅 November 2025 Forecast (M+2)")
        
        st.info("""
        **Forecast Information:**
        - Using data available till: **September 2025**
        - Forecast horizon: **2 months ahead (M+2)**
        - No actuals available (future month)
        - Expected accuracy based on October validation below
        """)
        
        # Show forecast totals
        st.markdown("### 📊 November Forecast")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "City × SKU Forecast",
                f"{total_forecast:,.0f}",
                help="November 2025 forecast at City × SKU level"
            )
        
        with col2:
            if tcpl_available:
                st.metric(
                    "TCPL × SKU Forecast",
                    f"{tcpl_total_forecast:,.0f}",
                    help="November 2025 forecast at TCPL × SKU level"
                )
        
        # ========================================================================
        # SHOW M+2 VALIDATION METRICS (from October)
        # ========================================================================
        
        st.markdown("---")
        st.markdown("### 🎯 Expected Accuracy (Based on October Validation)")
        
        st.info("The model was validated on October 2025 using August 2025 data. These metrics indicate expected accuracy for November forecast.")
        
        # Load October M+2 validation
        oct_val_m2 = data['oct_validation_m2']
        oct_val_tcpl_m2 = data['oct_validation_tcpl_m2']
        
        # Calculate October M+2 metrics
        wape_city_m2 = calculate_wape(oct_val_m2['actual_qty'], oct_val_m2['forecasted_qty'])
        wape_tcpl_m2 = calculate_wape(oct_val_tcpl_m2['actual_qty'], oct_val_tcpl_m2['forecasted_qty'])
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "City × SKU Expected Accuracy",
                f"{100 - wape_city_m2:.2f}%",
                help="Based on October validation"
            )
        
        with col2:
            st.metric(
                "City × SKU Expected WAPE",
                f"{wape_city_m2:.2f}%",
                delta=f"{wape_city_m2 - metadata_m1.get('city_sku_wape', 15.87):+.2f}pp vs M+1",
                delta_color="inverse"
            )
        
        with col3:
            if tcpl_available:
                st.metric(
                    "TCPL × SKU Expected Accuracy",
                    f"{100 - wape_tcpl_m2:.2f}%",
                    help="Based on October validation"
                )
        
        with col4:
            if tcpl_available:
                st.metric(
                    "TCPL × SKU Expected WAPE",
                    f"{wape_tcpl_m2:.2f}%",
                    delta=f"{wape_tcpl_m2 - 12.32:+.2f}pp vs M+1",
                    delta_color="inverse"
                )
        
        # ========================================================================
        # CONFIDENCE INTERVALS (Wider for M+2)
        # ========================================================================
        
        # ========================================================================
        # CONFIDENCE INTERVALS (WAPE-Proportional)
        # ========================================================================

        st.markdown("---")
        st.markdown("### 📊 Forecast Confidence Intervals")

        st.info("Confidence intervals are proportional to forecast accuracy. TCPL intervals are tighter, reflecting ~20% better WAPE from aggregation.")

        # Load October M+2 validation
        oct_val_m2 = data['oct_validation_m2']
        oct_val_tcpl_m2 = data['oct_validation_tcpl_m2']

        # Calculate actual errors for base standard error
        oct_val_m2['error'] = oct_val_m2['forecasted_qty'] - oct_val_m2['actual_qty']
        std_error_base = np.sqrt((oct_val_m2['error']**2).sum())

        # Calculate WAPEs
        wape_city_m2 = calculate_wape(oct_val_m2['actual_qty'], oct_val_m2['forecasted_qty'])
        wape_tcpl_m2 = calculate_wape(oct_val_tcpl_m2['actual_qty'], oct_val_tcpl_m2['forecasted_qty'])

        # TCPL standard error scaled by WAPE ratio (proportionally tighter)
        wape_ratio = wape_tcpl_m2 / wape_city_m2  # e.g., 15.19/18.91 = 0.803
        std_error_tcpl = std_error_base * wape_ratio

        # City × SKU CI
        st.markdown("**City × SKU Level**")
        col1, col2, col3 = st.columns(3)

        with col1:
            lower = total_forecast - 0.67 * std_error_base
            upper = total_forecast + 0.67 * std_error_base
            st.info(f"**50% CI:** {lower:,.0f} - {upper:,.0f}")

        with col2:
            lower = total_forecast - 1.15 * std_error_base
            upper = total_forecast + 1.15 * std_error_base
            st.info(f"**75% CI:** {lower:,.0f} - {upper:,.0f}")

        with col3:
            lower = total_forecast - 1.96 * std_error_base
            upper = total_forecast + 1.96 * std_error_base
            st.info(f"**95% CI:** {lower:,.0f} - {upper:,.0f}")

        # TCPL × SKU CI (Proportionally tighter)
        if tcpl_available:
            st.markdown("**TCPL × SKU Level**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                lower = tcpl_total_forecast - 0.67 * std_error_tcpl
                upper = tcpl_total_forecast + 0.67 * std_error_tcpl
                st.info(f"**50% CI:** {lower:,.0f} - {upper:,.0f}")
            
            with col2:
                lower = tcpl_total_forecast - 1.15 * std_error_tcpl
                upper = tcpl_total_forecast + 1.15 * std_error_tcpl
                st.info(f"**75% CI:** {lower:,.0f} - {upper:,.0f}")
            
            with col3:
                lower = tcpl_total_forecast - 1.96 * std_error_tcpl
                upper = tcpl_total_forecast + 1.96 * std_error_tcpl
                st.info(f"**95% CI:** {lower:,.0f} - {upper:,.0f}")
        
        # ========================================================================
        # VISUALIZATION TABS (Same structure as October)
        # ========================================================================
        
        st.markdown("---")
        viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs(["🏭 TCPL Forecast", "📍 City Forecast", "📦 SKU Forecast", "📋 Detailed Data"])
        
        with viz_tab1:
            st.subheader("TCPL Warehouse-Level Forecast")
            
            if tcpl_available:
                # Aggregate TCPL
                tcpl_summary = nov_data_tcpl.groupby('tcpl_plant_code').agg({
                    'forecasted_qty': 'sum'
                }).reset_index().sort_values('forecasted_qty', ascending=False)
                
                tcpl_summary['tcpl_plant_code'] = tcpl_summary['tcpl_plant_code'].astype(int)
                
                # Top 10 chart
                top_tcpl = tcpl_summary.head(10)
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=top_tcpl['tcpl_plant_code'].astype(str),
                    y=top_tcpl['forecasted_qty'],
                    name='November Forecast',
                    marker_color='darkgreen'
                ))
                
                fig.update_layout(
                    xaxis_title="TCPL Plant Code",
                    yaxis_title="Forecasted Sales",
                    height=400,
                    title="Top 10 TCPL Plants - November Forecast",
                    xaxis=dict(type='category', categoryorder='total descending')
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total TCPL Plants", tcpl_summary['tcpl_plant_code'].nunique())
                with col2:
                    st.metric("Total Forecast", f"{tcpl_total_forecast:,.0f}")
                
                # Detailed table
                st.markdown("---")
                st.markdown("### 📊 Detailed TCPL Forecast")
                
                display_tcpl = tcpl_summary[['tcpl_plant_code', 'forecasted_qty']].copy()
                display_tcpl.columns = ['TCPL Code', 'November Forecast']
                
                st.dataframe(
                    display_tcpl.style.format({
                        'TCPL Code': '{:,}',
                        'November Forecast': '{:,.0f}'
                    }),
                    use_container_width=True,
                    height=400,
                    hide_index=True
                )
            else:
                st.warning("⚠️ TCPL data not available")
        
        with viz_tab2:
            st.subheader("Top 10 Cities - November Forecast")
            
            city_summary = nov_data.groupby('city_norm').agg({
                'forecasted_qty': 'sum'
            }).reset_index().sort_values('forecasted_qty', ascending=False).head(10)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=city_summary['city_norm'],
                y=city_summary['forecasted_qty'],
                marker_color='darkblue'
            ))
            
            fig.update_layout(
                xaxis_title="City",
                yaxis_title="Forecasted Sales",
                height=400,
                title="Top 10 Cities - November Forecast"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed table
            st.markdown("---")
            st.markdown("### 📊 Detailed City Forecast")
            
            city_detail = nov_data.groupby('city_norm').agg({
                'forecasted_qty': 'sum'
            }).reset_index().sort_values('forecasted_qty', ascending=False)
            
            city_detail.columns = ['City', 'November Forecast']
            
            st.dataframe(
                city_detail.style.format({
                    'November Forecast': '{:,.0f}'
                }),
                use_container_width=True,
                height=400,
                hide_index=True
            )
        
        with viz_tab3:
            st.subheader("Top 10 SKUs - November Forecast")
            
            # Load SKU names
            try:
                sku_names = pd.read_csv('data/sku_names.csv')
            except:
                sku_names = pd.DataFrame({
                    'item_id': nov_data['item_id'].unique(),
                    'item_name': nov_data['item_id'].unique().astype(str)
                })
            
            # Aggregate by SKU
            sku_summary = nov_data.groupby('item_id').agg({
                'forecasted_qty': 'sum'
            }).reset_index()
            
            sku_summary = sku_summary.merge(sku_names, on='item_id', how='left')
            sku_summary['item_name'] = sku_summary['item_name'].fillna('Unknown SKU')
            
            sku_summary_top10 = sku_summary.sort_values('forecasted_qty', ascending=False).head(10)
            
            # Horizontal bar chart
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                y=sku_summary_top10['item_name'],
                x=sku_summary_top10['forecasted_qty'],
                orientation='h',
                marker_color='#006400',
                text=sku_summary_top10['forecasted_qty'].apply(lambda x: f'{x/1000:.0f}k' if x >= 1000 else f'{x:.0f}'),
                textposition='outside'
            ))
            
            fig.update_layout(
                xaxis_title="Forecasted Quantity",
                yaxis_title="SKU",
                height=500,
                title="Top 10 SKUs - November Forecast",
                yaxis={'categoryorder': 'total ascending'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed table
            st.markdown("---")
            st.markdown("### 📊 Detailed SKU Forecast")
            
            display_sku = sku_summary[['item_name', 'forecasted_qty']].copy()
            display_sku = display_sku.sort_values('forecasted_qty', ascending=False)
            display_sku.columns = ['SKU Name', 'November Forecast']
            
            st.dataframe(
                display_sku.style.format({
                    'November Forecast': '{:,.0f}'
                }),
                use_container_width=True,
                height=400,
                hide_index=True
            )

        with viz_tab4:
            st.subheader("📋 Detailed Forecast Data - November 2025")
            
            # Prepare data with TCPL mapping
            nov_detail = nov_data[['city_norm', 'item_id', 'forecasted_qty']].copy()
            nov_detail = nov_detail.merge(
                data['city_tcpl_mapping'][['city_norm', 'tcpl_plant_code', 'fe_city_name']],
                on='city_norm',
                how='left'
            )
            
            # Reorder columns
            nov_detail = nov_detail[['tcpl_plant_code', 'fe_city_name', 'city_norm', 'item_id', 'forecasted_qty']]
            nov_detail = nov_detail.sort_values(['tcpl_plant_code', 'city_norm', 'item_id'])
            
            # Filters
            st.markdown("#### 🔍 Filters")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                tcpl_opts = ['All'] + sorted(nov_detail['tcpl_plant_code'].dropna().unique().astype(int).tolist())
                sel_tcpl = st.selectbox("TCPL Plant", tcpl_opts, key='nov_detail_tcpl')
            
            with col2:
                city_opts = ['All'] + sorted(nov_detail['city_norm'].dropna().unique().tolist())
                sel_city = st.selectbox("City", city_opts, key='nov_detail_city')
            
            with col3:
                sku_opts = ['All'] + sorted(nov_detail['item_id'].dropna().unique().tolist())
                sel_sku = st.selectbox("SKU", sku_opts, key='nov_detail_sku')
            
            # Apply filters
            filtered = nov_detail.copy()
            if sel_tcpl != 'All':
                filtered = filtered[filtered['tcpl_plant_code'] == sel_tcpl]
            if sel_city != 'All':
                filtered = filtered[filtered['city_norm'] == sel_city]
            if sel_sku != 'All':
                filtered = filtered[filtered['item_id'] == sel_sku]
            
            st.markdown(f"**Showing {len(filtered):,} records**")
            
            # Display
            display_df = filtered.copy()
            display_df.columns = ['TCPL Code', 'City Name', 'City Norm', 'Item ID', 'Forecast']
            
            st.dataframe(display_df, use_container_width=True, height=400, hide_index=True)
            
            # Download
            csv = filtered.to_csv(index=False)
            st.download_button(
                label="📥 Download November Data (CSV)",
                data=csv,
                file_name="november_2025_forecast_detail.csv",
                mime="text/csv",
                key='download_nov_detail'
            )
    
    # ========================================================================
    # DECEMBER FORECAST TAB
    # ========================================================================
    
    # ========================================================================
    # DECEMBER 2025 TAB (M+3 FORECAST)
    # ========================================================================

    elif forecast_tab == "December 2025 (M+3)":
        
        # Load December forecast
        #dec_data = data['dec_forecast_m3'].copy()
        dec_data = data['dec_with_features'].copy()

        
        # Apply slider adjustments using MODEL
        if discount_change != 0 or marketing_change != 0:
            st.info(f"🎯 Applying adjustments: Discount {discount_change:+d}%, Marketing {marketing_change:+d}%")
            
            # Use model to re-predict with adjusted features
            dec_data = adjust_features_and_predict(
                base_data=dec_data,
                model=model_m3,
                scaler=scaler_m3,
                feature_cols=features_m3,
                discount_change=discount_change,
                marketing_change=marketing_change
            )
            
            if marketing_change != 0:
                st.info(f"""
                📊 **Marketing Impact Applied**  
                Historical elasticity: **0.2423**  
                Expected sales change: **{marketing_change * 0.2423 / 100 * 100:+.2f}%**
                """)
            
            st.success("✅ Forecast updated using M+3 model with adjusted features!")
        
        # Calculate totals
        total_forecast = dec_data['forecasted_qty'].sum()
        
        # Recalculate TCPL from adjusted city data
        try:
            # Add TCPL mapping
            city_tcpl = data['city_tcpl_mapping']
            dec_data_with_tcpl = dec_data.merge(
                city_tcpl[['city_norm', 'tcpl_plant_code', 'fe_city_name']],
                on='city_norm',
                how='left'
            )
            
            # Aggregate to TCPL × SKU
            dec_data_tcpl = dec_data_with_tcpl.groupby(['tcpl_plant_code', 'item_id']).agg({
                'forecasted_qty': 'sum'
            }).reset_index()
            
            # Handle NCR split (50-50 between 1214 and 1476)
            ncr_mask = dec_data_with_tcpl['fe_city_name'] == 'NCR'
            if ncr_mask.sum() > 0:
                ncr_tcpl_current = dec_data_with_tcpl[ncr_mask]['tcpl_plant_code'].iloc[0]
                ncr_rows = dec_data_tcpl[dec_data_tcpl['tcpl_plant_code'] == ncr_tcpl_current].copy()
                dec_data_tcpl_non_ncr = dec_data_tcpl[dec_data_tcpl['tcpl_plant_code'] != ncr_tcpl_current].copy()
                
                # Split NCR
                ncr_1214 = ncr_rows.copy()
                ncr_1214['tcpl_plant_code'] = 1214
                ncr_1214['forecasted_qty'] = ncr_1214['forecasted_qty'] * 0.5
                
                ncr_1476 = ncr_rows.copy()
                ncr_1476['tcpl_plant_code'] = 1476
                ncr_1476['forecasted_qty'] = ncr_1476['forecasted_qty'] * 0.5
                
                # Combine
                dec_data_tcpl = pd.concat([dec_data_tcpl_non_ncr, ncr_1214, ncr_1476], ignore_index=True)
                
                # Re-aggregate
                dec_data_tcpl = dec_data_tcpl.groupby(['tcpl_plant_code', 'item_id']).agg({
                    'forecasted_qty': 'sum'
                }).reset_index()
            
            tcpl_available = True
        except Exception as e:
            st.warning(f"⚠️ TCPL aggregation error: {e}")
            tcpl_available = False
        
        # Calculate totals
        total_forecast = dec_data['forecasted_qty'].sum()
        
        if tcpl_available:
            tcpl_total_forecast = dec_data_tcpl['forecasted_qty'].sum()
        
        # ========================================================================
        # DISPLAY METRICS
        # ========================================================================
        
        st.markdown("### 📅 December 2025 Forecast (M+3)")
        
        st.info("""
        **Forecast Information:**
        - Using data available till: **September 2025**
        - Forecast horizon: **3 months ahead (M+3)**
        - No actuals available (future month)
        - Expected accuracy based on October validation below
        """)
        
        # Show forecast totals
        st.markdown("### 📊 December Forecast")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "City × SKU Forecast",
                f"{total_forecast:,.0f}",
                help="December 2025 forecast at City × SKU level"
            )
        
        with col2:
            if tcpl_available:
                st.metric(
                    "TCPL × SKU Forecast",
                    f"{tcpl_total_forecast:,.0f}",
                    help="December 2025 forecast at TCPL × SKU level"
                )
        
        # ========================================================================
        # SHOW M+3 VALIDATION METRICS (from October)
        # ========================================================================
        
        st.markdown("---")
        st.markdown("### 🎯 Expected Accuracy (Based on October Validation)")
        
        st.info("The model was validated on October 2025 using July 2025 data. These metrics indicate expected accuracy for December forecast.")
        
        # Load October M+3 validation
        oct_val_m3 = data['oct_validation_m3']
        oct_val_tcpl_m3 = data['oct_validation_tcpl_m3']
        
        # Calculate October M+3 metrics
        wape_city_m3 = calculate_wape(oct_val_m3['actual_qty'], oct_val_m3['forecasted_qty'])
        wape_tcpl_m3 = calculate_wape(oct_val_tcpl_m3['actual_qty'], oct_val_tcpl_m3['forecasted_qty'])
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "City × SKU Expected Accuracy",
                f"{100 - wape_city_m3:.2f}%",
                help="Based on October validation"
            )
        
        with col2:
            st.metric(
                "City × SKU Expected WAPE",
                f"{wape_city_m3:.2f}%",
                delta=f"{wape_city_m3 - 18.91:+.2f}pp vs M+2",
                delta_color="inverse"
            )
        
        with col3:
            if tcpl_available:
                st.metric(
                    "TCPL × SKU Expected Accuracy",
                    f"{100 - wape_tcpl_m3:.2f}%",
                    help="Based on October validation"
                )
        
        with col4:
            if tcpl_available:
                st.metric(
                    "TCPL × SKU Expected WAPE",
                    f"{wape_tcpl_m3:.2f}%",
                    delta=f"{wape_tcpl_m3 - 15.19:+.2f}pp vs M+2",
                    delta_color="inverse"
                )
        
        # ========================================================================
        # CONFIDENCE INTERVALS (WAPE-Proportional)
        # ========================================================================
        
        st.markdown("---")
        st.markdown("### 📊 Forecast Confidence Intervals")
        
        st.info("Confidence intervals are proportional to forecast accuracy. TCPL intervals are tighter, reflecting aggregation benefits. M+3 intervals are wider than M+2 due to longer horizon.")
        
        # Calculate standard error from October M+3 validation
        oct_val_m3['error'] = oct_val_m3['forecasted_qty'] - oct_val_m3['actual_qty']
        std_error_base = np.sqrt((oct_val_m3['error']**2).sum())
        
        # TCPL scaled by WAPE ratio
        wape_ratio = wape_tcpl_m3 / wape_city_m3
        std_error_tcpl = std_error_base * wape_ratio
        
        # City × SKU CI
        st.markdown("**City × SKU Level**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            lower = total_forecast - 0.67 * std_error_base
            upper = total_forecast + 0.67 * std_error_base
            st.info(f"**50% CI:** {lower:,.0f} - {upper:,.0f}")
        
        with col2:
            lower = total_forecast - 1.15 * std_error_base
            upper = total_forecast + 1.15 * std_error_base
            st.info(f"**75% CI:** {lower:,.0f} - {upper:,.0f}")
        
        with col3:
            lower = total_forecast - 1.96 * std_error_base
            upper = total_forecast + 1.96 * std_error_base
            st.info(f"**95% CI:** {lower:,.0f} - {upper:,.0f}")
        
        # TCPL × SKU CI
        if tcpl_available:
            st.markdown("**TCPL × SKU Level**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                lower = tcpl_total_forecast - 0.67 * std_error_tcpl
                upper = tcpl_total_forecast + 0.67 * std_error_tcpl
                st.info(f"**50% CI:** {lower:,.0f} - {upper:,.0f}")
            
            with col2:
                lower = tcpl_total_forecast - 1.15 * std_error_tcpl
                upper = tcpl_total_forecast + 1.15 * std_error_tcpl
                st.info(f"**75% CI:** {lower:,.0f} - {upper:,.0f}")
            
            with col3:
                lower = tcpl_total_forecast - 1.96 * std_error_tcpl
                upper = tcpl_total_forecast + 1.96 * std_error_tcpl
                st.info(f"**95% CI:** {lower:,.0f} - {upper:,.0f}")
        
        # ========================================================================
        # VISUALIZATION TABS
        # ========================================================================
        
        st.markdown("---")
        viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs(["🏭 TCPL Forecast", "📍 City Forecast", "📦 SKU Forecast", "📋 Detailed Data"])
        
        with viz_tab1:
            st.subheader("TCPL Warehouse-Level Forecast")
            
            if tcpl_available:
                # Aggregate TCPL
                tcpl_summary = dec_data_tcpl.groupby('tcpl_plant_code').agg({
                    'forecasted_qty': 'sum'
                }).reset_index().sort_values('forecasted_qty', ascending=False)
                
                tcpl_summary['tcpl_plant_code'] = tcpl_summary['tcpl_plant_code'].astype(int)
                
                # Top 10 chart
                top_tcpl = tcpl_summary.head(10)
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=top_tcpl['tcpl_plant_code'].astype(str),
                    y=top_tcpl['forecasted_qty'],
                    name='December Forecast',
                    marker_color='darkred'
                ))
                
                fig.update_layout(
                    xaxis_title="TCPL Plant Code",
                    yaxis_title="Forecasted Sales",
                    height=400,
                    title="Top 10 TCPL Plants - December Forecast",
                    xaxis=dict(type='category', categoryorder='total descending')
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total TCPL Plants", tcpl_summary['tcpl_plant_code'].nunique())
                with col2:
                    st.metric("Total Forecast", f"{tcpl_total_forecast:,.0f}")
                
                # Detailed table
                st.markdown("---")
                st.markdown("### 📊 Detailed TCPL Forecast")
                
                display_tcpl = tcpl_summary[['tcpl_plant_code', 'forecasted_qty']].copy()
                display_tcpl.columns = ['TCPL Code', 'December Forecast']
                
                st.dataframe(
                    display_tcpl.style.format({
                        'TCPL Code': '{:,}',
                        'December Forecast': '{:,.0f}'
                    }),
                    use_container_width=True,
                    height=400,
                    hide_index=True
                )
            else:
                st.warning("⚠️ TCPL data not available")
        
        with viz_tab2:
            st.subheader("Top 10 Cities - December Forecast")
            
            city_summary = dec_data.groupby('city_norm').agg({
                'forecasted_qty': 'sum'
            }).reset_index().sort_values('forecasted_qty', ascending=False).head(10)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=city_summary['city_norm'],
                y=city_summary['forecasted_qty'],
                marker_color='darkblue'
            ))
            
            fig.update_layout(
                xaxis_title="City",
                yaxis_title="Forecasted Sales",
                height=400,
                title="Top 10 Cities - December Forecast"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed table
            st.markdown("---")
            st.markdown("### 📊 Detailed City Forecast")
            
            city_detail = dec_data.groupby('city_norm').agg({
                'forecasted_qty': 'sum'
            }).reset_index().sort_values('forecasted_qty', ascending=False)
            
            city_detail.columns = ['City', 'December Forecast']
            
            st.dataframe(
                city_detail.style.format({
                    'December Forecast': '{:,.0f}'
                }),
                use_container_width=True,
                height=400,
                hide_index=True
            )
        
        with viz_tab3:
            st.subheader("Top 10 SKUs - December Forecast")
            
            # Aggregate by SKU
            sku_summary = dec_data.groupby('item_id').agg({
                'forecasted_qty': 'sum'
            }).reset_index()
            
            sku_summary_top10 = sku_summary.sort_values('forecasted_qty', ascending=False).head(10)
            
            # Horizontal bar chart
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                y=sku_summary_top10['item_id'].astype(str),
                x=sku_summary_top10['forecasted_qty'],
                orientation='h',
                marker_color='#8B0000',
                text=sku_summary_top10['forecasted_qty'].apply(lambda x: f'{x/1000:.0f}k' if x >= 1000 else f'{x:.0f}'),
                textposition='outside'
            ))
            
            fig.update_layout(
                xaxis_title="Forecasted Quantity",
                yaxis_title="SKU",
                height=500,
                title="Top 10 SKUs - December Forecast",
                yaxis={'categoryorder': 'total ascending'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed table
            st.markdown("---")
            st.markdown("### 📊 Detailed SKU Forecast")
            
            display_sku = sku_summary[['item_id', 'forecasted_qty']].copy()
            display_sku = display_sku.sort_values('forecasted_qty', ascending=False)
            display_sku.columns = ['SKU ID', 'December Forecast']
            
            st.dataframe(
                display_sku.style.format({
                    'December Forecast': '{:,.0f}'
                }),
                use_container_width=True,
                height=400,
                hide_index=True
            )

        with viz_tab4:
            st.subheader("📋 Detailed Forecast Data - December 2025")
            
            # Prepare data with TCPL mapping
            dec_detail = dec_data[['city_norm', 'item_id', 'forecasted_qty']].copy()
            dec_detail = dec_detail.merge(
                data['city_tcpl_mapping'][['city_norm', 'tcpl_plant_code', 'fe_city_name']],
                on='city_norm',
                how='left'
            )
            
            # Reorder columns
            dec_detail = dec_detail[['tcpl_plant_code', 'fe_city_name', 'city_norm', 'item_id', 'forecasted_qty']]
            dec_detail = dec_detail.sort_values(['tcpl_plant_code', 'city_norm', 'item_id'])
            
            # Filters
            st.markdown("#### 🔍 Filters")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                tcpl_opts = ['All'] + sorted(dec_detail['tcpl_plant_code'].dropna().unique().astype(int).tolist())
                sel_tcpl = st.selectbox("TCPL Plant", tcpl_opts, key='dec_detail_tcpl')
            
            with col2:
                city_opts = ['All'] + sorted(dec_detail['city_norm'].dropna().unique().tolist())
                sel_city = st.selectbox("City", city_opts, key='dec_detail_city')
            
            with col3:
                sku_opts = ['All'] + sorted(dec_detail['item_id'].dropna().unique().tolist())
                sel_sku = st.selectbox("SKU", sku_opts, key='dec_detail_sku')
            
            # Apply filters
            filtered = dec_detail.copy()
            if sel_tcpl != 'All':
                filtered = filtered[filtered['tcpl_plant_code'] == sel_tcpl]
            if sel_city != 'All':
                filtered = filtered[filtered['city_norm'] == sel_city]
            if sel_sku != 'All':
                filtered = filtered[filtered['item_id'] == sel_sku]
            
            st.markdown(f"**Showing {len(filtered):,} records**")
            
            # Display
            display_df = filtered.copy()
            display_df.columns = ['TCPL Code', 'City Name', 'City Norm', 'Item ID', 'Forecast']
            
            st.dataframe(display_df, use_container_width=True, height=400, hide_index=True)
            
            # Download
            csv = filtered.to_csv(index=False)
            st.download_button(
                label="📥 Download December Data (CSV)",
                data=csv,
                file_name="december_2025_forecast_detail.csv",
                mime="text/csv",
                key='download_dec_detail'
            )

# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    main()