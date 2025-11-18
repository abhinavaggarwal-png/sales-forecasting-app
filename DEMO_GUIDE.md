# Quick Start Guide - For VC Presentation

## 🚀 5-Minute Setup

### Step 1: Place Your Data (1 min)
```bash
# Copy your data file to the data folder
cp /path/to/your/all_data2.csv sales_forecasting_app/data/
```

### Step 2: Install & Run (2 min)
```bash
cd sales_forecasting_app
pip install -r requirements.txt
streamlit run app.py
```

### Step 3: Demo the App (2 min)

**In the Sidebar:**
1. Click "Load Data" ✅
2. Wait for success message

**Main Dashboard:**
1. Move the "Discount % Change" slider → Show +20% discount
2. Move the "Budget % Change" slider → Show +50% budget
3. Click "Generate November Forecast" 🚀

**Show Results:**
1. **Summary Metrics** - Point to the increase in forecast
2. **City Analysis Tab** - Show top 10 cities chart
3. **SKU Analysis Tab** - Show top 10 SKUs chart
4. **Key Drivers Tab** - Show feature importance

---

## 📊 Demo Talking Points

### Opening (30 sec)
> "We've built a sophisticated sales forecasting engine that predicts SKU-city level sales and enables what-if scenario planning."

### The Problem (30 sec)
> "Traditional forecasting lacks:
> - Real-time scenario testing
> - Understanding of key drivers
> - Confidence in predictions"

### Our Solution (60 sec)
> "Our ML-powered dashboard provides:
> 
> 1. **Predictive Accuracy**: XGBoost model with [XX]% WAPE
> 2. **Actionable Insights**: See exactly how discount and marketing affect sales
> 3. **Confidence Intervals**: Know the range of likely outcomes
> 4. **Granular Control**: City x SKU level precision"

### Live Demo (90 sec)

**Scenario 1: Aggressive Promotion**
- Set Discount: +30%
- Set Budget: +50%
- Generate forecast
- "See? We expect a [X]% increase in total sales, with Mumbai and Delhi leading the growth"

**Scenario 2: Cost Optimization**
- Set Discount: -10%
- Set Budget: -20%
- Generate forecast
- "Even with reduced spend, we maintain [X]% of baseline sales, improving margins"

### Key Features (30 sec)
> Show each tab briefly:
> - City breakdown: "Understand regional performance"
> - SKU breakdown: "Identify star products"
> - Drivers: "Know what really moves the needle"
> - Export: "Take detailed data for further analysis"

### Technical Edge (30 sec)
> "Built on production-grade ML:
> - 127 features across pricing, availability, marketing
> - Trained on [X] months of data
> - Handles 101 SKUs × 221 cities = 22K+ forecasts
> - Confidence intervals quantify uncertainty"

### Closing (30 sec)
> "This transforms sales planning from guesswork to data-driven strategy. 
> Questions?"

---

## 🎯 Demo Tips

### DO:
✅ Prepare 2-3 scenarios in advance
✅ Know your baseline numbers cold
✅ Have a backup if something breaks
✅ Show the confidence intervals feature
✅ Emphasize the business value, not just tech

### DON'T:
❌ Get stuck in preprocessing details
❌ Apologize for UI (it's a prototype)
❌ Click through too fast
❌ Over-explain the ML model
❌ Forget to show download feature

---

## 🔧 Common Demo Scenarios

### Scenario A: "Festive Season Push"
- Discount: +25%
- Budget: +75%
- Message: "Maximize volume during high-demand period"

### Scenario B: "Margin Optimization"
- Discount: -15%
- Budget: +10% (targeted campaigns)
- Message: "Improve profitability while maintaining reach"

### Scenario C: "New Market Entry"
- Focus on specific city in Advanced mode
- Higher budget allocation to that city
- Message: "Simulate market entry strategy"

---

## 📋 Pre-Demo Checklist

- [ ] Data file loaded and tested
- [ ] App runs without errors
- [ ] Know your baseline WAPE
- [ ] Screenshots ready as backup
- [ ] Laptop fully charged
- [ ] Good internet connection (if live demo)
- [ ] Have Excel/CSV ready if download demo fails
- [ ] Know the top 3 SKUs and cities by name
- [ ] Understand what features drive predictions most
- [ ] Can explain confidence intervals simply

---

## 🎤 Handling Q&A

**"How accurate is the model?"**
> "Our WAPE is [XX]%, and we provide confidence intervals for every prediction. We're continuously improving with more data."

**"Can it handle [specific scenario]?"**
> "The current version handles discount and budget changes. We're building advanced controls for [their scenario] in the next version."

**"What data do you need?"**
> "Historical sales, pricing, promotional spend, and availability data. We can integrate with most retail systems."

**"How long to deploy?"**
> "After data integration, 2-4 weeks for initial deployment, then continuous improvement."

**"What's the ROI?"**
> "Clients typically see [X]% improvement in forecast accuracy, leading to [Y]% reduction in stockouts and [Z]% better margin management."

---

## 🚨 Emergency Backup

If the app crashes during demo:

1. **Have screenshots** of each tab ready
2. **Show the code** in utils.py to demonstrate technical depth
3. **Walk through** the feature engineering logic
4. **Pivot to** the business value proposition

---

**Good luck with your presentation! 🍀**
