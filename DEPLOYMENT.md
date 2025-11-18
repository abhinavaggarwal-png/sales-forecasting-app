# Deployment Guide - Sales Forecasting Dashboard

## 📦 Deployment Options

### Option 1: Local Deployment (Recommended for Demo)

**Pros:**
- Fast setup
- Full control
- No cloud costs
- Easy debugging

**Steps:**
```bash
# 1. Copy all files to your machine
cd sales_forecasting_app

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place your data
cp /path/to/all_data2.csv data/

# 4. Run
streamlit run app.py
```

Access at: `http://localhost:8501`

---

### Option 2: Streamlit Cloud (Recommended for Sharing)

**Pros:**
- Free hosting
- Easy sharing via URL
- Auto-updates from Git
- No server maintenance

**Steps:**

1. **Push to GitHub**
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/yourusername/sales-forecasting.git
git push -u origin main
```

2. **Deploy on Streamlit Cloud**
- Go to https://share.streamlit.io
- Sign in with GitHub
- Click "New app"
- Select your repository
- Set main file: `app.py`
- Click "Deploy"

3. **Configure Secrets** (if needed)
- In Streamlit Cloud dashboard
- Add any API keys or passwords in secrets.toml format

**Note:** You may need to upload data files separately due to GitHub size limits.

---

### Option 3: Docker Deployment

**Pros:**
- Consistent environment
- Easy to scale
- Cloud-agnostic

**Dockerfile:**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**Build & Run:**
```bash
docker build -t sales-forecast-app .
docker run -p 8501:8501 -v $(pwd)/data:/app/data sales-forecast-app
```

---

### Option 4: Cloud VM (AWS/GCP/Azure)

**Pros:**
- Full control
- Scalable
- Professional setup

**AWS EC2 Example:**

1. **Launch Instance**
- Instance type: t3.medium (2 vCPU, 4 GB RAM)
- OS: Ubuntu 22.04 LTS
- Storage: 20 GB
- Security group: Allow port 8501

2. **Setup on Instance**
```bash
# SSH into instance
ssh -i your-key.pem ubuntu@your-instance-ip

# Install dependencies
sudo apt update
sudo apt install python3-pip -y

# Clone your repo or upload files
git clone your-repo-url
cd sales_forecasting_app

# Install Python packages
pip3 install -r requirements.txt

# Run with nohup (keeps running after logout)
nohup streamlit run app.py --server.port=8501 --server.address=0.0.0.0 &
```

3. **Access**
- Open browser: `http://your-instance-ip:8501`

4. **Optional: Setup Domain & SSL**
- Point domain to instance IP
- Use Nginx as reverse proxy
- Setup SSL with Let's Encrypt

---

## 🔒 Security Considerations

### For Production Deployment:

1. **Authentication**
```python
# Add to app.py
import streamlit_authenticator as stauth

# Simple password protection
def check_password():
    def password_entered():
        if st.session_state["password"] == "your_secure_password":
            st.session_state["password_correct"] = True
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Password", type="password", 
                     on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Password", type="password", 
                     on_change=password_entered, key="password")
        st.error("Password incorrect")
        return False
    else:
        return True

if not check_password():
    st.stop()
```

2. **Data Security**
- Don't commit sensitive data to Git
- Use `.gitignore` for data files
- Encrypt data at rest if needed
- Use environment variables for secrets

3. **Access Control**
- Restrict IP ranges if possible
- Use VPN for internal access
- Implement user roles if needed

---

## 🔧 Configuration for Different Environments

### Development
```python
# config.py
DEBUG = True
DATA_PATH = "data/all_data2.csv"
```

### Staging
```python
# config.py
DEBUG = True
DATA_PATH = "/staging/data/all_data2.csv"
```

### Production
```python
# config.py
DEBUG = False
DATA_PATH = "/prod/data/all_data2.csv"
ENABLE_DOWNLOAD = False  # Restrict downloads
```

---

## 📊 Monitoring & Maintenance

### Logging
Add logging to track usage:
```python
import logging

logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Log user actions
logging.info(f"Forecast generated: Discount={discount_change}, Budget={budget_change}")
```

### Performance Monitoring
- Track page load times
- Monitor prediction latency
- Alert on errors

### Regular Updates
- Update data files monthly
- Retrain model quarterly
- Review and improve features
- Update dependencies for security

---

## 🚀 Scaling Considerations

### If Usage Grows:

1. **Cache Predictions**
```python
@st.cache_data(ttl=3600)
def get_predictions(discount, budget):
    # Cache results for 1 hour
    return predictions
```

2. **Optimize Data Loading**
```python
# Load only necessary columns
df = pd.read_csv('data.csv', usecols=required_cols)

# Use parquet format for faster loading
df = pd.read_parquet('data.parquet')
```

3. **Database Backend**
- Move from CSV to PostgreSQL/MySQL
- Use SQLAlchemy for queries
- Index frequently accessed columns

4. **API Backend**
- Separate prediction API from UI
- Use FastAPI or Flask
- Cache predictions in Redis

5. **Load Balancing**
- Deploy multiple instances
- Use nginx for load balancing
- Implement session stickiness

---

## 🐛 Troubleshooting

### Common Issues:

**Memory Errors**
- Reduce data size
- Use data chunking
- Increase instance memory

**Slow Loading**
- Implement caching
- Optimize data formats
- Use smaller data samples for testing

**Model Not Loading**
- Check pickle compatibility
- Verify XGBoost version
- Ensure model files aren't corrupted

**UI Not Responsive**
- Check network latency
- Optimize chart rendering
- Reduce data in visualizations

---

## 📋 Pre-Deployment Checklist

- [ ] All dependencies in requirements.txt
- [ ] Data files copied to correct location
- [ ] Model files verified and loading correctly
- [ ] Environment variables configured
- [ ] Security measures implemented
- [ ] Error handling tested
- [ ] Performance benchmarked
- [ ] Backup strategy in place
- [ ] Monitoring setup
- [ ] Documentation updated
- [ ] Demo scenarios tested
- [ ] Rollback plan prepared

---

## 🔄 CI/CD Pipeline (Optional)

### GitHub Actions Example:

```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    
    - name: Install dependencies
      run: pip install -r requirements.txt
    
    - name: Run tests
      run: pytest tests/
    
    - name: Deploy to server
      run: |
        ssh user@server 'cd /app && git pull && systemctl restart streamlit'
```

---

## 📞 Support Resources

### Getting Help:
- Streamlit Docs: https://docs.streamlit.io
- XGBoost Docs: https://xgboost.readthedocs.io
- Community Forum: https://discuss.streamlit.io

### Useful Commands:
```bash
# Check Streamlit version
streamlit --version

# Clear Streamlit cache
streamlit cache clear

# Run with custom config
streamlit run app.py --server.maxUploadSize=200

# Run on custom port
streamlit run app.py --server.port=8502
```

---

**Ready to deploy? Follow the option that best fits your needs!** 🚀
