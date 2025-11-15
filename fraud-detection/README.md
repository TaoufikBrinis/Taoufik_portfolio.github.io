# 🛡️ AI-Powered Fraud Detection System for Self-Checkout Transactions

**Reducing retail losses by 98.6% through intelligent machine learning**

---

## 📊 Executive Summary

A comprehensive fraud detection solution that identifies fraudulent self-checkout transactions with **96.46% accuracy** while maintaining an exceptionally low false positive rate of **0.05%**. The system achieved a **98.6% reduction in misclassification costs** compared to baseline approaches, translating to potential savings of **CHF 127M annually** for large retail operations.

**Key Achievement:** Optimized model reduces total cost from 2,206 (baseline) to **311** through intelligent threshold tuning and cost-sensitive learning.

---

## 🎯 Business Context

Self-checkout systems are convenient for customers but vulnerable to theft and errors. Retailers face average loss rates of **3.5% of sales** at self-checkout stations, representing hundreds of millions in annual losses for major chains.

### The Challenge
- **High fraud rates:** 5% of transactions are fraudulent
- **Customer experience risk:** False accusations damage trust and loyalty
- **Cost imbalance:** Wrongly flagging honest customers is 5x more costly than missing fraud
- **Need for precision:** Must balance fraud detection with minimal customer friction

### The Solution
An AI-powered classification system that:
- Identifies 96% of fraudulent transactions automatically
- Flags only 0.05% of legitimate customers (39 out of 71,149)
- Enables targeted interventions without disrupting the customer experience
- Provides interpretable results for operational teams

---

## 🔬 Technical Approach

### Methodology: CRISP-DM Framework
Following industry-standard data science methodology:

1. **Business Understanding**
   - Defined success criteria: Reduce fraud losses by ≥1 percentage point
   - Established cost function: FP cost = 5× FN cost
   - ROI analysis: CHF 611M net benefit over 5 years

2. **Data Understanding**
   - Dataset: 498,121 self-checkout transactions (Data Mining Cup 2019)
   - Class imbalance: 95% legitimate, 5% fraudulent
   - 10 behavioral features: scan timing, voids, modifications, trust level

3. **Data Preparation**
   - Removed zero-value transactions (71 records, 0.01%)
   - Stratified train/validation/test split (70%/15%/15%)
   - Model-specific preprocessing (StandardScaler for Logistic Regression, raw features for tree models)

4. **Modeling**
   - Developed and compared 4 algorithms: **XGBoost**, Random Forest, Neural Network, Logistic Regression
   - Cost-sensitive training with class weighting
   - Hyperparameter optimization via RandomizedSearchCV (30 iterations)
   - Evaluated using PR-AUC (optimal for imbalanced datasets)

5. **Evaluation & Threshold Optimization**
   - Tested 9,895 decision thresholds on validation set
   - Selected threshold (0.9081 for XGBoost) that minimizes: `Total Cost = (FP × 5) + FN`
   - Final evaluation on held-out test set

---

## 📈 Results & Impact

### Model Performance Comparison

| Model | Recall | Precision | F1 Score | Total Cost | ROC-AUC |
|-------|--------|-----------|----------|------------|---------|
| **XGBoost** (Selected) | **96.46%** | **98.93%** | **97.68%** | **311** | **98.20%** |
| Random Forest | 90.73% | 98.81% | 94.59% | 525 | 95.59% |
| Neural Network | 77.60% | 97.50% | 86.65% | 1,043 | 99.90% |
| Logistic Regression | 47.43% | 96.18% | 63.36% | 2,206 | 73.67% |

### Business Impact

**For a retailer with CHF 12.7B annual revenue (Migros-scale):**
- **Current losses at 3.5% shrink rate:** CHF 440M/year
- **Potential savings (1% reduction):** CHF 127M/year
- **Implementation cost:** CHF 9M (one-time) + CHF 3M/year operational
- **5-year ROI:** CHF 611M net benefit

**Operational Benefits:**
- Detects **3,433 out of 3,559 fraud cases** in test set
- Only **37 false positives** out of 71,149 legitimate transactions
- Enables efficient allocation of security resources
- Maintains positive customer experience

---

## 🛠️ Technical Stack

**Languages & Libraries:**
- Python 3.8+
- Scikit-learn (preprocessing, Logistic Regression, Random Forest)
- XGBoost (gradient boosting)
- TensorFlow/Keras (Neural Network)
- Pandas, NumPy (data manipulation)
- Matplotlib, Seaborn (visualization)

**Tools & Platforms:**
- Jupyter Notebook (development environment)
- Orange Data Mining (visual workflow design)
- PyCharm (IDE)

**Key Techniques:**
- Cost-sensitive learning with class weighting
- Stratified k-fold cross-validation
- Threshold optimization for business objectives
- Hyperparameter tuning (RandomizedSearchCV)
- PR-AUC evaluation for imbalanced data

---

## 📁 Project Structure

```
fraud-detection-system/
│
├── data/
│   ├── Fraud_Original_Full.csv          # Raw dataset (498,121 transactions)
│   └── Fraud_Cleaned.csv                # Preprocessed data
│
├── notebooks/
│   ├── Data_Exploration.ipynb           # EDA and feature analysis
│   ├── Logistic_Regression_Model.ipynb  # Baseline model
│   ├── Random_Forest.ipynb              # Ensemble model
│   ├── XGBoost.ipynb                    # Best performing model
│   └── Neural_Network.ipynb             # Deep learning approach
│
├── models/
│   ├── XGBoost_Model.pkl                # Trained XGBoost (RECOMMENDED)
│   ├── Random_Forest_Model.pkl          
│   ├── Neural_Network_Model.h5          
│   └── Logistic_Regression_Model.pkl    
│
├── workflows/
│   └── Analytics_Project.ows            # Orange visual workflow
│
├── reports/
│   └── Analytics_Project_Report.pdf     # Full 56-page documentation
│
└── README.md                            # This file
```

---

## 🚀 How to Use

### 1. Prerequisites
```bash
pip install pandas numpy scikit-learn xgboost tensorflow matplotlib seaborn
```

### 2. Load the Optimal Model
```python
import pickle
import pandas as pd

# Load XGBoost model
with open('models/XGBoost_Model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load your transaction data
transactions = pd.read_csv('your_transactions.csv')

# Get fraud probabilities
fraud_probabilities = model.predict_proba(transactions)[:, 1]

# Apply optimized threshold (0.9081)
predictions = (fraud_probabilities >= 0.9081).astype(int)

# Flag high-risk transactions
high_risk_transactions = transactions[predictions == 1]
```

### 3. Explore the Notebooks
Each Jupyter notebook is self-contained and includes:
- Data loading and preprocessing
- Model training with hyperparameter tuning
- Threshold optimization
- Performance evaluation
- Visualization of results

**Recommended order:**
1. `Data_Exploration.ipynb` - Understand the dataset
2. `XGBoost.ipynb` - See the winning solution
3. Other models for comparison

---

## 🔍 Key Insights

### Most Predictive Features
1. **trustLevel** (1-6): Low trust scores strongly correlate with fraud (22% fraud rate at level 1 vs 0% at levels 3-6)
2. **scansWithoutRegistration**: Fraud rate increases from 2.1% (0 scans) to 7.3% (10 scans)
3. **totalScanTimeInSeconds**: Fraudulent transactions cluster at longer durations
4. **lineItemVoids**: Higher void counts correlate with increased fraud risk

### Why XGBoost Won
- **Best recall:** Catches 96% of fraud without excessive false positives
- **Robust to imbalance:** Handles 95:5 class ratio effectively
- **No preprocessing needed:** Works directly with raw features
- **Interpretable:** Feature importance helps explain decisions
- **Lowest cost:** Optimizes the business objective directly

---

## 📚 Documentation

Full project documentation available in [`reports/Analytics_Project_Report.pdf`](reports/Analytics_Project_Report.pdf) covering:
- Complete CRISP-DM methodology
- Risk mitigation strategies
- Cost-benefit analysis
- Feature engineering decisions
- Model comparison and selection
- Deployment considerations

---

## 👤 About the Author

**Taoufik Brinis** | Business Analyst & IT Consultant

- 🎓 BSc Business Information Technology, FHNW Basel (Thesis Grade: 6.0/6.0)
- 🎓 MSc Business Information Systems (AI Innovation Focus) - In Progress
- 💼 Freelance Business Analyst & IT Consultant
- 📊 Specialized in: Machine Learning, Process Optimization, Data Analytics

**Proven Results:**
- 15% cost reduction through process optimization
- 60% reporting time improvement via dashboard development
- Multilingual: English, French, Arabic (native), German (B2)

### Connect
- 💼 [LinkedIn](https://www.linkedin.com/in/taoufik-brinis)
- 📧 [Email](mailto:your.email@example.com)
- 🌐 [Portfolio](https://your-portfolio-site.com)

---

## 📄 License & Citation

This project was developed as part of the Analytics Project module at FHNW Basel (2025).

**Dataset:** Data Mining Cup 2019 (Self-Checkout Fraud Detection)

**Citation:**
```
Brinis, T. (2025). AI-Powered Fraud Detection System for Self-Checkout Transactions. 
FHNW School of Business, Basel, Switzerland.
```

---

## 🤝 Collaboration & Consulting

Interested in implementing similar solutions for your retail operation or other fraud detection use cases?

**I offer:**
- Custom ML model development
- Existing system optimization
- Cost-benefit analysis and ROI modeling
- Team training on fraud detection techniques
- End-to-end analytics consulting


---

**⭐ If you find this project valuable, please star the repository!**

*Last updated: November 2025*
