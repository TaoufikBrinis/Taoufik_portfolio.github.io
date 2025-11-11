# Amazon Electronics Sales Analysis & ML Prediction

## 🎯 Project Overview

This project analyzes Amazon's electronics marketplace to identify key drivers of sales performance and build predictive models for monthly sales forecasting. The analysis combines exploratory data analysis (EDA) with machine learning to deliver actionable pricing and marketing recommendations.

## 💼 Business Problem

Amazon needs to understand:
- What pricing strategies maximize sales without heavy discounting?
- Do marketing features (coupons, "Best Seller" tags, sponsored ads) actually drive sales?
- Which product categories offer the best opportunities for growth?
- Can we predict monthly sales based on product attributes?

## 🔍 Key Findings

### 1. Moderate Discounts Win
- **10-20% discounts drive 5× higher sales** than deep discounts (>30%)
- Deep discounts often signal slow-moving inventory
- **Recommendation:** Focus on moderate, attractive discounts rather than extreme markdowns

### 2. Social Proof is Powerful
- Products with **"Best Seller" tags sell 500% more** than regular products
- Customer trust in popular products drives conversion
- **Recommendation:** Strategic product positioning matters more than aggressive discounting

### 3. Coupons Alone Don't Guarantee Success
- Products **without coupons actually sell more** on average
- Coupons may be used for low-demand or new products
- **Recommendation:** Combine coupons with strong product reputation and ratings

### 4. Category Performance Insights
- **Power & Batteries** dominate total sales (low-cost essentials)
- **Wearables and Speakers** have highest customer satisfaction but lower visibility
- **Opportunity:** Increase visibility or bundle high-rated products

## 🤖 Predictive Model

**Model Type:** Random Forest Regressor  
**Performance:** R² = 0.90 (explains 90% of variance in monthly sales)

**Most Important Features:**
1. Product ratings
2. Price point
3. Discount percentage
4. "Best Seller" tag
5. Product category

## 🛠️ Technologies Used

- **Python 3.x**
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning (Random Forest)
- **Matplotlib & Seaborn** - Data visualization
- **Jupyter Notebook** - Interactive development

## 📊 Dataset

- **Source:** Amazon Electronics marketplace data
- **Size:** [Add dataset size]
- **Features:** Price, discount percentage, ratings, reviews, monthly sales, product categories, marketing tags

## 🎓 Key Techniques

- Exploratory Data Analysis (EDA)
- Statistical Analysis
- Feature Engineering
- Random Forest Regression
- Model Evaluation (R², MSE)
- Data Visualization

## 💡 Business Impact

This analysis enables:
- **Data-driven pricing strategies** focusing on moderate discounts
- **Strategic product positioning** leveraging social proof
- **Marketing budget optimization** by understanding feature effectiveness
- **Sales forecasting** for inventory and supply chain planning

## 📈 Next Steps

1. A/B test moderate discount strategies vs. current approach
2. Implement dynamic "Best Seller" highlighting based on sales velocity
3. Analyze customer review sentiment for product improvement insights
4. Build real-time sales prediction dashboard for business stakeholders

---

## 💻 [View Full Analysis & Code](https://nbviewer.org/github/TaoufikBrinis/Taoufik_portfolio.github.io/blob/main/AMAZON%20ELECTRONICS%20SALES%20ANALYSIS%20PROJECT.ipynb)

## 🏠 [Back to Portfolio](../README.md)
