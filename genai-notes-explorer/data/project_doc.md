# 🛒 Retail Purchase Prediction (Random Forest)

## Problem
E-commerce managers want to know which customers are most likely to purchase during a session. Discounts are costly if given to everyone. Predicting “buyer vs non-buyer” helps optimize promotions and save revenue.

## Approach
- Dataset: Online Shoppers Intention (12k+ sessions, 18 features).
- Preprocessing: one-hot encoding, scaling, categorical handling.
- Modeling: Decision Tree + Random Forest.
- Handling imbalance with stratified sampling.
- Validation: 70/10/20 split with metrics tracked.

## Metrics
- **Precision** (focus): to avoid wrongly predicting buyers and wasting discounts.
- Recall, F1, ROC-AUC tracked for balance.
- Business framing: True Positive = targeted offer likely converts to sale; False Positive = wasted offer.