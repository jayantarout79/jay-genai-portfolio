# 🌲 Notes on Random Forest Implementation

- **Features driving buyers:**
  - `PageValues` → strongest predictor (value of pages visited).
  - `ProductRelated_Duration`, `ExitRates`, `SpecialDay`.

- **Imbalance:**
  - Buyers ~15%, Non-buyers ~85%.
  - Balanced via stratified split and `class_weight` tuning.

- **Metrics compared:**
  - High precision (~0.92) with depth control.
  - Recall was lower (~0.15 at 0.5 threshold) → trade-off accepted to save cost.

- **Takeaway:**
  Random Forest is less prone to overfitting vs single tree. Helps capture non-linear interactions and stabilizes predictions.