# Fetal Cardiotocography Classification using Histogram-based Gradient Boosting
**TM-110:** Steven Darren Wijaya, Bryan Winsley Louwren, Novadella Zen
## Project Description
A python-based machine learning project designed to provide reliable and interpretable classification of fetal health states from cardiotocography (CTG) data for clinical use.\
Applies Histogram-based Gradient Boosting (HGB) to classify fetal cardiotocography (CTG) data into three categories:
1. Normal – Healthy fetal heart rate patterns
2. Suspect – Borderline or unclear patterns
3. Pathologic – Abnormal patterns indicating potential fetal distress

HGB was chosen because it efficiently handles large tabular datasets, captures nonlinear feature interactions, provides feature importance insights for clinical interpretability, and supports class weighting to reduce false negatives.

  ### The solution follows the workflow:
  * Data standardization and light cleaning
  * Handling class imbalance with class weighting
  * Measuring performance with balanced accuracy and macro-F1

## Resources
All models were trained on the open-source CTG dataset provided by the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/193/cardiotocography)
