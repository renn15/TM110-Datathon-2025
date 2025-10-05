# Fetal Cardiotocography Classification using Histogram-based Gradient Boosting
**(TM-110)**\
**Team Leader:** Steven Darren Wijaya\
**Members:** Bryan Winsley Louwren, Novadella Zen\
Submission for 2025 Lifeline Datathon organized by MLDA@EEE

## Project Description
A python-based machine learning project designed to provide reliable and interpretable classification of fetal health states from cardiotocography (CTG) data for clinical use.\
Applies Histogram-based Gradient Boosting (HGB) to classify fetal cardiotocography (CTG) data into three categories:
1. Normal – Healthy fetal heart rate patterns
2. Suspect – Borderline or unclear patterns
3. Pathologic – Abnormal patterns indicating potential fetal distress

After comparing 3 models, HGB was chosen because it gave the most accurate results, trained faster than the others, and made fewer false negatives when detecting Pathologic cases.

  ### Our program follows the workflow:
  * Data visualization and relationship mapping
  * Seperating data into training and test subsets
  * Measuring performance with confusion matrix, balanced accuracy, and macro-F1

## Resources
All models were trained on the open-source CTG dataset provided by the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/193/cardiotocography)
