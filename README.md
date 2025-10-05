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

## Instruction
1. Download TM110_Training.py and TM110_Test.py and put into one folder
2. Open a terminal and move to the folder.
3. Run TM110_Training.py. Two pkl files and two csv files should be created in the folder.
4. Run the following command to test the model: ```TM110_Test.py --input Input --model Model --output Output```.
    * Input: The CTG data to be inputted. Required. Accepted file type: .xlsx, .xls, .csv
    * Model: The machine learning model to be used. Optional. Options: HGB_NSP.pkl or HGB_CLASS.pkl
    * Output: Where to save the predictions result. Optional. Default: predictions.csv

## Resources
All models were trained on the open-source CTG dataset provided by the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/193/cardiotocography)
