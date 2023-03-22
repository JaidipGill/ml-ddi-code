# ML Models for DDI Prediction

Builds and tests machine learning models for predicting drug-drug interactions. Uses features such as chemical descriptors, 
CYP activity profiles, fraction metabolised data, and population parameters to predict AUC ratio. Outputs plot of predicted AUC ratios vs observed values, and predicted classes of each interaction. Feature importance for
the model can also be extracted. These models were used for the paper ['Evaluating the performance of machine-learning regression models for pharmacokinetic drug–drug interactions'](https://ascpt.onlinelibrary.wiley.com/doi/full/10.1002/psp4.12884?campaign=wolearlyview)

## File descriptions:
* 'ML Models' - Machine learning models (regressor with classifier on top), feature importance extraction.
  * Change function arguments to run specific models
* 'SHAP importances' - Code to generate SHAP feature importance values and confusion matrices of model results
* Dats files are stored in the Data Files folder
  * Feature matrix stored in 'Timeseries features.csv'
  * Labels (AUC Ratios) are extracted within the code from the 'Input Data.csv' file


