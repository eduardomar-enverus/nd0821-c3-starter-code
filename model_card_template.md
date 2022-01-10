# Model Card

## Model Details
In this project, we developed a classification model on publicly available Census Bureau data to predict whether 
someone's income is above 50k. We evaluated the model performance on a general basis and on various slices of the data. 
Then, the model was deployed using the FastAPI package. Both the slice-validation and the API tests were incorporated 
into a CI/CD framework using GitHub Actions.

## Intended Use
Intended to be used on the publicly available Census Bureau data only

## Evaluation Data
Evaluated on general dataset and slices of categorical features used to create model

## Metrics
###General metrics
Precision = 0.71  
Recall = 0.24  
FBeta = 0.36  

###Example of slice metrics
Slice metrics - Feature: marital-status - Value: Divorced  
Category percent 13.39  
Precision = 0.56  
Recall = 0.30  
FBeta = 0.39  

## Ethical Considerations
We risk attributing someone's income solely to the features capture in the original dataset even though in reality there
are many more factors at play

## Caveats and Recommendations
Use with discretion and ensure the metrics for your interested slice are large and accurate enough