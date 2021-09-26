# Customer Lifetime Revenue (CLTR) Model

In this project, an end-to-end machine learning model is built to predict customer future revenue. The resulting model and ETL are deployed using AWS. The main goal is to forecast the revenue of a customer for the next 90 days given its purchase history.


## Model development

The model development process is presented on the flowchart below. We start with a transactional dataset with 350 million purchases from more than 300 thousand customers. A sample is used to perform a exploratory analysis for the raw data (first notebook). After cleaning the raw data, we use a PySpark script through a SageMaker Processor to aggregate transactions and extract almost 50 features from the transactional data, which are saved to an S3 bucket (second notebook). A baseline is built using the Lifetimes library (third notebook) in order to compare the results of the machine learning model. In the last notebook, we select 16 features to build a XGBoost model, perform hyperparameter tuning and deploy the final model to a SageMaker endpoint.


![model-dev-chart-png2](https://user-images.githubusercontent.com/36202803/134791421-8261ac3d-1d81-4a43-9fa4-28e10ad82fcd.png)


## Production enviroment (work in progress)

In order to use the XGBoost model for online predictions, we need to: 1) update our features according to new transactions and 2) retrieve features according to the customer id and pass it to the model endpoint. To achieve that, 

![prod-chart-png2](https://user-images.githubusercontent.com/36202803/134791839-a9fe620e-e59e-4ff8-a11a-b0c6dd1b3c7e.png)


## Model performance

The final model is compared to the baseline using an out-of-time sample. The main evaluation metric was the Root Mean Squared Log Error (RSMLE) which is robust to outliers.

## Main features

...
