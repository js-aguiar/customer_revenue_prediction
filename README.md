# Customer Lifetime Revenue Model (CLTR)

In this project, an end-to-end machine learning model is built to predict future customer revenue. The main goal is to forecast a customer's revenue for the next 90 days based on their purchase history at the basket-level. The resulting model and ETL are deployed using AWS services.


## Model development

The model development process is presented on the flowchart below. We start with a transactional dataset of 350 million purchases from more than 300 thousand customers. A sample is used to perform an exploratory analysis on the raw data (first notebook). After cleaning the initial dataset, we use a PySpark script through a SageMaker Processor to aggregate transactions and extract almost 50 features for each customer (second notebook). The resulting features are saved to an S3 bucket. Next, we build a baseline using the Lifetimes library (third notebook) to compare the results of the machine learning model with some benchmark. In the last notebook, we select 16 features to build a XGBoost model, perform hyperparameter tuning and deploy the final model to a SageMaker endpoint.


![model-dev-chart-png2](https://user-images.githubusercontent.com/36202803/134791421-8261ac3d-1d81-4a43-9fa4-28e10ad82fcd.png)


## Model performance and features

#### Results

The final model is compared to the baseline using an out-of-time sample (test dataset). The main evaluation metric was the Root Mean Squared Log Error (RMSLE), which has some desired properties considering the LTR problem (see the third notebook for details). The table below shows the resulting RMSE and RMSLE metrics for both models.

![image](https://user-images.githubusercontent.com/36202803/134818298-bba67545-103c-465f-8f32-f943ad5c3278.png)

#### Features

The next figure shows the most important features and their relationship with the target variable using the Shap library. The rate between customer's total purchase amount and the number of days since his first transaction was the most important feature. The purchase amount in the last 90 days and the store (chain) are also important. More recent spending is rewarded from the fourth to the seventh feature. The std_date_amount is penalizing customers with a more unstable spending profile, while recency_to_unique_dates (average number of days between purchases) is rewarding a higher purchase frequency.


![shap-plot](https://user-images.githubusercontent.com/36202803/134818379-c229f2eb-53d8-4f21-b23f-16f31c68396c.png)




## Production enviroment (work in progress)

In order to use the final XGBoost model endpoint for online predictions, we need to:
1) update our features according to new transactions;
2) retrieve features according to the customer id and pass it to the model endpoint.
3) return the predictions through an interface (API)

The flowchart below shows the architecture that is being build to perform these three tasks. The most up-to-date transactional data is extracted from some database or S3 bucket, processed using a Glue ETL job and stored in a relational database through AWS RDS. In parallel, an API Gateway is ready to receive requests from an external application and invoke a Lambda function, which will retrieve features according to the customer id from the relational database and pass it to the model endpoint. The model predictions are finally returned to the application.

![prod-chart-png2](https://user-images.githubusercontent.com/36202803/134791839-a9fe620e-e59e-4ff8-a11a-b0c6dd1b3c7e.png)


