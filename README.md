# Customer Lifetime Revenue (CLTR)

In this project, we will be building an end-to-end machine learning model to predict the revenue of a given customer for the next 90 days.

We will also compare the deployed model with a parametric model approach using the Lifetimes library.

The whole development process was done using SageMaker Studio, including the deployment of the model to production. The final model is hosted in AWS and it is possible to make online predictions through an endpoint.

### Requirements

- numpy
- pandas
- seaborn
- sagemaker
- boto3
- xgboost
- awswrangler
- lifetimes