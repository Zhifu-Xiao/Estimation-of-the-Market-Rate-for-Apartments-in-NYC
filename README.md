# Estimation of the Market Rate for Apartments in NYC with Python

### Summary
In this project, a real estate agent wants to estimate the market rate for some apartments in NYC that just went into the market again using the data posted by the Census in 2014. Throughout this project, we mainly did three parts: data preprocessing, training model and use trained model to fit the test set. Our final results yield over 80% on R-squared and near to 90%.

### Process and Results
There are six steps to preprocess, build and validate the data and our linear model. The steps are listed below.

1.	**Preprocessing:** According to the requirements, we should only use features that apply to pricing an apartment that is not currently rented. Therefore, we remove all the variables that are not related to the apartments. Then we handle missing value in different columns by replacing the different missing data (8,9,98,99…,9999998,9999999) by nan. Besides, we drop the observations with nan in objective feature before splitting data in order to use the more accurate data to train and fit the data. After preprocessing, we divided the dataset into X_train, X_test, y_train, y_test for next step

2.	**Imputation:** For training data, since the dataset includes both continuous and discrete features, we attempted to split the data into continuous and categorical columns and use different ways to impute them separately. As we all know, Model-Driven Imputation is very flexible, so we choose to use it in our dataset—Fanyimpute Knn in categorical data and mice in real valued columns (MICE is iterative and works well often). 

3. **Feature Distribution & One-Hot Encoder:** After we filled out the missing value, we then applied box-cox transformation on the continuous training data to scale the data, and for multi-levels categorical train data, we used OneHotEncoder to change multi-levels data into binary-level data. We then merged the continous and categorical data together for the next step.

4. **Feature Selection:** We chose LassoCV to select feature from more than 400 variables, and we finally obtained 39 features that are most important to the model, and we used GridSearchCV to find out the best parameter alpha.

5. **Linear Model Build:** After we selected 39 best features among more than 400 variables, we applied MinMaxScalar and use Lasso linear model to fit our training data. The R-squared on the training data is more than 0.9, and the R-squared on the cross-validated data is around 0.9, which is very satified. We then saved the model parameters in order to predict the test data.

6. **Predict test data:** We first applied the same imputation, feature distribution, One-Hot Encoder and feature selection on the test dataset to make sure that they are in the same standard. Then, we used the fitted model to predict our test data. The R-squared for test data is near 90%, which is higher than the threhold we set (80%). From the results we could see that through effective tools to preprocess and impute the dataset, the preducting accuracy could be relatively high and the simple model could be very useful to explain the results.
