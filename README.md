# Leveraging AI Analytics to Predict and Manage Customer Churn

## Inspiration
The cost of acquiring customers is often greater than retaining them.

Recognizing the importance of customer retention, we leveraged advanced data analytics and artificial intelligence to develop a machine-learning model to predict and analyze customer churn.

## What it does
Our machine-learning model employs a gradient-boosting ensemble algorithm, fine-tuned through hyperparameter optimization to enhance its predictive capabilities. The process involves feeding customer data— including unique identifiers and relevant features—into our model, which then effectively predicts customer churn with an F1 score of 0.97.

## How we built it
Our machine learning pipeline follows a systematic approach to handle data preprocessing and model selection. Initially, the data is structured using a Pandas data frame within Python. Matplotlib and NumPy were employed for exploratory data analysis, providing insights that guided our preprocessing steps.

Various feature engineering techniques were implemented based on the unique characteristics of our data. Python scripts were developed to handle tasks such as missing value thresholding and date-time feature extraction. Additionally, we utilized Orange Data Mining Software for feature-importance ranking via Gini Decrease analysis. To address missing values, we employed K-Nearest Neighbors imputation for date-time objects and mode/median imputation for categorical and numerical features, respectively. To mitigate high-class imbalance, we integrated oversampling techniques from the imbalanced-learn library. Scikit-learn was used in both label-encoding of our binary categorical features and one-hot encoding of our nominal categorical features. The library was also utilized for training our model—both AdaBoost and GradientBoosting, ensemble boosting algorithms, were tested across different numbers of base estimator trees to determine the configurations that produced optimal predictive performance. Finally, our optimized model—GradientBoosting ensemble model with 70 base estimator trees - was retrained with the full dataset.

## Challenges we ran into
### Encoding Predominantly Categorical Features + Computational Resource Limitations
Since one-hot-encoding creates a new column for every distinct value of a categorical feature, encoding a large dataframe of predominantly categorical features (with each containing numerous distinct values!) led to a dataframe size that exceeded our development environments computational resources. This led to repeated crashes and inability to execute our original scripts

To tackle this challenge, we utilized Orange Data Mining Software to process our feature-importance ranking by gini decrease and filtered these features from our dataframe

### Misalignments between training and test data
This is another challenge that stemmed from the categorical nature of the dataset—Our GradientBoosting classifier required both datasets to have the exact same features in the same order. However, since one-hot-encoding creates a new column for each distinct value of a categorical feature, it would only take the absence of a value present in the training set but missing in the test set, or vice versa, to disrupt the alignment of data structures and produce a model malfunction.

After troubleshooting and identifying the root cause of the malfunction, we implemented a function to rectify the misalignment between datasets. This involved removing encoded columns present in the inference data but not in the test data, and vice versa—creating encoded columns in the inference dataframe for features present in the test data but missing in the inference data, defaulting them to 0.
