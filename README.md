# Intern_Task

# Space-ship Task README
This repository contains code for the Space-ship Task, which involves predicting whether a passenger will be transported or not on a space ship based on various features. The task involves data preprocessing, feature engineering, visualization, and building classification models using different algorithms.

##Dataset
The dataset consists of two CSV files: train.csv and test.csv. The train.csv file is used for training the models, while the test.csv file is used for evaluating the models. Each row in the dataset represents a passenger and contains several features such as age, home planet, cryo sleep status, cabin type, destination, and more.

##Data Preprocessing
The first step in the analysis is to load the dataset using the Pandas library. The train.csv file is read into a Pandas DataFrame called train_df, and the test.csv file is read into another DataFrame called test_df. Basic information about the train_df DataFrame is printed using the info() function, and the first few rows of the DataFrame are displayed using the head() function.

Next, missing values in the numerical columns of train_df are imputed using the mean strategy. The SimpleImputer class from the scikit-learn library is used for this purpose. The missing values in categorical columns are imputed using the most frequent strategy.

##Feature Engineering
A new feature called Total_Bill is computed as the sum of the RoomService, ShoppingMall, Spa, and VRDeck columns. The mean, median, maximum, and minimum values of the Total_Bill column are also calculated and printed.

The categorical columns in the dataset are encoded using label encoding. The LabelEncoder class from scikit-learn is used for this purpose. The encoded DataFrame is stored in train_df_encoded. A pairplot and a heatmap are created to visualize the relationships between the features in the encoded DataFrame.

##Visualization
Several visualizations are created to explore the data. A scatter plot is generated to visualize the relationship between the Age and Total_Bill columns. A countplot is created to show the distribution of CryoSleep values with respect to the Transported variable. The unique values of the HomePlanet, Destination, and Cabin columns are printed, and a countplot is created to visualize the distribution of CryoSleep values with respect to the Transported variable.

##Dimensionality Reduction
Two dimensionality reduction techniques, t-SNE (t-Distributed Stochastic Neighbor Embedding) and PCA (Principal Component Analysis), are applied to the encoded DataFrame for visualization purposes. The reduced features are plotted in scatter plots to visualize the clusters.

##Model Building
The encoded DataFrame is split into input features (X) and the target variable (Y). The data is then split into training and testing sets using the train_test_split function from scikit-learn. Four classification models, Logistic Regression, Decision Tree, Random Forest, and Support Vector Machine (SVM), are trained on the training set and evaluated on the testing set. The accuracy score is calculated for each model to measure their performance.

Finally, an XGBoost model is trained on the training set and evaluated on the testing set. The XGBoost package is installed using pip before training the model. The accuracy score is calculated and printed.

##Requirements
The code is written in Python and requires the following libraries:

pandas
numpy
seaborn
scikit-learn
matplotlib
xgboost
These libraries can be installed using pip or conda.

##Usage
Make sure you have Python installed.
Clone this repository or download the code files (space_ship_task.ipynb).
Install the required libraries mentioned in the Requirements section.
Open the space_ship_task.ipynb file in a Jupyter Notebook or any other Python IDE.
Run the cells in the notebook sequentially to execute the code.

##Conclusion
The Space-ship Task code provides a comprehensive analysis of the given dataset and demonstrates the process of data preprocessing, feature engineering, visualization, and building classification models. The models can be further fine-tuned and optimized to achieve better performance.
