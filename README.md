# Intern_Task

# Space-ship Task 
This repository contains code for the Space-ship Task, which involves predicting whether a passenger will be transported or not on a space ship based on various features. The task involves data preprocessing, feature engineering, visualization, and building classification models using different algorithms.

## Dataset
The dataset consists of two CSV files: train.csv and test.csv. The train.csv file is used for training the models, while the test.csv file is used for evaluating the models. Each row in the dataset represents a passenger and contains several features such as age, home planet, cryo sleep status, cabin type, destination, and more.

## Data Preprocessing
The first step in the analysis is to load the dataset using the Pandas library. The train.csv file is read into a Pandas DataFrame called train_df, and the test.csv file is read into another DataFrame called test_df. Basic information about the train_df DataFrame is printed using the info() function, and the first few rows of the DataFrame are displayed using the head() function.

Next, missing values in the numerical columns of train_df are imputed using the mean strategy. The SimpleImputer class from the scikit-learn library is used for this purpose. The missing values in categorical columns are imputed using the most frequent strategy.

## Feature Engineering
A new feature called Total_Bill is computed as the sum of the RoomService, ShoppingMall, Spa, and VRDeck columns. The mean, median, maximum, and minimum values of the Total_Bill column are also calculated and printed.

The categorical columns in the dataset are encoded using label encoding. The LabelEncoder class from scikit-learn is used for this purpose. The encoded DataFrame is stored in train_df_encoded. A pairplot and a heatmap are created to visualize the relationships between the features in the encoded DataFrame.

## Visualization
Several visualizations are created to explore the data. A scatter plot is generated to visualize the relationship between the Age and Total_Bill columns. A countplot is created to show the distribution of CryoSleep values with respect to the Transported variable. The unique values of the HomePlanet, Destination, and Cabin columns are printed, and a countplot is created to visualize the distribution of CryoSleep values with respect to the Transported variable.

## Dimensionality Reduction
Two dimensionality reduction techniques, t-SNE (t-Distributed Stochastic Neighbor Embedding) and PCA (Principal Component Analysis), are applied to the encoded DataFrame for visualization purposes. The reduced features are plotted in scatter plots to visualize the clusters.

## Model Building
The encoded DataFrame is split into input features (X) and the target variable (Y). The data is then split into training and testing sets using the train_test_split function from scikit-learn. Four classification models, Logistic Regression, Decision Tree, Random Forest, and Support Vector Machine (SVM), are trained on the training set and evaluated on the testing set. The accuracy score is calculated for each model to measure their performance.

Finally, an XGBoost model is trained on the training set and evaluated on the testing set. The XGBoost package is installed using pip before training the model. The accuracy score is calculated and printed.

## Requirements
The code is written in Python and requires the following libraries:

pandas
numpy
seaborn
scikit-learn
matplotlib
xgboost
These libraries can be installed using pip or conda.

## Usage
Make sure you have Python installed.
Clone this repository or download the code files (space_ship_task.ipynb).
Install the required libraries mentioned in the Requirements section.
Open the space_ship_task.ipynb file in a Jupyter Notebook or any other Python IDE.
Run the cells in the notebook sequentially to execute the code.

## Conclusion
The Space-ship Task code comprehensively analyzes the given dataset and demonstrates the process of data preprocessing, feature engineering, visualization, and building classification models. The models can be further fine-tuned and optimized to achieve better performance.

## IPL Task 
This code performs an analysis of IPL (Indian Premier League) data using two datasets: deliveries.csv and matches.csv. The code utilizes Python libraries such as pandas, numpy, and matplotlib for data manipulation, analysis, and visualization.

# Deliveries Analysis
The deliveries.csv dataset contains information about each ball delivered in IPL matches. The code performs the following analysis:

Prints the information about the dataset using the info() function.
Displays the shape of the dataset using the shape attribute.
Shows the first few rows of the dataset using the head() function.
Provides summary statistics of the dataset using the describe() function.
Checks for missing values using the isnull().sum() function.
Identifies unique values in certain columns such as batting team, bowling team, batsman, and bowler.
Visualizes the first and second innings scores for each batting team using horizontal bar plots.
Calculates and visualizes the total runs scored by the top 10 batsmen using a horizontal bar plot.
Counts the number of dismissals (wickets) using the player_dismissed column and visualizes the top 10 bowlers with the most dismissals using a horizontal bar plot.
Computes the number of sixes hit by each batting team and visualizes it using a horizontal bar plot.
Computes the number of fours hit by each batting team and visualizes it using a horizontal bar plot.
Computes the number of fours hit by each player and visualizes the top 10 players with the most fours using a horizontal bar plot.
Computes the number of sixes hit by each player and visualizes the top 10 players with the most sixes using a horizontal bar plot.

## Match Analysis
The matches.csv dataset contains information about each IPL match. The code performs the following analysis:

Prints the information about the dataset using the info() function.
Counts the number of times each player received the "Player of the Match" award and visualizes the top 10 players using a horizontal bar plot.
Visualizes the number of matches won by each team at each venue using bar plots.
The code provides insights into various aspects of IPL matches, including team performance, player performance, and match outcomes. The visualizations help in better understanding the data and identifying trends.

## Requirements
The code is written in Python and requires the following libraries:

pandas
numpy
matplotlib
These libraries can be installed using pip or conda.

## Usage
Make sure you have Python installed.
Download the deliveries.csv and matches.csv datasets.
Install the required libraries mentioned in the Requirements section.
Open a Python IDE or Jupyter Notebook.
Import the necessary libraries (pandas, numpy, matplotlib.pyplot).
Copy the code into the Python environment.
Modify the file paths in the pd.read_csv() function to load the datasets.
Run the code to perform the analysis and visualize the results.

## Conclusion
The IPL Task code provides an analysis of IPL data, exploring various aspects such as team performance, player performance, and match outcomes. The code can be extended and modified to perform further analysis based on specific requirements and objectives.
