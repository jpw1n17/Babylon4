# kaggle-Job-Salary-Prediction
Advanced Machine Learning group coursework

Describtion of Adzuna Salary Prediction [here](https://www.kaggle.com/c/job-salary-prediction)

Aim of this project:
*    Using different method to predict
*    Comprasion

# Step 1: data_cleanning

fill NaN and transofom strings to integer vector. At the sametime, update location information by mathching LocationTree.

orginal data format: 
![image](https://raw.githubusercontent.com/Trouble404/kaggle-Job-Salary-Prediction/master/readme_pic/word.PNG)

after [step 1](https://github.com/jpw1n17/Babylon4/blob/master/jz1g17/job-salary-datacleaning.ipynb)

![image](https://raw.githubusercontent.com/Trouble404/kaggle-Job-Salary-Prediction/master/readme_pic/wordtovec.PNG)

# Step 2: Train model

transfrom vector to one-hot vector and using Embedding or HashingVectorizer to process data in FullDescribtuion
