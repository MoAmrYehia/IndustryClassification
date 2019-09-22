# Industry Classification


Description:

You can think of the job industry as the category or general field in which you work. On a job application, "industry" refers to a broad category under which a number of job titles can fall. For example, sales is an industry; job titles under this category can include sales associate, sales manager,manufacturing sales rep, pharmaceutical sales and so on.

Details:

Given a dataset that has two variables (Job title & Industry) in a csv format of more than 8,500 samples.This dataset is imbalanced (Imbalance means that the number of data points available for different classes is different) as follows:

IT 4746
Marketing 2031
Education 1435
Accountancy 374

1) I started using NLTK and RE libraries to drop Special characters. Then to make all characters in lower case. After that, I made a list of words to remove irritant word using stopwords.Then I used STEM to covert verbs to the root word.
2)I created a simple comparison between different algorithms to take a look on mean and standard deviation. You can find the code in Compare.py

How do you deal with (Imbalance learning)?

1. Change the performance metric
2. Change the algorithm
3. Oversample minority class
4. Undersample majority class
5. Generate synthetic samples

How can you extend the model to have better performance?

Use Ensemble Modeling to improve accuracy by applying stacking, that often considered heterogeneous learners, learns them in parallel and combines them by training a meta-model to output a prediction based on the different models predictions.

What are the limitations of Logistic regression or Where does your approach fail?

Logistic regression attempts to predict outcomes based on a set of independent variables, but if researchers include the wrong independent variables, the model will have little to no predictive value. This means that logistic regression is not a useful tool unless researchers have already identified all the relevant independent variables.
