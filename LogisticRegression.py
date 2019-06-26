from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	df= pd.read_csv("Job titles and industries.csv", encoding="latin-1")
	# Features and Labels
	df['industry'] = df['industry'].map({'IT': 0, 'Accountancy': 1, 'Marketing':2, 'Education':3 })
	X = df['job title']
	y = df['industry']
	
	# Extract Feature With CountVectorizer
	cv = CountVectorizer()
	X = cv.fit_transform(X) # Fit the Data
	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

	#Logistic Regression Classifier
	clf = LogisticRegression()
	clf.fit(X_train,y_train)
	clf.score(X_test,y_test)


	if request.method == 'POST':
		message = request.form['job title']
		data = [message]
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
	return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)
