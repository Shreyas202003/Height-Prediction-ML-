# Height-Prediction-ML-

<h3>Height Prediction Using ML</h3>
This repository contains a Jupyter Notebook that demonstrates how to predict the height of a child based on their parents' heights and the child's gender using a linear regression model.

Dataset
The dataset used is GaltonFamilies.csv, which includes the following columns:

rownames: Row identifier
family: Family identifier
father: Father's height
mother: Mother's height
midparentHeight: Midparent height
children: Number of children in the family
childNum: Child number
gender: Child's gender
childHeight: Child's height
Getting Started
Prerequisites
Ensure you have the following Python libraries installed:

sh
Copy code
pip install numpy pandas scikit-learn
Running the Notebook
Clone this repository to your local machine.
Open the Jupyter Notebook Height Prediction(assi-2).ipynb.
Ensure the GaltonFamilies.csv file is in the same directory as the notebook.
Run all cells in the notebook to train the model and make predictions.
Code Overview
Import Libraries:

python
Copy code
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
Load Data:

python
Copy code
dia = pd.read_csv('GaltonFamilies.csv')
Data Preparation:

python
Copy code
x = dia[['father', 'mother', 'gender']]
y = dia['childHeight']
Model Training:

python
Copy code
lr = LinearRegression()
lr.fit(x, y)
Prediction:

python
Copy code
a = lr.predict([[70, 65, 1]])
print(a[0])
Example Prediction
To predict the height of a child given a father's height of 70 inches, a mother's height of 65 inches, and the child being male (represented by 1):

python
Copy code
a = lr.predict([[70, 65, 1]])
print(a[0])  # Outputs the predicted height
Contributing
Feel free to fork this repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.
