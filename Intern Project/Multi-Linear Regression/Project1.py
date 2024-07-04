import pandas as pd
import statsmodels.api as sm

# Load the dataset
data = pd.read_csv(r"C:\Users\sreev\OneDrive\Desktop\Project.csv")

# Define the dependent variable (Sales)
Y = data['Sales']

# Define the independent variables (TV, Radio, Newspaper)
X = data[['TV', 'Radio', 'Newspaper']]

# Add a constant to the model (intercept)
X = sm.add_constant(X)

# Fit the multiple linear regression model
model = sm.OLS(Y, X).fit()

# Print the model summary
print(model.summary())
