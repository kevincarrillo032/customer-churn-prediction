# Customer Churn Prediction
This project aims to predict customer churn for a telecommunications company using machine learning techniques. The provided Jupyter Notebooks contain the data preprocessing steps, model selection process, and evaluation of the chosen model. Additionally, a Flask web application is included for testing the model locally.

## Overview
The project consists of the following components:

Data Preprocessing: The TelcoCustomerChurn.csv file contains the dataset used for training the machine learning model. The ModelSelection.ipynb notebook covers data cleaning, feature engineering, and encoding categorical variables.
Model Training: The Train_model.ipynb notebook focuses on training various machine learning models and selecting the best-performing one based on evaluation metrics.
Flask Web Application: The app.py file contains the Flask web application code, allowing users to interact with the trained model through a simple user interface.

## Getting Started
To clone and deploy this project, follow these steps:

### 1. Clone the Repository:
```bash
git clone https://github.com/kevincarrillo032/customer-churn-prediction.git
```

### 2. Install Dependencies:
```bash
pip install -r requirements.txt
```

### 3. Run the Flask Application Locally:
```bash
python app.py
```
Access the application at **'http://localhost:5000'**

### 4. Deploy to Heroku:
Create a new Heroku app using the Heroku ClI:
```bash
heroku create
```
Deploy the application to Heroku:
```bash
git push heroku main
```
Open the deployed application in your web browser:
```bash
heroku open
```

## File Structure
**.github/:** GitHub Actions workflow files for continuous integration.

**.ipynb_checkpoints/:** Checkpoint files generated by Jupyter Notebooks.

**templates/:** HTML templates for the Flask web application.

**Dockerfile:** Dockerfile for containerizing the application.

**Procfile:** Heroku Procfile specifying the commands to run the web application.

**TelcoCustomerChurn.csv:** Dataset containing telecommunications customer data.

**Train_model.ipynb:** Jupyter Notebook for training machine learning models.

**app.py:** Flask web application code.

**churnmodel.pkl:** Serialized trained machine learning model.

**heroku.yml:** Heroku configuration file for specifying runtime environment.
