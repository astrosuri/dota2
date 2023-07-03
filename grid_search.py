import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np

def run_grid_search(X_train, y_train, X_test, y_test, model, cv, param_grid, scoring):
    '''
    Wrapper function to do grid search with cross validation.
    
    Parameters:
        - X_train: dataframe
            Training dataset
        - X_test: dataframe
            Test dataset
        - y_test: array
            True target
        - y_pred: array
            Predicted target
        - model: object
            A regressor or a classifier
        - cv: int
            Number of cross validation subsets
        - param_grid: dict
            Hyperparameters and the parameter space to perform the search
        - scoring: str, callable, list, tuple, dict
            Method to evaluate the cross validated models
            
    Returns:
        - best_model, rmse, r2
    '''

    # Create a GridSearchCV object
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring=scoring)

    # Perform grid search on your training data
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters and best model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    # Train the best model on the entire training data
    best_model.fit(X_train, y_train)

    # Evaluate the best model on the test data
    y_pred = best_model.predict(X_test)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Print the evaluation results
    print(f"Best Parameters: {best_params}")
    print(f"RMSE: {rmse}")
    print(f"R-squared: {r2}")

    # Plot the actual vs fitted values
    plt.figure(figsize=(10, 8))
    ax = sns.distplot(y_test, hist=False, color="k", label="True win rate")
    sns.distplot(y_pred, hist=False, color="#008B8B", label="Predicted win rate", ax=ax)
    plt.title('Actual vs Fitted Values for Win Rate')
    plt.legend(loc=2)
    plt.show()

    # Return the best model and evaluation results
    return best_model, rmse, r2
