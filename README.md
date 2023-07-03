# Predicting Dota 2 Win Rates and KDA

## Introduction
In this project we build an XGBRegressor model to predict the performance (win rate and KDA ratio) of Dota 2 players for their 
10th hero based on their performance with their top 9 most played heroes. By analyzing the characteristics and attributes of Dota 2 heroes, 
the model learns patterns and correlations to make predictions.

## Dataset
The dataset we are using for this project is publicly available on Kaggle (https://www.kaggle.com/datasets/arashnic/dota2game).
The data are already split to training (9 heroes) and test (10th hero) datasets. 
The data consists of player performance (win rates and KDA ratios), and hero attributes.
The datasets are stored in a structured format (CSV) and is we process it to extract the relevant features for training the model.

## Dependencies
- Python (version 3.7)
- pandas 
- scikit-learn 
- xgboost 
- matplotlib 
- seaborn
- numpy

## Results
The trained XGBRegressor model achieves a decent performance in predicting KDA ratio of Dota 2 players for their 10th hero.
However, it does underperform in predicting win rates.  
We use evaluation metrics, such as RMSE and R-squared, to indicate the model's effectiveness in capturing the correlations in the dataset.

## Conclusion
Approximately 65% of the variability observed in the KDA ratio can be predicted by our model, though this value drops to about 40% for the win rate predictions. 
Regardless, both models perform better than a simple baseline model, demonstrating the power of machine learning models.

## Acknowledgments
- Data used in this project is downloaded from Kaggle: https://www.kaggle.com/datasets/arashnic/dota2game
- [OpenAI](https://openai.com/) for developing the GPT-3.5 language model used for assistance in this project
