
# Sentiment Analysis on Customer Reviews

## Project Overview
This project performs sentiment analysis on customer reviews using Natural Language Processing (NLP) techniques. The goal is to classify reviews as either positive or negative based on the text content. We use machine learning models like Logistic Regression and Random Forest to make predictions based on features extracted from the review text.

## Dataset
The dataset consists of a small set of customer reviews with the following columns:
- **Id**: Unique identifier for the review.
- **ProductId**: Identifier for the product being reviewed.
- **UserId**: Identifier for the user who left the review.
- **ProfileName**: The profile name of the reviewer.
- **HelpfulnessNumerator**: The number of users who found the review helpful.
- **HelpfulnessDenominator**: The number of users who rated the review.
- **Score**: The rating given by the reviewer (1 to 5), where ratings of 3 or higher are considered positive and less than 3 are considered negative.
- **Summary**: A short summary of the review.
- **Text**: The full review text used for sentiment analysis.

### Example Rows from the Dataset:
| Id  | Score | Text                                                       |
| --- | ----- | ----------------------------------------------------------- |
| 1   | 5     | I have bought several of the Vitality canned dog food ...    |
| 2   | 1     | Product arrived labeled as Jumbo Salted Peanuts ...          |
| 3   | 4     | This is a confection that has been around a few centuries... |
| 4   | 2     | If you are looking for the secret ingredient in cough...     |
| 5   | 5     | Great taffy at a great price...                              |

## Project Workflow
1. **Data Loading**: We load the review dataset and inspect the contents.
2. **Text Preprocessing**:
   - Remove special characters and convert text to lowercase.
   - Remove stopwords (commonly used words like "and", "the", "is").
3. **Feature Engineering**: Convert the cleaned text into numerical features using the `CountVectorizer`, which creates a document-term matrix.
4. **Model Training**:
   - **Logistic Regression**: A simple linear model for binary classification.
   - **Random Forest**: An ensemble model based on decision trees, used for improving accuracy.
5. **Model Evaluation**: Measure model performance using accuracy, precision, recall, and F1 score.
6. **Data Visualization**: 
   - Visualize word frequencies from the reviews.
   - Display feature importance for the Random Forest model.

## Machine Learning Models
- **Logistic Regression**: Used as a baseline model for classification.
- **Random Forest**: Used to improve the baseline and better capture complex patterns in the data.

## How to Run the Project
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/sentiment-analysis.git
   ```
2. **Install Dependencies**:
   Install the required Python libraries using:
   ```bash
   pip install pandas numpy scikit-learn nltk matplotlib
   ```
3. **Download the Dataset**:
   Place the provided `sample_reviews_dataset.csv` in the project directory.
4. **Run the Python Script**:
   Execute the Python script to train models and visualize results:
   ```bash
   python sentiment_analysis.py
   ```

## Results
- **Logistic Regression**:
  - Accuracy: ~80% (depends on dataset).
- **Random Forest**:
  - Accuracy: Higher than Logistic Regression, around ~85%.

## Visualization
- **Top Words by Frequency**: Shows the most frequent words in the dataset.
- **Random Forest Feature Importance**: Displays which features had the most influence on the model's predictions.

## Future Work
- Explore more advanced NLP techniques like TF-IDF, word embeddings (Word2Vec), and deep learning models (LSTM, BERT).
- Scale the project to a larger dataset for improved generalizability.
