import spacy
import pandas as pd
from textblob import TextBlob

# Configure pandas display options for better readability
pd.set_option('expand_frame_repr', False)  # Prevent wrapping to new line if too many columns
pd.set_option('display.max_rows', 5000)  # Set maximum number of rows to display
pd.set_option('display.max_columns', 5000)  # Set maximum number of columns to display
# Ensure proper alignment for Chinese characters
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)

# Load the spaCy English language model
nlp = spacy.load("en_core_web_sm")

# Define a function to preprocess text by lemmatizing and removing stopwords and punctuation
def preprocess_text(text):
    doc = nlp(text)
    clean_text = ' '.join(token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.lemma_.isalpha())
    return clean_text

# Define a sentiment analysis function using TextBlob after preprocessing text with spaCy
def analyze_sentiment(review):
    clean_review = preprocess_text(review)
    analysis = TextBlob(clean_review)
    sentiment = analysis.sentiment.polarity
    return "Positive" if sentiment > 0.1 else "Negative" if sentiment < -0.1 else "Neutral"

# Load the dataset
df = pd.read_csv("amazon_product_reviews.csv")

# Map numerical ratings to sentiment scores and then to sentiment labels
sentiment_score = {1: -1, 2: -1, 3: 0, 4: 1, 5: 1}
sentiment = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}

df['sentiment_score'] = df['reviews.rating'].map(sentiment_score)
df['sentiment'] = df['sentiment_score'].map(sentiment)

# Select the 'reviews.text' column and remove missing values
clean_reviews = df.dropna(subset=['reviews.text'])

# Apply sentiment analysis to the entire cleaned reviews and save the results in a new column 'sentiment_predict'
clean_reviews['sentiment_predict'] = clean_reviews['reviews.text'].apply(analyze_sentiment)

# Preview the results
print(clean_reviews[['reviews.text', 'sentiment', 'sentiment_predict']].head())

# Calculate the accuracy of sentiment predictions
correct_predictions = (clean_reviews['sentiment'] == clean_reviews['sentiment_predict']).sum()
total_predictions = clean_reviews.shape[0]
accuracy = correct_predictions / total_predictions

print(f"Accuracy: {accuracy:.4f}")

# 5.1 This dataset is a list of 5,000 consumer reviews for Amazon products like the Kindle, Fire TV Stick,
# and more from Datafiniti's Product Database updated between September 2017 and October 2018. Each product listing
# includes the name Amazon in the Brand and Manufacturer field. All fields within this dataset have been flattened,
# with some omitted, to streamline your data analysis. This version is a sample of a large dataset.

# 5.2. Details of the Preprocessing Steps
# The preprocessing steps are crucial for preparing the text data for sentiment analysis. These steps involve:
# 1. Loading the spaCy Model: Utilizing the 'en_core_web_sm' model from spaCy for its NLP capabilities.
# 2. Text Cleaning Function: A custom function 'preprocess_text' performs several key operations:
#    - Tokenization and Lemmatization: Converts text into tokens and reduces them to their base forms.
#    - Removing Stopwords and Punctuation: Eliminates common words and punctuation to focus on meaningful content.
#    - Filtering Non-Alpha Tokens: Keeps only alphabetical tokens, discarding numbers and symbols.
# These steps help in standardizing the text, making it more suitable for analysis by reducing noise.

# 5.3. Evaluation of Results
# The model's performance is evaluated by calculating accuracy, which is the ratio of correctly predicted sentiments to the total predictions:
# - Correct Predictions: Counts instances where predicted sentiment matches the actual sentiment.
# - Total Predictions: The overall number of predictions made by the model.
# - Accuracy(81.42%): Determined by dividing the number of correct predictions by the total predictions, providing a measure of the model's effectiveness.

# 5.4. Insights into the Model's Strengths and Limitations
# Strengths:
# - Flexibility and Extensibility: The model's use of spaCy and TextBlob allows for easy adaptation to more complex tasks.
# - Efficiency: Preprocessing steps like stopwords removal and lemmatization improve the model's efficiency by focusing on relevant text.
# - Simplicity: Offers a straightforward approach to sentiment analysis, suitable for quick insights or projects with limited resources.
# Limitations:
# - Dependence on Predefined Lexicons: Reliance on TextBlob's lexicons may limit accuracy for nuanced or domain-specific texts.
# - Lack of Context Awareness: Processes each review independently, missing out on contextual cues that could influence sentiment.
# - Binary and Neutral Classification: The model's simplistic classification might not capture the full range of emotions in the reviews.