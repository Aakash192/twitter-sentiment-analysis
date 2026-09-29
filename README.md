# Twitter Sentiment Analysis

Classifies tweets as Positive, Negative or Neutral. The notebook compares a rule based baseline (VADER) with a Multinomial Naive Bayes classifier trained on the cleaned tweet text.

Everything is in `Twitter_Sentiment_Analysis.ipynb`.

## Data

`twitter_training.csv` from the Twitter Entity Sentiment Analysis dataset on Kaggle. Each row has an ID, a topic (a brand or game such as Borderlands), a sentiment label and the tweet text. Rows labelled `Irrelevant` are removed. The file is not committed to this repo.

## Pipeline

1. **Explore.** Plot the label distribution and word clouds for positive and negative tweets.
2. **Clean.** Lowercase, strip punctuation, tokenise with NLTK, remove English stopwords and apply Porter stemming.
3. **Baseline.** Label each tweet with NLTK's VADER lexicon, using compound score thresholds of +0.05 and -0.05.
4. **Classifier.** A scikit-learn pipeline of `CountVectorizer` (bag of words) and `MultinomialNB`, trained on an 80/20 split with `random_state=42`.

## Results

Naive Bayes on the 12,339 tweet validation set:

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Negative | 0.75 | 0.84 | 0.79 |
| Neutral | 0.83 | 0.62 | 0.71 |
| Positive | 0.75 | 0.81 | 0.78 |
| **Overall accuracy** | | | **0.77** |

Neutral tweets are the hardest class. Recall is 0.62 and most misses are predicted as Negative.

The VADER baseline often disagrees with the dataset's labels. For example, gaming tweets like "I will murder you all" are labelled Positive in the data, and VADER reads them as Negative. This is why a model trained on the dataset's own labels does better than a general purpose lexicon here.

## Limitations and next steps

- The dataset contains several near identical versions of the same tweet under one ID. A random split can put versions of the same tweet in both training and validation, which may inflate the score. Splitting by tweet ID would give a more honest estimate.
- Try TF-IDF weighting and a linear model such as Logistic Regression or a linear SVM, and compare against this baseline.
- Evaluate on `twitter_validation.csv`, the held out file that ships with the dataset.

## Run it

```bash
pip install pandas numpy nltk scikit-learn matplotlib seaborn wordcloud
jupyter notebook Twitter_Sentiment_Analysis.ipynb
```
