# Emotional Arousal in News Headlines as a Predictor of Stock Market Behavior

Explored the impact of emotionally charged news headlines on short-term stock market behavior using natural language processing (NLP) and machine learning (ML) techniques. Applied emotion and sentiment classification models (fine-tuned DistilRoBERTa and DistilBERT), conducted feature engineering, and trained a regression model to predict 3-day stock returns. Completed as part of the AI4ALL Ignite accelerator, with a focus on responsible AI and data-driven insight into market psychology.


## Problem Statement

As emotionally charged headlines become increasingly prevalent in modern news media, understanding their influence on investor behavior grows ever more critical. Given that financial decisions are often guided by current events, the emotional tone of news may play a powerful role in shaping market trends.

<img width="700" height="379" alt="image" src="https://github.com/user-attachments/assets/0b213b70-6c73-4fd6-b54b-7cf096c03f94" />

> _Note._ Figure showing the shift in news headlines from positive sentiment to increasingly negative sentiment between 2000 and 2019. From "Longitudinal analysis of sentiment and emotion in news media headlines using  automated labelling with Transformer language models," by D. Rozado, R. Hughes, & J. Halberstadt, 2022, _PLOS ONE_, 17(10), e0276367. https://doi.org/10.1371/journal.pone.0276367

<img width="700" height="789" alt="image" src="https://github.com/user-attachments/assets/23ff5410-005b-4291-971d-39c1bf18a759" />

> _Note._ Figure showing the increase in news articles labeled with the emotions anger, disgust, fear, joy, and sadness, and a corresponding decrease in neutral articles from 2000-2019. From "Longitudinal analysis of sentiment and emotion in news media headlines using automated labelling with Transformer language models," by D. Rozado, R. Hughes, & J. Halberstadt, 2022, _PLOS ONE_, 17(10), e0276367.https://doi.org/10.1371/journal.pone.0276367

## Key Results 

1. Collected over 23 million news headlines from 47 media outlets and classified them by emotion and sentiment using pre-trained transformer models.
2. Filtered and analyzed a subset of headlines from 4 major news outlets (2015–2019) to align with S&P 500 stock market data.
3. Engineered a feature-rich dataset combining daily emotion/sentiment trends with 3-day stock market returns.
4. Trained and tuned a Random Forest Regression model using GridSearchCV to predict short-term market movement.
   - Achieved strong generalization performance with R² ≈ `0.74` and low mean absolute error.
5. Identified key emotional drivers of market fluctuation:
   - Days with high **fear** or **disgust** scores often corresponded with negative returns.
   - Days dominated by **joy** or **surprise** had a higher likelihood of positive returns.
6. Built an interactive Streamlit app that:
   - Classifies the sentiment and emotion of any headline input.
   - Predicts the expected 3-day market return.

## Visualizations & Graphs

### Classification Model
<img width="1400" height="1000" alt="classified_headlines_full" src="https://github.com/user-attachments/assets/644c6a22-a7d7-499c-9db9-50ddb4a3a402" />  

**Distribution of Emotions and Sentiments**  
Visualizes the emotional and sentiment breakdown of all classified headlines.

- Emotion
   - Mostly neutral headlines
   - Sadness as most prevalent emotion
   - Disgust as least prevalent emotion
   - Disgust as strongest and most unambiguous emotion
   - Surprise as weakest and most ambiguous emotion
- Sentiment
   - More negative than positive headlines
   - Positive and negative sentiments equal in strength and ambiguity

---

### Regression Model

<img width="1000" height="600" alt="actual_vs_predicted_mean_return_by_emotion" src="https://github.com/user-attachments/assets/0217555d-c331-47d5-b0cd-da02edbd5e06" />  

**Actual vs. Predicted Mean Return by Emotion**  
Compares the model’s predicted 3-day return to actual returns, averaged by emotion.
- Disgust with largest change in market
   - General increase to 3-day return
- Most emotions increase 3-day return
- Sadness with least change in market


<img width="1000" height="600" alt="feature_importances" src="https://github.com/user-attachments/assets/4829bec4-2b01-4a95-8b16-f28ac290b7d2" />  

**Feature Importances from Random Forest**  
Ranks the most predictive features used by the Random Forest model.
- High sentiment and headline count as most important features
- Disgust and anger as top two most important emotions

<img width="800" height="600" alt="scatter_actual_vs_predicted" src="https://github.com/user-attachments/assets/14eb3019-cade-4020-aee4-9b3b0dc83b43" />  

**Scatter Plot: Actual vs. Predicted Returns**  
Visualizes model fit by plotting predicted vs. actual returns for all days.


<img width="800" height="500" alt="abs_error_hist" src="https://github.com/user-attachments/assets/053fd5b5-bc98-4892-aa29-b4c1562dc8f4" />  

**Prediction Error Distribution (Histogram)**  
Shows how prediction errors are distributed, helping assess overall accuracy.

<img width="1000" height="600" alt="boxplot_error_by_emotion" src="https://github.com/user-attachments/assets/c4835ca8-c209-47d7-9d71-56f43012852b" />  

**Boxplot of Errors by Emotion**  
Highlights the spread and variance of prediction errors grouped by emotion.

<img width="800" height="500" alt="headlines_by_emotion" src="https://github.com/user-attachments/assets/00793517-12ce-44eb-8419-7440276bd2ee" />  

**Count of Headlines by Emotion**  
Displays the number of headlines per emotion to contextualize sample size.

## Methodologies 

To explore how emotionally charged news headlines influence short-term stock market movements, I combined natural language processing with financial data analysis.

- Fine-tuned two Hugging Face transformer models to classify over 23 million news headlines from 47 media outlets by **emotion** (j-hartmann/emotion-english-distilroberta-base) and **sentiment** (distilbert-base-uncased-finetuned-sst-2-english).

- Filtered and merged this data with S&P 500 stock market trends (2015–2019) collected via the Yahoo Finance Python API.

- Using `pandas`, we engineered daily features based on aggregated emotion and sentiment scores, such as average joy level, standard deviation of sentiment, and emotional volatility.

- Trained and tuned a **Random Forest Regressor** using `GridSearchCV` to predict the S&P 500’s 3-day return based on these features.

- To ensure generalizability and prevent overfitting, I used a 70/30 train-test split and 5-fold cross-validation during training.

- Visualizations and model interpretation were performed using matplotlib and seaborn to understand how certain emotions (like fear or joy) correlated with market movement.


## Data Sources 

- **News + Emotion Dataset**: [Zenodo Link](https://zenodo.org/records/7073014)  
  A large-scale dataset of news headlines (23M+ from 47 outlets) labeled with emotion and sentiment scores.

- **Transformer Models Used** (via Hugging Face):
  - **Sentiment**: [DistilBERT Base Uncased fine-tuned on SST-2](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english)  
  - **Emotion**: [English DistilRoBERTa-base fine-tuned for Emotion](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base)  


**Market Data Source**: [Yahoo Finance API](https://pypi.org/project/yfinance/#description) (`yfinance`)  
  Extracted historical S&P 500 data from **2015–2019**


## Technologies Used 

- Python
- pandas
- scikit-learn
- Hugging Face Transformers
- Streamlit
- Yahoo Finance API (`yfinance`)
- matplotlib
- seaborn


## Author  
- *Rachael DelNegro ([sep8vb@virginia.edu](mailto:sep8vb@virginia.edu))*
