# Twitter Sentiment Analysis with XGBoost and Logistic Regression

A comprehensive machine learning project for sentiment analysis on Twitter data using advanced NLP preprocessing, XGBoost, and Logistic Regression models with hyper-parameter optimisation.

## Project Overview

This project implements a robust sentiment analysis pipeline that classifies Twitter posts as positive or negative sentiment. The solution combines traditional machine learning algorithms (XGBoost and Logistic Regression) with advanced NLP preprocessing techniques to achieve high accuracy in sentiment classification.

## Features

- **Advanced Text Preprocessing**: Custom tokenization using spaCy with lemmatization
- **Multiple Model Implementation**: XGBoost and Logistic Regression with GridSearchCV
- **Feature Engineering**: TF-IDF vectorization with customizable parameters
- **Dimensionality Reduction**: TruncatedSVD and PCA for feature optimization
- **Cloud Integration**: AWS S3 integration for model storage and deployment
- **Comprehensive Evaluation**: Detailed performance metrics and confusion matrices
- **Scalable Pipeline**: Modular design for easy customization and extension

## Dataset

The project uses Twitter sentiment data with the following structure:
- **target**: Sentiment label (0 = negative, 4 = positive)
- **id**: Tweet ID
- **date**: Tweet timestamp
- **flag**: Query flag
- **user**: Username
- **text**: Tweet content

## Requirements

Install the required dependencies:
```bash
pip install -r requirements.txt
```


**Key Dependencies:**
numpy  
pandas  
scikit-learn  
xgboost  
nltk  
spacy  
beautifulsoup4  
boto3  
dill  
tqdm

## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/RAKSAurum/Sentiment_Analysis_on_Twitter_Data_Using_XGB_LR.git
cd Sentiment_Analysis_on_Twitter_Data_Using_XGB_LR
```

## Usage

### Running the Analysis

1. **Open the Jupyter notebook:**
```bash
jupyter notebook Sentiment_Analysis_Twitter_Data_XGB_&_LR.ipynb
```

2. **Execute the cells sequentially** to:
   - Load and preprocess the Twitter dataset
   - Perform exploratory data analysis
   - Train XGBoost and Logistic Regression models
   - Evaluate model performance
   - Save trained models to AWS S3

### Key Components

**Data Preprocessing Pipeline:**
- HTML tag removal with BeautifulSoup
- Custom tokenization with spaCy lemmatization
- Stop word removal and text normalization
- TF-IDF vectorization with custom parameters

**Model Training:**
##### XGBoost with GridSearchCV

```python
xgb_model = XGBClassifier(  
learning_rate=0.01,  
max_depth=5,  
n_estimators=5000,  
colsample_bytree=0.8,  
gamma=0.1,  
min_child_weight=5  
)
```

#### Logistic Regression with optimization

```python
lr_model = LogisticRegression()
```


**Feature Engineering:**
- TF-IDF vectorization with custom tokenizer
- Dimensionality reduction using TruncatedSVD
- Pipeline integration for streamlined processing

## Model Architecture

### XGBoost Configuration
- **Learning Rate**: 0.01
- **Max Depth**: 5
- **Estimators**: 5000
- **Column Sample**: 0.8
- **Gamma**: 0.1
- **Min Child Weight**: 5

### Logistic Regression
- Optimized with GridSearchCV
- L2 regularization
- Custom solver selection

## AWS Integration

The project includes AWS S3 integration for:
- Model persistence and versioning
- Automated model deployment
- Scalable storage solutions
```python
# Model upload to S3
def create_upload(object, file_name):  
session = datetime.now().strftime('%Y%m%d_%H%M%S')  
full_name = '{}_{}.pkl'.format(file_name, session)  
with open(full_name, 'wb') as file:  
pickle.dump(object, file)
```

## Performance Metrics

The project evaluates models using:
- **Accuracy Score**
- **Confusion Matrix**
- **Classification Report** (Precision, Recall, F1-Score)
- **Cross-validation Scores**

## Technical Details

### Text Preprocessing
- **Tokenization**: spaCy-based with lemmatization
- **Cleaning**: HTML removal, URL handling, special character processing
- **Feature Extraction**: TF-IDF with n-gram support
- **Dimensionality Reduction**: SVD for computational efficiency

### Hyperparameter Optimization
- GridSearchCV for systematic parameter tuning
- Cross-validation for robust model evaluation
- Performance-based model selection

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/enhancement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/enhancement`)
5. Create a Pull Request

## Future Enhancements

- **Deep Learning Integration**: BERT, LSTM, or Transformer models
- **Real-time Prediction**: Streaming data processing capabilities
- **Multi-class Classification**: Extended sentiment categories
- **API Development**: REST API for model deployment
- **Docker Integration**: Containerized deployment solution

## Acknowledgments

- **spaCy** for advanced NLP processing capabilities
- **XGBoost** for gradient boosting implementation
- **scikit-learn** for machine learning utilities
- **AWS** for cloud storage and deployment infrastructure
- **NLTK** for natural language processing tools
