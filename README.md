This project is a language detection model that can identify 31 different languages from a given text. It was developed as a part of a Machine Learning course at the University of Mumbai. The project includes data collection, preprocessing, and the training and evaluation of various machine learning models.

<br>

<br>

⚙️ Project Stages

    Data Collection and Preparation: I collected a diverse dataset of texts across 31 languages, and stored this data in unicode format to avoid data loss due to encoding errors of special characters.

    Data Cleaning and Preprocessing: The text data was cleaned by removing unnecessary characters and converting it into a consistent format suitable for model training. I used CountVectorizer to transform the cleaned text into a numerical matrix of token counts.

    Exploratory Data Analysis (EDA): The dataset was analyzed to understand its characteristics, including word frequencies and text distribution across languages.

    Model Training and Evaluation: I trained and evaluated six different machine learning models to find the most accurate one for the task.

<br>

<br>

🛠️ Models Used

The following models were implemented and evaluated based on their accuracy:
| Model Name | Training Accuracy | Testing Accuracy |
| :--- | :---: | :---: |
| K-Nearest Neighbours | 52.0952 | 41.0195 |
| Decision Tree Classifier | 99.6513 | 82.3034 |
| Random Forest | 99.6513 | 88.6603 |
| Logistic Regression | 97.4208 | 90.4063 |
| Multinomial Naive Bayes | 95.9779 | 92.955 |
| Linear SVC | 99.5371 | 93.0753 |
