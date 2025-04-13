# SMS-Spam-Classifier-App-using-Kivy-and-SVM

A simple Kivy-based app that classifies SMS messages as Spam or Not Spam using an SVM model with TF-IDF and custom text features like punctuation, numbers, and links.

# 📱 SMS Spam Classifier App

A simple desktop application built using **Kivy** that classifies SMS messages as **Spam** or **Not Spam**. It uses a pre-trained **SVM model** with TF-IDF vectorization and additional custom features like punctuation, numbers, links, and unique word count.

---

## 🚀 Features

- 🧠 Predicts whether an SMS is spam or not
- 🔤 Preprocesses input (cleaning, stopword removal, stemming)
- 📊 Uses TF-IDF + custom features for classification
- 🖥️ Clean and simple Kivy-based UI
- ✅ Real-time prediction on button click

---

## 🛠️ Tech Stack

- Python 3
- Kivy
- Scikit-learn
- NLTK
- NumPy
- Pickle

---

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/sms-spam-classifier.git
   cd sms-spam-classifier


2.  Install dependencies:
    
    bash
    
    CopyEdit
    
    `pip install -r requirements.txt` 
    
3.  Download NLTK stopwords:
    
    python
    
    CopyEdit
    
    `import nltk
    nltk.download('stopwords')` 
    
4.  Make sure the following files are present:
    
    -   `vectorizer2.pkl`
        
    -   `svm2.pkl`
        

----------

## ▶️ Run the App

bash

CopyEdit

`python app.py` 

----------

## 📁 File Structure

bash

CopyEdit

`sms-spam-classifier/
│
├── app.py # Main Kivy application ├── vectorizer2.pkl # Saved TF-IDF vectorizer ├── svm2.pkl # Trained SVM model ├── README.md
└── requirements.txt # Python dependencies` 

----------

## 🧪 Model Info

-   Trained using TF-IDF vectorization
    
-   Additional features:
    
    -   Character count
        
    -   Punctuation count
        
    -   Presence of numbers and links
        
    -   Unique word count
        

----------

## 📜 License

This project is licensed under the MIT License.

----------

## 🤝 Contributing

Feel free to fork this repo and submit a pull request for improvements or new features!
