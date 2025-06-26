# Project Details: Movie Recommendation System

## 📌 Overview
This project is a Movie Recommendation System built using Python. It uses a dataset (`movies3.csv`) and applies machine learning techniques to recommend movies to users. The application runs locally and may include a GUI (Tkinter-based) or terminal interaction.

---

## 🎯 Objectives
- Load movie metadata from CSV.
- Train a machine learning model or apply similarity-based logic.
- Recommend movies to users based on their preferences.
- Optionally integrate with a GUI for ease of use.

---

## 📁 Dataset: `movies3.csv`
This file contains information such as:
- Movie Titles
- Genres
- Ratings
- User preferences (if applicable)

The dataset is preprocessed and used to extract relevant features for recommendations.

---

## 🧠 Model Used
**Model Type:**  
Depending on the version you’re running, the model may use one or more of the following:
- **GRU (Gated Recurrent Unit):** If it’s a deep learning-based recommendation.
- **Cosine Similarity / Nearest Neighbors:** For collaborative or content-based filtering.
- **OneHotEncoder + ML Model:** Encodes genre/title as inputs for ML.

**Steps:**
1. Data cleaning and encoding (OneHotEncoder or LabelEncoder).
2. Train/test split (optional).
3. Use a recommender logic (content similarity, deep learning model, or pre-trained weights).
4. Output top recommended movies.

---

## 🖼️ GUI (If included)
- Tkinter-based interface.
- Allows users to select a movie or genre.
- Displays recommended movies dynamically.

---

## ⚙️ Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Tkinter (for GUI)
- [Optional] TensorFlow/Keras (for GRU or deep models)

---

## 📝 How to Extend
- Replace `movies3.csv` with a larger dataset.
- Deploy using Flask or Streamlit as a web app.
- Integrate with a user login system to track preferences.

---

## 👨‍💻 Author
K. Narsimhulu  
B.Tech (CSE – AI & ML)  
Passionate about Machine Learning and Software Development.
