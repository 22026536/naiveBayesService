from fastapi import FastAPI, Request
from pymongo import MongoClient
import pandas as pd
import numpy as np
from bson import ObjectId

# Class NaiveBayesClassifier
class NaiveBayesClassifier:
    def __init__(self):
        self.class_probs = {}  
        self.feature_probs = {}  

    def fit(self, X, y):
        unique_classes, class_counts = np.unique(y, return_counts=True)
        total_samples = len(y)
        for cls, count in zip(unique_classes, class_counts):
            self.class_probs[cls] = count / total_samples

        self.feature_probs = {cls: {} for cls in unique_classes}
        for cls in unique_classes:
            X_cls = X[y == cls]  
            for feature in X.columns:
                value_counts = X_cls[feature].value_counts().to_dict()
                total_feature_count = len(X_cls)
                feature_prob = {
                    val: (count + 1) / (total_feature_count + len(X[feature].unique()))
                    for val, count in value_counts.items()
                }
                self.feature_probs[cls][feature] = feature_prob

    def predict(self, X):
        predictions = []
        for _, row in X.iterrows():
            class_scores = {}
            for cls in self.class_probs:
                log_prob = np.log(self.class_probs[cls])
                for feature, value in row.items():
                    if value in self.feature_probs[cls][feature]:
                        log_prob += np.log(self.feature_probs[cls][feature].get(value, 1 / (len(X) + len(X[feature].unique()))))
                    else:
                        log_prob += np.log(1 / (len(X) + len(X[feature].unique())))
                class_scores[cls] = log_prob
            predictions.append(int(max(class_scores, key=class_scores.get)))
        return predictions


app = FastAPI()

# Middleware CORS
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://anime-fawn-five.vercel.app"],  # Allow specific origins
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB connection
client = MongoClient('mongodb+srv://sangvo22026526:5anG15122003@cluster0.rcd65hj.mongodb.net/anime_tango2')  
db = client['anime_tango2']  
anime_collection = db['Anime']
user_rating_collection = db['UserRating']

# Helper function to convert MongoDB ObjectId to string
def serialize_object_id(obj):
    if isinstance(obj, ObjectId):
        return str(obj)
    return obj


# Convert MongoDB document to dictionary, handling ObjectId
def get_anime_data():
    anime_data = list(anime_collection.find())
    for item in anime_data:
        item['_id'] = str(item['_id'])  # Convert ObjectId to string
    return pd.DataFrame(anime_data)

# Get user ratings
def get_user_ratings(user_id):
    user_ratings = list(user_rating_collection.find({'User_id': user_id}))
    for item in user_ratings:
        item['_id'] = str(item['_id'])  # Convert ObjectId to string
    return pd.DataFrame(user_ratings)

# Process anime genres and other features
genres = ['Action', 'Adventure','Avant Garde','Award Winning','Ecchi','Girls Love','Mystery','Sports','Supernatural','Suspense', 'Sci-Fi', 'Comedy', 'Drama', 'Romance', 'Horror', 'Fantasy', 'Slice of Life']

anime_df = get_anime_data()

anime_df['Score_'] = anime_df['Score'].apply(lambda x: 0 if x < 8 else (1 if x <= 9 else 2))  # Classify score into 3 categories

# Categorize Favorites
def categorize_favorites(favorites_count):
    if favorites_count <= 5000:
        return 0  # Low
    elif favorites_count <= 20000:
        return 1  # Medium
    else:
        return 2  # High

anime_df['Favorites_'] = anime_df['Favorites'].apply(categorize_favorites)

# Categorize Japanese Level
def categorize_japanese_level(level):
    if level in ['N4', 'N5']:  # Beginner levels
        return 0
    elif level in ['N2', 'N3']:  # Intermediate levels
        return 1
    else:
        return 2  # Advanced

anime_df['JapaneseLevel_'] = anime_df['JapaneseLevel'].apply(categorize_japanese_level)

# Categorize Age
def categorize_age(age_str):
    if '7+' in age_str:
        return 0  # Age 7+
    elif '13+' in age_str:
        return 1  # Age 13+
    elif '16+' in age_str:
        return 2  # Age 16+
    elif '17+' in age_str:
        return 3  # Age 17+
    elif '18+' in age_str:
        return 4  # Age 18+
    else:
        return 0  # No specific age category

anime_df['AgeCategory'] = anime_df['Old'].apply(categorize_age)

# One-hot encoding of genres
for genre in genres:
    anime_df[genre] = anime_df['Genres'].apply(lambda x: 1 if genre in x else 0)

# Train Naive Bayes for a user
def train_naive_bayes(user_id):
    user_ratings = get_user_ratings(user_id)
    rated_animes = user_ratings['Anime_id'].tolist()

    user_animes = anime_df[anime_df['Anime_id'].isin(rated_animes)]
    X = user_animes[genres + ['Favorites_','JapaneseLevel_','AgeCategory']]
    y = user_animes['Score_']

    clf = NaiveBayesClassifier()
    clf.fit(X, y)

    return clf

@app.post('/naivebayes')
async def recommend_anime(request: Request):
    data = await request.json()
    user_id = data.get("user_id")
    n = data.get("n", 10)

    clf = train_naive_bayes(user_id)

    anime_features = anime_df[genres + ['Favorites_','JapaneseLevel_','AgeCategory']]
    predictions = clf.predict(anime_features)

    recommended_anime_indices = np.where(np.array(predictions) >= 1)[0]
    recommended_anime = anime_df.iloc[recommended_anime_indices]

    user_ratings = get_user_ratings(user_id)
    rated_anime_ids = user_ratings['Anime_id'].tolist()
    recommended_anime = recommended_anime[~recommended_anime['Anime_id'].isin(rated_anime_ids)]
    recommended_anime['Anime_id'] = recommended_anime['Anime_id'].astype(int)
    recommended_anime['Score'] = recommended_anime['Score'].astype(int)

    # Ensure response is serializable
    recommended_anime = recommended_anime.applymap(lambda x: int(x) if isinstance(x, np.int64) else x)

    return recommended_anime.head(n)

import uvicorn
import os
if __name__ == "__main__":
    port = int(os.getenv("PORT", 4001))
    uvicorn.run("naivebayes:app", host="0.0.0.0", port=port)
