from fastapi import FastAPI, Request
from pymongo import MongoClient
import pandas as pd
import numpy as np
from bson import ObjectId

from fastapi import FastAPI, Request
from pymongo import MongoClient
import pandas as pd
import numpy as np
from bson import ObjectId
from fastapi.middleware.cors import CORSMiddleware

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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

client = MongoClient('mongodb+srv://sangvo22026526:5anG15122003@cluster0.rcd65hj.mongodb.net/anime_tango2')  
db = client['anime_tango2']  
anime_collection = db['Anime']
user_rating_collection = db['UserRating']

def serialize_object_id(obj):
    if isinstance(obj, ObjectId):
        return str(obj)
    return obj

def get_anime_data():
    anime_data = list(anime_collection.find())
    return pd.DataFrame(anime_data)

def get_user_ratings(user_id):
    user_ratings = list(user_rating_collection.find({'User_id': user_id}))
    return pd.DataFrame(user_ratings)

anime_df = get_anime_data()
anime_df2 = anime_df

# Cập nhật để phân loại cột 'Score' theo các điều kiện
def categorize_score(score):
    if score < 8:
        return 0  # Loại 0: Score < 8
    elif 8 <= score <= 9:
        return 1  # Loại 1: 8 <= Score <= 9
    else:
        return 2  # Loại 2: Score >= 9

# Thêm cột 'Type' dựa trên cột 'Score'
anime_df['Score_'] = anime_df['Score'].apply(categorize_score)

# Chuyển Genres thành các cột nhị phân (one-hot encoding)
genres = ['Action', 'Adventure','Avant Garde','Award Winning','Ecchi','Girls Love','Mystery','Sports','Supernatural','Suspense', 'Sci-Fi', 'Comedy', 'Drama', 'Romance', 'Horror', 'Fantasy', 'Slice of Life']
for genre in genres:
    anime_df[genre] = anime_df['Genres'].apply(lambda x: 1 if genre in x else 0)


# Thêm cột 'Favorites' dựa trên số lượng Favorites
def categorize_favorites(favorites_count):
    if favorites_count <= 5000:
        return 0  # Thấp
    elif favorites_count <= 20000:
        return 1  # Trung bình
    else:
        return 2  # Cao

anime_df['Favorites_'] = anime_df['Favorites'].apply(categorize_favorites)

# Thêm cột 'JapaneseLevel' từ Anime
def categorize_japanese_level(level):
    if level in ['N4', 'N5']:  # Các mức độ dễ học
        return 0
    elif level in ['N2', 'N3']:  # Các mức độ dễ học
        return 1
    else :
        return 2

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

def train_naive_bayes(user_id):
    user_ratings = get_user_ratings(user_id)
    rated_anime_ids = [rating['Anime_id'] for rating in user_ratings]

    # Tạo tập dữ liệu
    anime_features = anime_df[genres + ['Favorites_', 'JapaneseLevel_', 'AgeCategory', 'Score_']]
    X = anime_features.values
    y = np.array([1 if anime_id in rated_anime_ids else 0 for anime_id in anime_df['Anime_id']])

    # Huấn luyện mô hình Naive Bayes
    clf = NaiveBayesClassifier()
    clf.fit(X, y)

    return clf


@app.post('/naivebayes')
async def recommend_anime(request: Request):
    data = await request.json()
    user_id = data.get("user_id")
    n = data.get("n", 10)

    clf = train_naive_bayes(user_id)

    anime_features = anime_df[genres + ['Favorites_','JapaneseLevel_','AgeCategory', 'Score_']]
    predictions = clf.predict(anime_features)

    recommended_anime_indices = np.where(predictions >= 1)[0]
    recommended_anime = anime_df2.iloc[recommended_anime_indices]

    user_ratings = get_user_ratings(user_id)
    rated_anime_ids = [rating['Anime_id'] for rating in user_ratings]
    recommended_anime = recommended_anime[~recommended_anime['Anime_id'].isin(rated_anime_ids)]

    recommended_anime = recommended_anime.head(n)[['Anime_id', 'Name','English name','Score', 'Genres', 'Synopsis','Type','Episodes','Duration', 'Favorites','Scored By','Members','Image URL','Old', 'JapaneseLevel']]

    return {"recommended_anime": recommended_anime.to_dict(orient="records")}

import uvicorn
import os
if __name__ == "__main__":
    port = int(os.getenv("PORT", 4001))
    uvicorn.run("naivebayes:app", host="0.0.0.0", port=port)
