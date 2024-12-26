from fastapi import FastAPI, Request
from pymongo import MongoClient
import pandas as pd
import numpy as np

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
                        log_prob += np.log(self.feature_probs[cls][feature][value])
                    else:
                        log_prob += np.log(1 / (len(X) + len(X[feature].unique())))
                class_scores[cls] = log_prob
            predictions.append(max(class_scores, key=class_scores.get))
        return predictions


app = FastAPI()

# Middleware CORS
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://anime-fawn-five.vercel.app"],  # Cho phép tất cả origin
    allow_methods=["*"],
    allow_headers=["*"],
)

# Kết nối MongoDB
client = MongoClient('mongodb+srv://<your_connection_string>')  
db = client['anime_tango2']  
anime_collection = db['Anime']
user_rating_collection = db['UserRating']

# Hàm lấy dữ liệu Anime
def get_anime_data():
    anime_data = list(anime_collection.find())
    return pd.DataFrame(anime_data)

# Hàm lấy dữ liệu UserRatings
def get_user_ratings(user_id):
    user_ratings = list(user_rating_collection.find({'User_id': user_id}))
    return pd.DataFrame(user_ratings)

# Chuyển đổi bảng 'Genres' thành các cột nhị phân
genres = ['Action', 'Adventure','Avant Garde','Award Winning','Ecchi','Girls Love','Mystery','Sports','Supernatural','Suspense', 'Sci-Fi', 'Comedy', 'Drama', 'Romance', 'Horror', 'Fantasy', 'Slice of Life']

anime_df = get_anime_data()

# Cập nhật Anime DataFrame
anime_df['Score_'] = anime_df['Score'].apply(lambda x: 0 if x < 8 else (1 if x <= 9 else 2))  # Phân loại score thành 3 loại

# Chuyển Genres thành các cột nhị phân (one-hot encoding)
for genre in genres:
    anime_df[genre] = anime_df['Genres'].apply(lambda x: 1 if genre in x else 0)

# Hàm train Naive Bayes cho mỗi user
def train_naive_bayes(user_id):
    # Lấy dữ liệu từ người dùng
    user_ratings = get_user_ratings(user_id)
    rated_animes = user_ratings['Anime_id'].tolist()

    # Lọc anime đã được người dùng đánh giá
    user_animes = anime_df[anime_df['Anime_id'].isin(rated_animes)]

    # Sử dụng Genres + Các đặc trưng khác làm đặc trưng cho mô hình
    X = user_animes[genres + ['Favorites_','JapaneseLevel_','AgeCategory']]
    y = user_animes['Score_']  # Nhãn để dự đoán

    # Huấn luyện mô hình Naive Bayes
    clf = NaiveBayesClassifier()
    clf.fit(X, y)

    return clf

@app.post('/naivebayes')
async def recommend_anime(request: Request):
    data = await request.json()
    user_id = data.get("user_id")
    n = data.get("n", 10)  # Số lượng gợi ý, mặc định là 10

    clf = train_naive_bayes(user_id)

    # Dự đoán các anime còn lại
    anime_features = anime_df[genres + ['Favorites_','JapaneseLevel_','AgeCategory']]
    predictions = clf.predict(anime_features)

    # Lọc các anime được dự đoán thích (Score >= 1)
    recommended_anime_indices = np.where(np.array(predictions) >= 1)[0]
    recommended_anime = anime_df.iloc[recommended_anime_indices]

    # Loại bỏ anime mà người dùng đã đánh giá
    user_ratings = get_user_ratings(user_id)
    rated_anime_ids = user_ratings['Anime_id'].tolist()
    recommended_anime = recommended_anime[~recommended_anime['Anime_id'].isin(rated_anime_ids)]

    # Trả về n anime gợi ý
    return recommended_anime.head(n)[['Anime_id', 'Name', 'Score', 'Genres', 'Synopsis']]

import uvicorn
import os
if __name__ == "__main__":
    port = int(os.getenv("PORT", 4001))
    uvicorn.run("naivebayes:app", host="0.0.0.0", port=port)
