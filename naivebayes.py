from fastapi import FastAPI, Request
from pymongo import MongoClient
import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np

# Khởi tạo app
app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

# Thêm middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://animetangobackend.onrender.com"],  # Cho phép tất cả origin
    allow_methods=["*"],
    allow_headers=["*"],
)

# Kết nối đến MongoDB
client = MongoClient('mongodb+srv://sangvo22026526:5anG15122003@cluster0.rcd65hj.mongodb.net/anime_tango2')  # Sử dụng URL kết nối MongoDB của bạn
db = client['anime_tango2']  # Tên cơ sở dữ liệu của bạn
anime_collection = db['Anime']
user_rating_collection = db['UserRating']

# Hàm lấy dữ liệu Anime
def get_anime_data():
    anime_data = list(anime_collection.find())
    return pd.DataFrame(anime_data)

# Hàm lấy dữ liệu UserRatings (thay vì UserFavorites)
def get_user_ratings(user_id):
    user_ratings = list(user_rating_collection.find({'User_id': user_id}))
    return user_ratings

# Lấy dữ liệu Anime
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

# Cập nhật phần 'Is_13_plus' và thêm các độ tuổi khác
def categorize_age(age_str):
    if '7+' in age_str:
        return 0  # Các anime có độ tuổi 13+
    elif '13+' in age_str:
        return 1  # Các anime có độ tuổi 13+
    elif '16+' in age_str:
        return 2  # Các anime có độ tuổi 16+
    elif '17+' in age_str:
        return 3  # Các anime có độ tuổi 17+
    elif '18+' in age_str:
        return 4  # Các anime có độ tuổi 18+
    else:
        return 0  # Các anime không có độ tuổi

anime_df['AgeCategory'] = anime_df['Old'].apply(categorize_age)

# Hàm lấy đặc trưng từ lịch sử đánh giá của người dùng
def get_user_features(user_id):
    user_ratings = get_user_ratings(user_id)
    user_ratings_df = pd.DataFrame(user_ratings)
    user_anime_df = anime_df[anime_df['Anime_id'].isin(user_ratings_df['Anime_id'])]
    features = {}

    # Tính toán các đặc trưng
    features['Avg_Old'] = user_anime_df['AgeCategory'].apply(lambda x: 1 if x == 1 else 0).mean()
    features['Avg_Favorites'] = user_anime_df['Favorites_'].mean()
    features['Avg_JapaneseLevel'] = user_anime_df['JapaneseLevel_'].mean()
    features['Avg_Score'] = user_anime_df['Score_'].mean()

    for genre in genres:
        features[f'Avg_{genre}'] = user_anime_df[genre].mean()

    return features

from sklearn.naive_bayes import MultinomialNB  # Hoặc GaussianNB nếu phù hợp
def train_naive_bayes(user_id):
    # Lấy dữ liệu gợi ý
    user_features = get_user_features(user_id)

    # Tạo một dataframe các đặc trưng người dùng cho tất cả anime
    anime_features = anime_df[genres + ['Favorites_', 'JapaneseLevel_', 'AgeCategory', 'Score_']]  # Đảm bảo sử dụng Score_

    # Chuẩn bị dữ liệu cho người dùng (một dòng đặc trưng)
    user_feature_vector = np.array([user_features[f'Avg_{genre}'] for genre in genres] +
                                   [user_features['Avg_Favorites'], user_features['Avg_JapaneseLevel'],
                                    user_features['Avg_Old'], user_features['Avg_Score']])  # Thêm 'Avg_Score'

    # Dữ liệu huấn luyện (anime_features)
    X = anime_features
    y = anime_df['Score_']  # Dùng Score_ thay vì Score


    # Tạo mô hình Naive Bayes
    clf = MultinomialNB()
    clf.fit(X, y)

    return clf

@app.post('/naivebayes')
async def recommend_anime(request: Request):
    data = await request.json()
    user_id = data.get("user_id")
    n = data.get("n", 10)  # Số lượng gợi ý, mặc định là 10
    clf = train_naive_bayes(user_id)
    anime_features = anime_df[genres + ['Favorites_', 'JapaneseLevel_', 'AgeCategory', 'Score_']]
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
    port = int(os.getenv("PORT", 4001))  # Render sẽ cung cấp cổng trong biến PORT.
    uvicorn.run("naivebayes:app", host="0.0.0.0", port=port)
