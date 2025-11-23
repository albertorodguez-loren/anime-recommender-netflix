import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
import requests
import random

# Título épico
st.set_page_config(page_title="Anime Recommender Netflix-Style", layout="centered")
st.title("Anime Recommender Netflix-Style")
st.markdown("### By Alberto Rodríguez Loren – 2025 Portfolio")

# Carga de datos (una sola vez)
@st.cache_data
def load_data():
    animes = pd.read_csv('anime-dataset-2023.csv')
    ratings = pd.read_csv('users-score-2023.csv', nrows=500_000)
    return animes, ratings

@st.cache_resource
def train_model(ratings):
    user_item = ratings.pivot_table(index='user_id', columns='anime_id', values='rating', fill_value=0)
    sparse = csr_matrix(user_item.values)
    svd = TruncatedSVD(n_components=50, random_state=42)
    svd.fit(sparse)
    user_factors = svd.transform(sparse)
    pred_matrix = np.dot(user_factors, svd.components_)
    pred_df = pd.DataFrame(pred_matrix, index=user_item.index, columns=user_item.columns)
    return pred_df, user_item

animes, ratings = load_data()
pred_df, user_item = train_model(ratings)

# Sidebar
st.sidebar.header("Elige tu usuario")
all_users = ratings['user_id'].unique().tolist()
user_id = st.sidebar.selectbox("Usuario ID", options=[None] + all_users, format_func=lambda x: "Aleatorio" if x is None else x)

if user_id is None:
    user_id = random.choice(all_users)

vistos = ratings[ratings['user_id'] == user_id]['anime_id'].tolist()
recs = pred_df.loc[user_id].drop(vistos, errors='ignore').sort_values(ascending=False).head(12)
results = recs.reset_index()
results.columns = ['anime_id', 'predicted_rating']
results = results.merge(animes[['anime_id', 'Name', 'Genres', 'Type', 'Score', 'Image URL']], on='anime_id')
results['predicted_rating'] = results['predicted_rating'].round(2)

st.write(f"### Recomendaciones TOP 12 para el usuario **{user_id}**")

cols = st.columns(3)
for idx, row in results.iterrows():
    with cols[idx % 3]:
        st.image(row['Image URL'], use_column_width=True)
        st.caption(f"**{row['Name']}**")
        st.write(f"Predicción: **{row['predicted_rating']}** | MAL Score: {row['Score']}")
        st.write(f"*{row['Genres']} – {row['Type']}*")
        st.markdown("---")
