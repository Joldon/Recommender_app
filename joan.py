# other recommenders lirbaries
# https://surpriselib.com/
# https://github.com/JoanClaverol/streamlit_deploy_example
import streamlit as st

# import data 
import pandas as pd

ratings = pd.read_csv('ml-latest-small/ratings.csv')
movies = pd.read_csv('ml-latest-small/movies.csv')

# print on terminal if the data was loaded successfully
if not ratings.empty and not movies.empty:
    print('Data loaded successfully!')

# what should have our app?
# 1. a title
st.title('Movie Recommender System')
# 2. a sidebar with options
st.sidebar.title('Options')
# 2.1. input filed to select a movie
sel_movie = st.sidebar.selectbox('Select a movie', 
                     movies['title'].unique().tolist())
# 2.2. a slider to select the number of recommendations
n_movies = st.sidebar.number_input('Number of recommendations',
                        min_value=1,
                        max_value=10,
                        value=5)

# 3. a main area where we show the results
# 3.1. Render the movie title
top_n = (
ratings
    .merge(movies, on='movieId', how='left')
    .assign(moviesId_title = lambda x: x['movieId'].astype(str) + ' - ' + x['title'])
    .groupby('moviesId_title')
    .agg({'rating': 'mean'})
    .reset_index()
    .loc[lambda x: ~x['moviesId_title'].str.contains(sel_movie, regex=False)]
    .sort_values(
        ['rating', 'moviesId_title'], 
        ascending=[False, False]
        )
    .head(int(n_movies))
    )
st.header("Top Movies")
st.dataframe(top_n)

print("App running!")