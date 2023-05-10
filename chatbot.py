import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

st.title('Top Recommendation')
st.sidebar.header('User Input')


# Load the data
links = pd.read_csv('ml-latest-small/links.csv')
movies = pd.read_csv('ml-latest-small/movies.csv')
ratings = pd.read_csv('ml-latest-small/ratings.csv')
user_ratings = pd.pivot_table(data=ratings, values='rating', index='userId', columns='movieId')
user_ratings.fillna(0, inplace=True)
user_similarities = pd.DataFrame(cosine_similarity(user_ratings), columns=user_ratings.index, index=user_ratings.index)

# Define the recommend_movies function
def recommend_movies(userId, n):
    # Add code for the recommend_movies function here
    weights = (
        user_similarities.query('userId != @userId')[userId] / sum(user_similarities.query('userId != @userId')[userId])
    )
    user_unrated_movies = user_ratings.loc[user_ratings.index != userId, user_ratings.loc[userId, :] == 0]
    weighted_averages = pd.DataFrame(user_unrated_movies.T.dot(weights), columns=['predicted_rating'])
    movie_recommendations = weighted_averages.merge(movies, left_index=True, right_on='movieId').sort_values('predicted_rating', ascending=False)
    return movie_recommendations.head(n)

# def recommend_movies(userId, n):
#     # compute the weights
#     weights = (
#         user_similarities.query('userId != @userId')[userId] / sum(user_similarities.query('userId != @userId')[userId])
#     )
#     user_unrated_movies = user_ratings.loc[user_ratings.index != userId, user_ratings.loc[userId, :] == 0]
#     weighted_averages = pd.DataFrame(user_unrated_movies.T.dot(weights), columns=['predicted_rating'])
    
#     # preprocess the movie titles
#     movies['title'] = movies['title'].str.lower().str.rstrip(' the')
#     user_unrated_movies = user_unrated_movies.rename(columns=movies.set_index('movieId')['title'].to_dict())
    
#     # get the predicted ratings for unrated movies
#     weighted_averages = pd.DataFrame(user_unrated_movies.T.dot(weights), columns=['predicted_rating'])
    
#     # merge with the original movie dataframe to get movie names
#     movie_recommendations = weighted_averages.merge(movies, left_index=True, right_on='movieId').sort_values('predicted_rating', ascending=False)
    
#     # postprocess the movie names
#     movie_recommendations['name'] = movie_recommendations['title'].str.rstrip(' the')
    
#     return movie_recommendations.head(n)

#in this version the chatbot appears on the sidebar and the output appears on the main page
def chat_bot():
    st.sidebar.write("Hi! I'm your personal recommender. Tell me your userId.")
    while True:
        userId = st.sidebar.text_input("Enter your userId:", key='userId')
        try:
            userId = int(userId) # Attempt to convert input to integer
            movie_input = st.sidebar.text_input("Enter a movie you like:", key='movieId')
            movie_input = movie_input.lower().rstrip(' the')
            recom = recommend_movies(userId, 5) # Get 5 recommendations
            if recom.empty: # Check if userId is recognized by recommend_movies function
                st.write(f"Sorry, I couldn't find any recommendations for userId {userId}. Please try again.")
                continue
            st.write("Here are your top 5 movie recommendations:")
            for title in list(recom['title'])[:5]: # Display top 5 recommendations
                st.write(title)
            break
        except ValueError: # Catch the ValueError if the input cannot be converted to an integer
            st.sidebar.write("Please enter a valid userID.")
            continue

# here everyting appears in the sidebar
# def chat_bot():
#     st.write("Hi! I'm your personal recommender. Tell me your userId.")
#     while True:
#         userId = st.text_input("Enter your userId:", key='userId')
#         try:
#             userId = int(userId) # Attempt to convert input to integer
#             movie_input = st.text_input("Enter a movie you like:", key='movieId')
#             movie_input = movie_input.lower().rstrip(' the')
#             recom = recommend_movies(userId, 5) # Get 5 recommendations
#             if recom.empty: # Check if userId is recognized by recommend_movies function
#                 st.write(f"Sorry, I couldn't find any recommendations for userId {userId}. Please try again.")
#                 continue
#             # st.write(f"You will probably like the movie: {list(recom['title'])[0]}")
#             st.write("Here are your top 5 movie recommendations:")
#             for title in list(recom['title'])[:5]: # Display top 5 recommendations
#                 st.write(title)
#             break
#         except ValueError: # Catch the ValueError if the input cannot be converted to an integer
#             st.write("Please enter a valid userID.")
#             continue

# def chat_bot():
#     st.write("Hi! I'm your personal recommender. Tell me your userID.")
#     while True:
#         userId = st.text_input("Enter your userID:")
#         try:
#             userId = int(userId) # Attempt to convert input to integer
#             recom = recommend_movies(userId, 1)
#             if recom.empty: # Check if userId is recognized by recommend_movies function
#                 st.write(f"Sorry, I couldn't find any recommendations for user ID {userId}. Please try again.")
#                 continue
#             st.write(f"You will probably like the movie: {list(recom['name'])[0]}")
#             break
#         except ValueError: # Catch the ValueError if the input cannot be converted to an integer
#             st.write("Please enter a valid userID.")
#             continue

#     while True:
#         movie_input = st.text_input("Enter a movie you like:")
#         movie_input_key = f"{userId}-movie-input" # Generate unique key for movie input widget
#         if movie_input.strip() == "":
#             continue
#         else:
#             movie_title = clean_title(movie_input)
#             if movie_title not in movie_title_to_id:
#                 st.write(f"Sorry, I couldn't find any movies matching '{movie_input}'. Please try again.")
#                 continue
#             movie_id = movie_title_to_id[movie_title]
#             break
#     recom = recommend_movies(userId, 1, movie_id)
#     st.write(f"You will probably like the movie: {list(recom['name'])[0]}")


# Create the Streamlit app
def main():
    # st.title("Movie Recommendation System")
    # output in the main area
    
    # with st.sidebar:
    chat_bot()

if __name__ == '__main__':
    main()

# def main():
#     st.title("Movie Recommendation System")
#     st.write("Enter your user ID and favorite movie below to get a movie recommendation.")

#     # Input fields
#     userId = st.text_input("User ID")
#     movie_input = st.text_input("Favorite movie")

#     # Button to trigger recommendation
#     if st.button("Recommend"):
#         # Call recommendation function and display results
#         recom = recommend_movies(userId, 1)
#         st.write(f"You will probably like the movie: {list(recom['title'])[0]}")

# if __name__ == '__main__':
#     main()