import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
#%matplotlib inline

st.title("Recommendation System based on IMDB user ratings")

st.sidebar.header("User Input")
uploaded_file = st.sidebar.file_uploader("Upload IMDB Ratings (CSV format)", type=["csv"])

#->>>>> DRIVE DOESNT ALLOW THAT MANY CALLS, import in local
# #--- IMPORT DATASETS (i had to upload them to my google drive)
# # links.csv
# url = 'https://drive.google.com/file/d/1HT07oBreEpcsZypvnh8r2Ll_TBQuL99-/view?usp=sharing' 
# path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
# links = pd.read_csv(path)

# # movies.csv
# url = 'https://drive.google.com/file/d/1alTHupTjqzEQsmnbLRWFBb6lShtQzjS_/view?usp=sharing' 
# path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
# movies = pd.read_csv(path)

# # ratings.csv
# url = 'https://drive.google.com/file/d/1JApul2ah2godYurqOTAYK4BBSkcj66Tf/view?usp=sharing' 
# path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
# ratings = pd.read_csv(path)
# pd.set_option('display.max_colwidth', 300)

links = pd.read_csv('ml-latest-small/links.csv')
movies = pd.read_csv('ml-latest-small/movies.csv')
ratings = pd.read_csv('ml-latest-small/ratings.csv')




def prepare_data_indie(uploaded_file, n):
    #-------TRANSFORMING THE IMDB USER DATABASE TO INCLUDE IT IN THE LMS DATABASE
    imdb = uploaded_file #pd.read_csv(uploaded_file) #uploaded file is the imdb user database, transformed into pandas dataframe
    
    # add tt0 at the beginning of each row in the column imdbld (this is the 2023 format of IMDB Urls)
    links['imdbId'] = 'tt0' + links['imdbId'].astype(str) 

    #merge ratings, links and movies
    ratings_links = ratings.merge(links, left_on='movieId', right_on='movieId')
    ratings_links = ratings_links.merge(movies, left_on='movieId', right_on='movieId')

    #Manual bug fixing, add 0 to the "broken" imdbId

    #in ratings_links add a 0 after tt in imdbId to the rows where title contains "Star" or "Godfather" or "Indiana"
    ratings_links.loc[ratings_links['title'].str.contains('Star|Godfather|Indiana'), 'imdbId'] = ratings_links['imdbId'].str[:2] + '0' + ratings_links['imdbId'].str[2:]

    #save all the imdbId in a list, to iterate later
    imdbId = ratings_links["imdbId"].tolist()

    #check with movies are in both datasets
    imdb_user = imdb.loc[imdb["Const"].isin(imdbId), :]

    #select columns that are useful in the imdb dataset
    imdb_user = imdb_user[["Const", "Title", "Genres", "Your Rating"]]

    # rename columns
    imdb_user.columns = ["imdbId", "title", "genres", "rating"]

    # convert "rating" into a 5-star rating format
    imdb_user["rating"] = imdb_user["rating"] / 2

    # check whats the last user id and add 1
    imdb_user["userId"] = ratings_links["userId"].max() + 1


    #saving the userID to ireate later
    my_userID = ratings_links["userId"].max() + 1

    #initialize the columnn; otherwise will give an error
    imdb_user["movieId"] = 0
    #for each movie in imdb_user, find the movieId in ratings_links
    for i in range(len(imdb_user)):
        imdb_user.iloc[i, 5] = ratings_links.loc[ratings_links["imdbId"]==imdb_user.iloc[i, 0], "movieId"].values[0]
    # add a new user to the ratings dataframe
    ratings_user = ratings_links.append(imdb_user, ignore_index=True)

    #------------------
    #START OF THE RECOMMENDED SYSTEM
    users_items = pd.pivot_table(data=ratings_user,
                                values="rating",
                                index="userId",
                                columns="movieId")

    #Filling Na with 0
    users_items.fillna(0, inplace=True)

    #Compute cosine similarities
    user_similarities = pd.DataFrame(cosine_similarity(users_items),
                                    columns=users_items.index, 
                                    index=users_items.index)

    user_id = my_userID

    weights = (
        user_similarities.query("userId!=@user_id")[user_id] / sum(user_similarities.query("userId!=@user_id")[user_id])
            )

    # select movies that the inputed user has not watched/rated
    not_watched_movies = users_items.loc[users_items.index!=user_id, users_items.loc[user_id,:]==0]



    # dot product between the not-visited-restaurants and the weights
    weighted_averages = pd.DataFrame(not_watched_movies.T.dot(weights), columns=["predicted_rating"])

    recommendations = weighted_averages.merge(movies, left_index=True, right_on="movieId")

    #merge with links
    recommendations = recommendations.merge(links, left_on='movieId', right_on='movieId')
    # en vez de las 10 primeras, muestrame del 5 al 15
    #preguntar al usuario si quiere ver las 20 recomendaciones "Mainstream" o "No Mainstream"
    #st.write("##### Do you want to see mainstream or indie movies?")
    st.write("##### Mainstream movies are the ones that have been rated by more than 50 users")
    st.write("##### Indie movies are the ones that have been rated between 20 and 30 users")
    # st.write("##### If you want to see mainstream movies, type 'M'")
    # st.write("##### If you want to see indie movies, type 'I'")
    mainstream_or_indie = st.text_input("Mainstream or Indie?")
   
    
    if mainstream_or_indie == "M":
        return list(recommendations.sort_values("predicted_rating", ascending=False).head(n)["title"])
    elif mainstream_or_indie == "I":
        ratings_indie = ratings.groupby("movieId").count()
        indie_movies = ratings_indie.loc[ratings_indie["userId"].between(15, 30) , :].index.tolist()
        recommendations_indies = recommendations.loc[recommendations["movieId"].isin(indie_movies), :]
        return list(recommendations_indies.sort_values("predicted_rating", ascending=False).head(n)["title"])
    else:
        st.write("### Please type 'M' or 'I'")


    #  # add movie length selection
    # st.write("##### Select movie length:")
    # length_options = ["Short (less than 90 mins)", "Medium (between 90 and 120 mins)", "Lengthy (more than 120 mins)"]
    # selected_length = st.selectbox("", length_options)

    # if selected_length == length_options[0]:
    #     recommendations = recommendations.loc[recommendations["Runtime(mins)"] < 90]
    # elif selected_length == length_options[1]:
    #     recommendations = recommendations.loc[recommendations["Runtime(mins)"].between(90, 120)]
    # elif selected_length == length_options[2]:
    #     recommendations = recommendations.loc[recommendations["Runtime(mins)"] > 120]
    # else:
    #     st.write("### Please select a movie length option")
    #     return None
    
    #     # Get user preference for movie length
    # length_preference = st.selectbox("What length of movie do you prefer?", ["Short", "Medium", "Long"])

    # if length_preference == "Short":
    #     not_watched_movies = not_watched_movies.loc[not_watched_movies["Runtime(mins)"] <= 90, :]
    # elif length_preference == "Medium":
    #     not_watched_movies = not_watched_movies.loc[(not_watched_movies["Runtime(mins)"] > 90) & (not_watched_movies["Runtime(mins)"] <= 120), :]
    # elif length_preference == "Long":
    #     not_watched_movies = not_watched_movies.loc[not_watched_movies["Runtime(mins)"] > 120, :]
    # else:
    #     st.write("Please select a valid option")


if uploaded_file:
    
    st.write("### Your IMDB genre stats")
    #formating genres to choose only the first one
    imdb = pd.read_csv(uploaded_file)
    imdb['Genres'] = imdb['Genres'].str.split(',').str[0]
    #creating a histogram with the genres of the imdb_user dataset
    fig = plt.figure(figsize=(12, 6))
    # ordenar por numero de peliculas
    sns.countplot(x="Genres", data=imdb, palette="mako", order=imdb['Genres'].value_counts().index)
    #plt.xticks(rotation=90)
    plt.ylabel("Number of movies")
    st.pyplot(fig)

    st.write("### Your IMDB Director stats")
    #creating a horizontal bar chart with the directors of the imdb 
    fig2 = plt.figure(figsize=(12, 6))
    # ordenar por numero de peliculas
    sns.countplot(y="Directors", data=imdb, palette="magma", order=imdb['Directors'].value_counts().iloc[:10].index)
    plt.xlabel("Number of movies")
    st.pyplot(fig2)
    st.write("### Your top Recommendations")
    #ask for number of recommendations
    n = st.number_input('How many recommendations do you want?', min_value=1, max_value=100, value=10, step=1)
    st.table(prepare_data_indie(imdb, n))


    
else:
    st.write("Please upload a dataset.")