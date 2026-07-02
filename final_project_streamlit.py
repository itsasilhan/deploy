import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("IMDb_Top_250.csv")
    df['Genres_List'] = df['Genres'].apply(lambda x: [genre.strip() for genre in x.split(',')] if isinstance(x, str) else [])
    return df

df = load_data()

# Encode genres
mlb = MultiLabelBinarizer()
genres_encoded = mlb.fit_transform(df['Genres_List'])
genre_similarity = cosine_similarity(genres_encoded, genres_encoded)

# Format duration
def format_duration(duration):
    try:
        duration = int(duration)    
        hours = duration // 60
        minutes = duration % 60
        return f"{hours} h {minutes} min"
    except:
        return "Not specified"

# Format genres into HTML
def format_genres(genres):
    genre_labels = ""
    for genre in genres.split(","):
        genre_labels += f"""
        <span style="
            display: inline-block;
            background-color: #FFC107;
            color: black;
            font-size: 12px;
            font-weight: bold;
            border-radius: 5px;
            padding: 5px 10px;
            margin-right: 5px;
            margin-bottom: 5px;
        ">{genre.strip()}</span>
        """
    return genre_labels

# Recommend movies based on genres
def recommend_movies_by_genres(movie_title, num_recommendations=5):
    if movie_title not in df['Title'].values:
        return None

    # Get the index of the selected movie
    idx = df[df['Title'] == movie_title].index[0]

    # Calculate genre similarity
    sim_scores = list(enumerate(genre_similarity[idx]))

    # Prioritize recommendations based on title match
    def prioritize_title(sim_score):
        index, similarity = sim_score
        other_title = df.iloc[index]['Title']
        matching_words = len(set(movie_title.split()).intersection(set(other_title.split())))
        return (matching_words >= 2, similarity)

    # Sort recommendations by title priority and similarity
    sim_scores = sorted(sim_scores, key=prioritize_title, reverse=True)

    # Select top-N recommendations (excluding the selected movie)
    sim_scores = [score for score in sim_scores if score[0] != idx][:num_recommendations]
    movie_indices = [i[0] for i in sim_scores]

    return df.iloc[movie_indices][['Title', 'Genres', 'Rating (Numeric)', 'Rating (Label)', 'Duration in minutes', 'Description', 'Poster Link', 'Link']]

# Streamlit Interface
st.title("Interactive IMDb Dashboard")

# Step 1: Select a movie
selected_movie = st.selectbox("Choose a movie:", df['Title'].unique())

# Step 2: Display information about the selected movie
if selected_movie:
    movie_info = df[df['Title'] == selected_movie].iloc[0]
    
    # Create two columns for the poster and movie details
    col1, col2 = st.columns([1, 2])  # 1 part for the poster, 2 parts for the details
    
    with col1:
        st.image(movie_info['Poster Link'], use_container_width=True)
    
    with col2:
        st.subheader(movie_info['Title'])
        st.markdown(format_genres(movie_info['Genres']), unsafe_allow_html=True)
        st.markdown(
            f"<p style='color: #FFC107; display: inline;'><b>Rating (Numeric):</b></p> <span style='color: white;'>{movie_info['Rating (Numeric)']} ⭐</span>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<p style='color: #FFC107; display: inline;'><b>Rating (Label):</b></p> <span style='color: white;'>{movie_info['Rating (Label)']}</span>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<p style='color: #FFC107; display: inline;'><b>Duration:</b></p> <span style='color: white;'>{format_duration(movie_info['Duration in minutes'])}</span>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<p style='color: #FFC107; display: inline;'><b>Description:</b></p> <span style='color: white;'>{movie_info['Description']}</span>",
            unsafe_allow_html=True,
        )
        if pd.notna(movie_info['Link']):
            st.write(f"[Watch on IMDb]({movie_info['Link']})")
    st.markdown("---")

    # Step 3: Recommendations
    st.subheader("Recommended Movies")
    recommendations = recommend_movies_by_genres(selected_movie, num_recommendations=5)

    if recommendations is not None:
        for _, rec in recommendations.iterrows():
            with st.expander(rec['Title']):  
                # Create two columns for the poster and information
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.image(rec['Poster Link'], use_container_width=True)
                
                with col2:
                    st.markdown(format_genres(rec['Genres']), unsafe_allow_html=True)
                    st.markdown(
                        f"<p style='color: #FFC107; display: inline;'><b>Rating (Numeric):</b></p> <span style='color: white;'>{rec['Rating (Numeric)']} ⭐</span>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<p style='color: #FFC107; display: inline;'><b>Rating (Label):</b></p> <span style='color: white;'>{rec['Rating (Label)']}</span>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<p style='color: #FFC107; display: inline;'><b>Duration:</b></p> <span style='color: white;'>{format_duration(rec['Duration in minutes'])}</span>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<p style='color: #FFC107; display: inline;'><b>Description:</b></p> <span style='color: white;'>{rec['Description']}</span>",
                        unsafe_allow_html=True,
                    )
                    if pd.notna(rec['Link']):
                        st.markdown(f"[More details on IMDb]({rec['Link']})", unsafe_allow_html=True)
    else:
        st.write("Recommendations are unavailable. Please try another movie.")
