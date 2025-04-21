import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set page config
st.set_page_config(
    page_title="Game Recommendation System",
    page_icon=":video_game:",
    layout="wide",
    initial_sidebar_state="expanded"
)

pages = {
    "Home": "🏠",
    "Content-Based Recommendations": "🔍",
    "Top 10 Recommendation based on User Preferences": "📈",
    "Game Correlation Finder": "🔗",
    "About": "ℹ️"
}

# Load the data for content-based recommendations
@st.cache_data
def load_content_data():
    df = pd.read_csv('all_video_games(cleaned).csv')
    df = df.dropna(subset=['Genres', 'Platforms', 'Publisher', 'User Score'])  # Drop rows with essential missing values
    df['User Score'] = df['User Score'].astype(float)  # Ensure correct data type for user score
    df['content'] = df['Genres'] + ' ' + df['Platforms'] + ' ' + df['Publisher']
    return df

# Load the data for correlation finder
@st.cache_data
def load_correlation_data():
    path = 'all_video_games(cleaned).csv'
    df = pd.read_csv(path)
    path_user = 'User_Dataset.csv'
    userset = pd.read_csv(path_user)
    data = pd.merge(df, userset, on='Title').dropna()  
    return data

df_content = load_content_data()
df_corr = load_correlation_data()

# Function to recommend games based on cosine similarity
def content_based_recommendations(game_name, num_recommendations=5):
    vectorizer = TfidfVectorizer(stop_words='english')
    content_matrix = vectorizer.fit_transform(df_content['content'])

    try:
        cosine_sim = cosine_similarity(content_matrix, content_matrix)
        idx = df_content[df_content['Title'].str.lower() == game_name.lower()].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_indices = [i[0] for i in sim_scores[1:num_recommendations+1]]
        return df_content.iloc[sim_indices][['Title', 'Genres', 'User Score', 'Platforms', 'Release Date']]
    except IndexError:
        return pd.DataFrame(columns=['Title', 'Genres', 'User Score'])

# Function to recommend games based on file upload and filters
def recommend_games(df, preferences):
    genre_filter = df['Genres'].str.contains(preferences['Genres'], case=False, na=False)
    score_filter = df['User Score'] >= preferences['Minimum User Score']
    filtered_df = df[genre_filter & score_filter]
    return filtered_df

# Create a custom sidebar menu
st.sidebar.title("Navigation")
for page, icon in pages.items():
    if st.sidebar.button(f"{icon} {page}"):
        st.session_state.page = page

# Set the default page
if 'page' not in st.session_state:
    st.session_state.page = "Home"

# Page Navigation
page = st.session_state.page

# Home Page
if page == "Home":
    st.title("🎮 Welcome to the Game Recommendation System")

    st.markdown("""
    <div style='text-align: center;'>
        <h2 style='color: #4CAF50; font-family: "Comic Sans MS", cursive; font-size: 2.5em;'>Discover Your Next Favorite Game!</h2>
        <p style='font-size: 18px; color: white; font-family: "Arial", sans-serif;'>
            Our Game Recommendation System helps you find games you’ll love based on various methods.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Section: Content-Based Recommendations
    st.markdown("""
    <div style='background-color: #f9f9f9; padding: 20px; border-radius: 8px; box-shadow: 0px 4px 8px rgba(0,0,0,0.1);'>
        <h3 style='color: #2196F3; font-family: "Verdana", sans-serif; font-size: 1.8em;'>🔍 Content-Based Recommendations</h3>
        <p style='font-size: 16px; color: black; font-family: "Georgia", serif; line-height: 1.6;'>
            Find games similar to the ones you already enjoy! Enter the title of your favorite game, and we'll recommend similar titles based on genres, platforms, and publishers.
        </p>
        <ul style='font-size: 16px; color: black; font-family: "Georgia", serif; line-height: 1.6; list-style-type: square;'>
            <li>Discover new titles that match your interests.</li>
            <li>Get personalized recommendations based on your game library.</li>
            <li>Easy to use with just a few clicks.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Section: Top 10 Recommendations based on User Preferences
    st.markdown("""
    <div style='background-color: #e8f5e9; padding: 20px; border-radius: 8px; margin-top: 20px; box-shadow: 0px 4px 8px rgba(0,0,0,0.1);'>
        <h3 style='color: #4CAF50; font-family: "Verdana", sans-serif; font-size: 1.8em;'>📈 Top 10 Recommendations based on User Preferences</h3>
        <p style='font-size: 16px; color: black; font-family: "Georgia", serif; line-height: 1.6;'>
            Upload your own game data and filter results based on your preferences. Customize your search by entering preferred genres and minimum user scores to get the top 10 recommendations tailored just for you.
        </p>
        <ul style='font-size: 16px; color: black; font-family: "Georgia", serif; line-height: 1.6; list-style-type: square;'>
            <li>Upload your latest dataset for up-to-date recommendations.</li>
            <li>Apply filters to match your taste and preferences.</li>
            <li>Download your personalized recommendations as a CSV file.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Section: Game Correlation Finder
    st.markdown("""
    <div style='background-color: #fff3e0; padding: 20px; border-radius: 8px; margin-top: 20px; box-shadow: 0px 4px 8px rgba(0,0,0,0.1);'>
        <h3 style='color: #FF5722; font-family: "Verdana", sans-serif; font-size: 1.8em;'>🔗 Game Correlation Finder</h3>
        <p style='font-size: 16px; color: black; font-family: "Georgia", serif; line-height: 1.6;'>
            Explore how games are related based on user ratings. Select a game to see its correlation with others, and find out which games share similar user reception.
        </p>
        <ul style='font-size: 16px; color: black; font-family: "Georgia", serif; line-height: 1.6; list-style-type: square;'>
            <li>Identify games with similar user scores.</li>
            <li>Understand game relationships through detailed correlations.</li>
            <li>Discover new games based on user ratings and correlations.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align: center; margin-top: 40px; font-family: "Arial", sans-serif;'>
        <h4>Use the navigation sidebar to explore different features of the app.</h4>
    </div>
    """, unsafe_allow_html=True)

# Page 1: Content-Based Recommendations
elif page == "Content-Based Recommendations":
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🎮 Game Recommendation System</h1>", unsafe_allow_html=True)
    st.markdown("<h2>Find Games Similar to Your Favorite</h2>", unsafe_allow_html=True)
    st.write("This app helps you find games similar to the ones you like. Enter the game title below to get recommendations.")

    # Add a selectbox for game selection
    game_list = df_content['Title'].unique()
    game_input = st.selectbox("Choose a game from the list:", game_list)

    # Filters within the main page
    st.subheader("Filters")
    num_recommendations = st.slider('Number of recommendations', min_value=1, max_value=10, value=5)

    # Game information display
    if game_input:
        game_info = df_content[df_content['Title'] == game_input].iloc[0]
        st.markdown(f"### Selected Game: {game_info['Title']}")
        st.write(f"Genres: {game_info['Genres']}")
        st.write(f"Platforms: {game_info['Platforms']}")
        st.write(f"Publisher: {game_info['Publisher']}")
        st.write(f"User Score: {game_info['User Score']}")
        st.write(f"Release Date: {game_info['Release Date']}")

    # Button to get recommendations
    if st.button('Get Recommendations'):
        recommendations = content_based_recommendations(game_input, num_recommendations)
        if not recommendations.empty:
            st.markdown(f"### Games similar to {game_input}:")
            st.table(recommendations)
        else:
            st.write("No matching game found. Please try another.")

elif page == "Top 10 Recommendation based on User Preferences":
    st.title("🎮 Personalized Game Recommendations")
    st.markdown("""
    <div style='background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 20px;'>
        <h3 style='color: #2e7d32;'>How to use this feature:</h3>
        <ol style='line-height: 1.6;'>
            <li>Upload your game dataset in CSV format (ensure it contains 'Genres' and 'User Score' columns)</li>
            <li>Set your preferences using the filters below</li>
            <li>Click "Get Recommendations" to see your personalized top 10 games</li>
            <li>Download your recommendations or save them for later</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    # Upload section with improved UI
    with st.expander("📤 Upload Your Game Dataset", expanded=True):
        uploaded_file = st.file_uploader(
            "Drag and drop your CSV file here or click to browse",
            type="csv",
            help="Your file should include columns for 'Title', 'Genres', 'User Score', and other relevant game information."
        )

    if uploaded_file is not None:
        try:
            # Load and process data with progress indicators
            with st.spinner("Processing your dataset..."):
                df_uploaded = pd.read_csv(uploaded_file)
                
                # Data validation and processing
                required_columns = ['Title', 'Genres', 'User Score']
                missing_cols = [col for col in required_columns if col not in df_uploaded.columns]
                
                if missing_cols:
                    st.error(f"Missing required columns: {', '.join(missing_cols)}")
                    st.stop()
                
                # Data cleaning
                df_uploaded['Genres'] = df_uploaded['Genres'].astype(str).fillna('')
                df_uploaded['User Score'] = pd.to_numeric(df_uploaded['User Score'], errors='coerce')
                df_uploaded = df_uploaded.dropna(subset=['User Score'])
                
                # Display dataset information
                st.success(f"✅ Successfully loaded dataset with {len(df_uploaded)} games")
                
                # Show basic stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Games", len(df_uploaded))
                with col2:
                    st.metric("Unique Genres", df_uploaded['Genres'].nunique())
                with col3:
                    avg_score = df_uploaded['User Score'].mean()
                    st.metric("Average User Score", f"{avg_score:.1f}/10")

            # Filter options in an expandable section
            with st.expander("🔍 Set Your Preferences", expanded=True):
                st.subheader("Filter Options")
                
                # Multi-select for genres with search capability
                all_genres = sorted(set(g for genres in df_uploaded['Genres'].str.split(', ') 
                                     for g in genres if g))
                selected_genres = st.multiselect(
                    "Select preferred genres:",
                    options=all_genres,
                    default=[],
                    help="Select one or more genres you're interested in"
                )
                
                # Slider for user score with better formatting
                min_score, max_score = st.slider(
                    "Select acceptable user score range:",
                    min_value=0.0,
                    max_value=10.0,
                    value=(6.0, 10.0),
                    step=0.1,
                    format="%.1f",
                    help="Adjust the range to filter by user ratings"
                )
                
                # Additional filters
                st.subheader("Advanced Filters")
                col1, col2 = st.columns(2)
                with col1:
                    min_reviews = st.number_input(
                        "Minimum user reviews:",
                        min_value=0,
                        value=10,
                        help="Filter out games with few user reviews"
                    )
                with col2:
                    year_range = st.slider(
                        "Release year range:",
                        min_value=1980,
                        max_value=2023,
                        value=(2000, 2023),
                        help="Filter by when games were released"
                    )

            # Recommendation button with loading state
            if st.button("🎯 Get My Recommendations", use_container_width=True):
                with st.spinner("Finding the perfect games for you..."):
                    try:
                        # Apply filters
                        filtered_df = df_uploaded.copy()
                        
                        # Genre filter
                        if selected_genres:
                            genre_filter = filtered_df['Genres'].apply(
                                lambda x: any(genre in x for genre in selected_genres)
                            filtered_df = filtered_df[genre_filter]
                        
                        # Score filter
                        score_filter = (filtered_df['User Score'] >= min_score) & \
                                      (filtered_df['User Score'] <= max_score)
                        filtered_df = filtered_df[score_filter]
                        
                        # Additional filters
                        if 'User Ratings Count' in filtered_df.columns:
                            filtered_df = filtered_df[filtered_df['User Ratings Count'] >= min_reviews]
                        
                        if 'Release Date' in filtered_df.columns:
                            filtered_df['Release Year'] = pd.to_datetime(
                                filtered_df['Release Date'], errors='coerce').dt.year
                            year_filter = (filtered_df['Release Year'] >= year_range[0]) & \
                                         (filtered_df['Release Year'] <= year_range[1])
                            filtered_df = filtered_df[year_filter]
                        
                        # Sort and select top 10
                        if not filtered_df.empty:
                            recommended_games = filtered_df.sort_values(
                                by='User Score', ascending=False).head(10)
                            
                            # Display results in a nice format
                            st.subheader("🌟 Your Top 10 Recommended Games")
                            
                            # Show as cards for better visual appeal
                            cols = st.columns(2)
                            for idx, game in recommended_games.iterrows():
                                with cols[idx % 2]:
                                    with st.container():
                                        st.markdown(f"""
                                        <div style='background-color: #333333; padding: 15px; border-radius: 10px; margin-bottom: 15px;'>
                                            <h4 style='color: #4CAF50;'>{game['Title']}</h4>
                                            <p><b>Genre:</b> {game['Genres']}</p>
                                            <p><b>User Score:</b> {game['User Score']:.1f}/10</p>
                                            {'<p><b>Platforms:</b> ' + game['Platforms'] + '</p>' if 'Platforms' in game else ''}
                                            {'<p><b>Release Date:</b> ' + str(game['Release Date']) + '</p>' if 'Release Date' in game else ''}
                                        </div>
                                        """, unsafe_allow_html=True)
                            
                            # Download options
                            st.markdown("---")
                            st.subheader("📥 Download Your Recommendations")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                csv = recommended_games.to_csv(index=False)
                                st.download_button(
                                    label="Download as CSV",
                                    data=csv,
                                    file_name='my_game_recommendations.csv',
                                    mime='text/csv',
                                    help="Save your recommendations as a CSV file"
                                )
                            with col2:
                                json_data = recommended_games.to_json(orient='records')
                                st.download_button(
                                    label="Download as JSON",
                                    data=json_data,
                                    file_name='my_game_recommendations.json',
                                    mime='application/json',
                                    help="Save your recommendations as a JSON file"
                                )
                            with col3:
                                if st.button("Save to My Profile", help="Coming soon - save to your account"):
                                    st.info("This feature is coming soon!")
                        else:
                            st.warning("""
                            No games match your current filters. Try:
                            - Broadening your genre selection
                            - Adjusting the score range
                            - Removing some filters
                            """)
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
                        st.exception(e)
        except Exception as e:
            st.error(f"Failed to process your file: {str(e)}")
    else:
        st.info("""
        ℹ️ To get started, please upload a CSV file containing game data. 
        Need sample data? [Download example dataset](#) (link coming soon)
        """)
        
# Page 3: Game Correlation Finder
elif page == "Game Correlation Finder":
    st.title('🎮 Game Correlation Finder')
    st.markdown("Find out how games are related based on user ratings!")

    @st.cache_data
    def load_data():
        path = 'all_video_games(cleaned).csv'
        df = pd.read_csv(path)
        path_user = 'User_Dataset.csv'
        userset = pd.read_csv(path_user)
        data = pd.merge(df, userset, on='Title').dropna()  
        return data

    data = load_data()

    score_matrix = data.pivot_table(index='user_id', columns='Title', values='user_score', fill_value=0)

    game_titles = score_matrix.columns.sort_values().tolist()

    # Split layout into two columns for better organization
    col1, col2 = st.columns([1, 3])

    with col1:
        # Game title selection
        game_title = st.selectbox("Select a game title", game_titles, help="Choose a game to see its correlation with others.")

    # Divider line for better visual separation
    st.markdown("---")

    if game_title:
        game_user_score = score_matrix[game_title]
        similar_to_game = score_matrix.corrwith(game_user_score)
        corr_drive = pd.DataFrame(similar_to_game, columns=['Correlation']).dropna()

        # Display the top 10 correlated games in the second column
        with col2:
            st.subheader(f"🎯 Correlations for '{game_title}'")
            st.dataframe(corr_drive.sort_values('Correlation', ascending=False).head(10))

        # Display number of user scores for each game
        user_scores_count = data.groupby('Title')['user_score'].count().rename('total num_of_user_score')
        merged_corr_drive = corr_drive.join(user_scores_count, how='left')

        # Calculate average user score for each game
        avg_user_score = data.groupby('Title')['user_score'].mean().rename('avg_user_score')
        detailed_corr_info = merged_corr_drive.join(avg_user_score, how='left')

        # Add developer and publisher columns (assuming they're in the dataset)
        additional_info = data[['Title', 'Developer', 'Genres']].drop_duplicates().set_index('Title')
        detailed_corr_info = detailed_corr_info.join(additional_info, how='left')

        # Show detailed high-score correlations with more information
        st.subheader("Games you may like (with more than 10 number of user scores and average user score):")
        high_score_corr = detailed_corr_info[detailed_corr_info['total num_of_user_score'] > 10].sort_values('Correlation', ascending=False).head()
        
        # Display the dataframe with additional details including average user score
        st.dataframe(high_score_corr[['Correlation', 'total num_of_user_score', 'avg_user_score', 'Developer', 'Genres']])

    else:
        st.warning("Please select a game title from the dropdown to see the correlations.")


# About Page
elif page == "About":
    st.title("📖 About")
    st.markdown("""
    This Game Recommendation System was developed to help gamers discover new titles they might enjoy.
    
    ### How It Works:
    - *Content-Based Recommendations*: This method uses the genres, platforms, and publishers of games to find similar titles.
    - *Top 10 Recommendations*: Upload your own dataset and apply filters to get personalized game recommendations.
    - *Game Correlation Finder*: Analyze game correlations based on user ratings to find titles that have similar user reception.
    
    ### Technologies Used:
    - *Python*: For building the recommendation algorithms.
    - *Streamlit*: For creating an interactive and user-friendly interface.
    - *Pandas & Scikit-learn*: For data manipulation and machine learning.

    ### Creator:
    This app was created as part of an AI project to explore recommendation systems in gaming.
    """)

# Footer
st.markdown("<h5 style='text-align: center;'>Powered by Streamlit</h5>", unsafe_allow_html=True)
