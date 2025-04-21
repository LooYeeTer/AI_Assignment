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

# Page 2: File Upload and Filters
elif page == "Top 10 Recommendation based on User Preferences":
    st.title("🎮 Personalized Game Recommendations")
    
    # =============================================
    # 1. Upload Section
    # =============================================
    st.markdown("""
    <div style='background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 20px;'>
        <h3 style='color: #2e7d32;'>How to use this feature:</h3>
        <ol style='line-height: 1.6;'>
            <li>Upload your game dataset in CSV format</li>
            <li>Set your preferences using the filters</li>
            <li>Click "Get Recommendations"</li>
            <li>Download or save your results</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📤 Upload Your Game Dataset", expanded=True):
        uploaded_file = st.file_uploader(
            "Drag and drop your CSV file here or click to browse",
            type="csv",
            help="Required columns: 'Title', 'Genres', 'User Score'"
        )

    # =============================================
    # 2. Data Processing
    # =============================================
    if uploaded_file is not None:
        try:
            with st.spinner("Processing your dataset..."):
                df = pd.read_csv(uploaded_file)
                
                # Data validation
                required_columns = ['Title', 'Genres', 'User Score']
                missing_cols = [col for col in required_columns if col not in df.columns]
                
                if missing_cols:
                    st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                    st.stop()
                
                # Data cleaning
                df['Genres'] = df['Genres'].astype(str).str.strip()
                df['User Score'] = pd.to_numeric(df['User Score'], errors='coerce')
                df = df.dropna(subset=['User Score'])
                
                # Display stats
                st.success(f"✅ Loaded {len(df)} games")
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Games", len(df))
                col2.metric("Unique Genres", df['Genres'].nunique())
                col3.metric("Avg Score", f"{df['User Score'].mean():.1f}/10")

            # =============================================
            # 3. Filters Section
            # =============================================
            with st.expander("🔍 Set Your Preferences", expanded=True):
                # Genre filter (with search)
                all_genres = sorted({g.strip() for genres in df['Genres'].str.split(',') for g in genres})
                selected_genres = st.multiselect(
                    "Preferred genres:",
                    options=all_genres,
                    default=[],
                    help="Start typing to search genres"
                )
                
                # Score filter
                min_score, max_score = st.slider(
                    "User score range:",
                    min_value=0.0,
                    max_value=10.0,
                    value=(6.0, 10.0),
                    step=0.1,
                    format="%.1f"
                )
                
                # Advanced filters
                st.subheader("Advanced Options")
                adv_col1, adv_col2 = st.columns(2)
                with adv_col1:
                    min_reviews = st.number_input(
                        "Minimum reviews:",
                        min_value=0,
                        value=10
                    )
                with adv_col2:
                    year_range = st.slider(
                        "Release years:",
                        min_value=1980,
                        max_value=2023,
                        value=(2000, 2023)
                    )

            # =============================================
            # 4. Recommendations Section
            # =============================================
            if st.button("🎯 Get Recommendations", type="primary", use_container_width=True):
                with st.spinner("Finding your perfect games..."):
                    # Apply filters
                    filtered_df = df.copy()
                    
                    # Genre filter
                    if selected_genres:
                        filtered_df = filtered_df[
                            filtered_df['Genres'].apply(
                                lambda x: any(g.strip() in x for g in selected_genres)
                            )
                        ]
                    
                    # Score filter
                    filtered_df = filtered_df[
                        (filtered_df['User Score'] >= min_score) & 
                        (filtered_df['User Score'] <= max_score)
                    ]
                    
                    # Additional filters
                    if 'User Ratings Count' in filtered_df.columns:
                        filtered_df = filtered_df[filtered_df['User Ratings Count'] >= min_reviews]
                    
                    if 'Release Date' in filtered_df.columns:
                        filtered_df['Year'] = pd.to_datetime(filtered_df['Release Date']).dt.year
                        filtered_df = filtered_df[
                            (filtered_df['Year'] >= year_range[0]) & 
                            (filtered_df['Year'] <= year_range[1])
                        ]
                    
                    # Display results
                    if not filtered_df.empty:
                        top_games = filtered_df.sort_values('User Score', ascending=False).head(10)
                        
                        st.subheader("🌟 Your Top 10 Recommendations")
                        st.markdown("---")
                        
                        # Display as cards
                        for _, game in top_games.iterrows():
                            with st.container():
                                st.markdown(f"""
                                <div style='
                                    background: #333333;
                                    padding: 15px;
                                    border-radius: 8px;
                                    margin-bottom: 15px;
                                '>
                                    <h4 style='color: #4CAF50;'>{game['Title']}</h4>
                                    <p><b>Genre:</b> {game['Genres']}</p>
                                    <p><b>Score:</b> {game['User Score']:.1f}/10</p>
                                    {f"<p><b>Platform:</b> {game['Platforms']}</p>" if 'Platforms' in game else ""}
                                    {f"<p><b>Released:</b> {game['Release Date']}</p>" if 'Release Date' in game else ""}
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Download options
                        st.markdown("---")
                        st.subheader("📥 Export Results")
                        dl_col1, dl_col2 = st.columns(2)
                        with dl_col1:
                            st.download_button(
                                "Download CSV",
                                data=top_games.to_csv(index=False),
                                file_name="game_recommendations.csv"
                            )
                        with dl_col2:
                            st.download_button(
                                "Download JSON",
                                data=top_games.to_json(orient="records"),
                                file_name="game_recommendations.json"
                            )
                    else:
                        st.warning("No games match your filters. Try adjusting them.")
                        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    else:
        st.info("ℹ️ Upload a CSV file to get started")
        
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
