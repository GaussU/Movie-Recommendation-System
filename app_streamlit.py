#!/usr/bin/env python3
"""
Streamlit Movie Recommendation System Web Application
Integrates functionality from train_model.py and init_db.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from pathlib import Path

# Import system modules
from movie_system_simplified import (
    PersistenceService,
    RecommendationService,
    DataIngestionService,
    DomainRegistry,
    UserModel,
    MovieModel,
    RatingModel,
    GenreModel,
    OccupationModel,
    setup_chinese_font
)
from sqlalchemy import func

# Import training modules
from train_model import (
    load_data_to_database,
    build_ratings_matrix,
    build_enhanced_user_and_movie_features,
    cluster_users_enhanced,
    cluster_movies_enhanced,
    train_matrix_factorization_model,
    train_test_split_ratings,
    evaluate_matrix_factorization_model,
    handle_new_user_recommendations_enhanced,
    MatrixFactorizationModel,
    EnhancedUserProfileBuilder,
    EnhancedMovieProfileBuilder,
    parse_binary_int
)
# EnhancedNewUserHandler is imported from movie_system_simplified

# Helper function: Ensure proper handling of binary data
def safe_parse_rating(rating_value):
    """Safely parse rating value, handling binary and integer formats"""
    if isinstance(rating_value, bytes):
        return parse_binary_int(rating_value)
    try:
        return int(rating_value)
    except (ValueError, TypeError):
        return 3  # Default value

# ===========================================================================
# Streamlit Configuration (only executed in Streamlit environment)
# ===========================================================================

def configure_streamlit():
    """Configure Streamlit page (only called in Streamlit environment)"""
    st.set_page_config(
        page_title="Movie Recommendation System",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    # Setup Chinese font
    setup_chinese_font()

# ===========================================================================
# Configuration and Constants
# ===========================================================================

OCCUPATION_MAP = {
    0: "other", 1: "academic/educator", 2: "artist", 3: "clerk",
    4: "college/grad student", 5: "customer service", 6: "doctor/health care",
    7: "executive/managerial", 8: "farmer", 9: "homemaker", 10: "K-12 student",
    11: "lawyer", 12: "programmer", 13: "retired", 14: "sales/marketing",
    15: "scientist", 16: "self-employed", 17: "technician/engineer",
    18: "tradesman/craftsman", 19: "unemployed", 20: "writer"
}

AGE_MAP = {
    1: "Under 18", 18: "18-24", 25: "25-34", 35: "35-44", 
    45: "45-49", 50: "50-55", 56: "56+"
}

# ===========================================================================
# Cached Services
# ===========================================================================

@st.cache_resource
def get_persistence_service():
    """
    Get persistence service (with caching)
    Add retry mechanism to handle database locking issues
    """
    import time
    max_retries = 3
    retry_delay = 1  # seconds
    
    for attempt in range(max_retries):
        try:
            return PersistenceService()
        except Exception as e:
            error_msg = str(e)
            if "database is locked" in error_msg.lower() and attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            else:
                raise

@st.cache_resource
def get_recommendation_service(_persistence):
    """Get recommendation service and train model (with caching, enhanced new user recommendation enabled by default)"""
    # Enable enhanced new user recommendation by default
    rec_service = RecommendationService(_persistence, n_factors=50, use_enhanced_new_user=True)
    if not rec_service._is_trained:
        with st.spinner("Training recommendation model..."):
            rec_service.train()
    return rec_service

# ===========================================================================
# Database Initialization Functions (shared by CLI and Web interface)
# ===========================================================================

def init_database_cli(data_path=None, drop_existing=False):
    """
    Database initialization function for CLI mode
    Does not depend on Streamlit, can be called directly from command line
    Uses MovieLensSystem class from preprocessing.py
    
    Parameters:
        data_path (str, optional): Data folder path, defaults to "../movie_dataset"
        drop_existing (bool): Whether to delete existing database, defaults to False
    
    Returns:
        bool: Returns True on success, False on failure
    """
    # Use MovieLensSystem from preprocessing.py (consistent with preprocessing.ipynb)
    from preprocessing import MovieLensSystem
    import os
    
    if data_path is None:
        data_path = "../movie_dataset"
    
    if not os.path.exists(data_path):
        print(f"❌ Error: Data path does not exist: {data_path}")
        return False
    
    # Check required files
    required_files = ["users.dat", "movies.dat", "ratings.dat"]
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(data_path, f))]
    if missing_files:
        print(f"❌ Error: Missing required files: {', '.join(missing_files)}")
        return False
    
    try:
        system = MovieLensSystem(data_path)
        system.initialize(drop_existing_db=drop_existing)
        return True
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def init_database_ui():
    """Database initialization interface"""
    st.header("📊 Database Initialization")
    st.markdown("---")
    
    # Check current database status
    persistence = get_persistence_service()
    overview = persistence.get_system_overview()
    
    if overview['n_ratings'] > 0:
        st.info(f"📊 Current database has data: {overview['n_users']} users, {overview['n_movies']} movies, {overview['n_ratings']} ratings")
        st.warning("⚠️ If re-initializing, existing data will be preserved (no duplicate insertion).")
        st.markdown("---")
    
    # Data path configuration
    st.subheader("Data Path Configuration")
    st.info("💡 **Tip**: This feature uses the exact same initialization logic as `preprocessing.ipynb` and `init_db.py` (based on `MovieLensSystem` class in `preprocessing.py`), ensuring consistency.")
    
    default_data_path = "../movie_dataset"
    data_path = st.text_input(
        "Data Folder Path",
        value=default_data_path,
        help="Path to folder containing .dat files (e.g., users.dat, movies.dat, ratings.dat)"
    )
    
    # Check path and files
    path_valid = os.path.exists(data_path) if data_path else False
    if path_valid:
        required_files = ["users.dat", "movies.dat", "ratings.dat"]
        existing_files = [f for f in required_files if os.path.exists(os.path.join(data_path, f))]
        missing_files = [f for f in required_files if f not in existing_files]
        
        if existing_files:
            st.success(f"✅ Found files: {', '.join(existing_files)}")
        if missing_files:
            st.error(f"❌ Missing files: {', '.join(missing_files)}")
    else:
        st.warning(f"⚠️ Path does not exist: {data_path}")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        init_button = st.button("🚀 Initialize Database", type="primary", use_container_width=True)
    
    if init_button:
        if not path_valid:
            st.error(f"❌ Error: Cannot find data folder `{data_path}`")
            st.info("Please ensure the data folder exists and contains the following files:\n- users.dat\n- movies.dat\n- ratings.dat")
            return
        
        # Check files
        required_files = ["users.dat", "movies.dat", "ratings.dat"]
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(data_path, f))]
        if missing_files:
            st.error(f"❌ Missing required files: {', '.join(missing_files)}")
            return
        
        # Execute initialization
        with st.spinner("Initializing database, this may take a few minutes..."):
            try:
                # Use MovieLensSystem from preprocessing.py (consistent with preprocessing.ipynb)
                from preprocessing import MovieLensSystem
                system = MovieLensSystem(data_path)
                # Don't delete existing data, only add new data
                system.initialize(drop_existing_db=False)
                success = True
                
                if success:
                    st.success("✅ Database initialization successful!")
                    st.balloons()
                    
                    # Display database status
                    persistence = get_persistence_service()
                    overview = persistence.get_system_overview()
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Users", overview['n_users'])
                    with col2:
                        st.metric("Movies", overview['n_movies'])
                    with col3:
                        st.metric("Ratings", overview['n_ratings'])
                    
                    # Clear cache, force reload
                    st.cache_resource.clear()
                    if 'db_init_prompt_shown' in st.session_state:
                        del st.session_state['db_init_prompt_shown']
                    
                    st.info("💡 Tip: You can now start training models or using recommendation features!")
                else:
                    st.error("❌ Database initialization failed, please check error messages")
            except Exception as e:
                st.error(f"❌ Error during initialization: {str(e)}")
                st.exception(e)

# ===========================================================================
# Model Training Functions
# ===========================================================================

def train_model_ui():
    """Model training interface"""
    st.header("🤖 Model Training")
    st.markdown("---")
    
    # Check database status
    persistence = get_persistence_service()
    overview = persistence.get_system_overview()
    
    if overview['n_ratings'] == 0:
        st.warning("⚠️ Database is empty! Please initialize the database first.")
        return
    
    st.info(f"📊 Current database status: {overview['n_users']} users, {overview['n_movies']} movies, {overview['n_ratings']} ratings")
    
    # Training parameter configuration
    st.subheader("Training Parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        n_factors = st.slider("Number of Factors", min_value=10, max_value=100, value=50, step=10)
    with col2:
        learning_rate = st.slider("Learning Rate", min_value=0.001, max_value=0.01, value=0.005, step=0.001, format="%.3f")
    with col3:
        n_epochs = st.slider("Number of Epochs", min_value=10, max_value=50, value=20, step=5)
    
    reg_param = st.slider("Regularization Parameter", min_value=0.01, max_value=0.1, value=0.02, step=0.01, format="%.2f")
    test_size = st.slider("Test Set Ratio", min_value=0.1, max_value=0.3, value=0.2, step=0.05, format="%.2f")
    
    # Training options
    st.subheader("Training Options")
    auto_cluster = st.checkbox("Auto-select number of user clusters", value=True)
    train_mf_model = st.checkbox("Train Matrix Factorization Model (Gradient Descent)", value=True)
    train_svd_model = st.checkbox("Train SVD Recommendation Model", value=True)
    
    # Start training button
    if st.button("🚀 Start Training", type="primary", use_container_width=True):
        if not (train_mf_model or train_svd_model):
            st.warning("Please select at least one model to train")
            return
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # STEP 1: Build ratings matrix
            status_text.text("Step 1/8: Building ratings matrix...")
            progress_bar.progress(10)
            ratings_matrix, user_ids, movie_ids = build_ratings_matrix()
            
            # STEP 2: Feature engineering
            status_text.text("Step 2/8: Building user and movie features...")
            progress_bar.progress(20)
            (user_builder, movie_builder, user_features_X, movie_features_X,
             user_features_reduced, movie_features_reduced, user_features_full, movies_df, ratings_df) = \
                build_enhanced_user_and_movie_features()
            
            # STEP 3: User clustering
            status_text.text("Step 3/8: User clustering...")
            progress_bar.progress(30)
            user_clusters = cluster_users_enhanced(user_builder, user_features_reduced, auto_select=auto_cluster)
            
            # STEP 4: Movie clustering
            status_text.text("Step 4/8: Movie clustering...")
            progress_bar.progress(40)
            movie_clusters = cluster_movies_enhanced(movie_builder, movie_features_reduced, n_clusters=8)
            
            # STEP 5: Split train/test sets
            status_text.text("Step 5/8: Splitting train/test sets...")
            progress_bar.progress(50)
            train_ratings, test_ratings = train_test_split_ratings(ratings_matrix, test_size=test_size)
            
            # STEP 6: Train matrix factorization model
            if train_mf_model:
                status_text.text("Step 6/8: Training matrix factorization model...")
                progress_bar.progress(60)
                mf_model = train_matrix_factorization_model(
                    train_ratings,
                    val_ratings=None,
                    n_factors=n_factors,
                    learning_rate=learning_rate,
                    reg_param=reg_param,
                    n_epochs=n_epochs
                )
                
                # STEP 7: Model evaluation
                status_text.text("Step 7/8: Evaluating model performance...")
                progress_bar.progress(80)
                metrics = evaluate_matrix_factorization_model(mf_model, test_ratings)
                
                # Display evaluation results
                st.subheader("📈 Model Evaluation Results")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("RMSE", f"{metrics['rmse']:.4f}")
                with col2:
                    st.metric("MAE", f"{metrics['mae']:.4f}")
                with col3:
                    st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                with col4:
                    st.metric("F1 Score", f"{metrics['f1']:.4f}")
                
                # Save model to session state
                st.session_state['mf_model'] = mf_model
                st.session_state['user_features_X'] = user_features_X
                st.session_state['user_ids'] = user_ids
            
            # STEP 8: Train SVD model
            if train_svd_model:
                status_text.text("Step 8/8: Training SVD recommendation model (enhanced new user recommendation enabled)...")
                progress_bar.progress(90)
                # Cache will be used here, but we need to retrain
                persistence = get_persistence_service()
                # Enable enhanced new user recommendation
                rec_service = RecommendationService(persistence, n_factors=n_factors, use_enhanced_new_user=True)
                rec_service.train()
                st.session_state['rec_service'] = rec_service
            
            progress_bar.progress(100)
            status_text.text("✅ Training complete!")
            st.success("🎉 Model training completed successfully!")
            st.balloons()
            
        except Exception as e:
            st.error(f"❌ Error during training: {str(e)}")
            st.exception(e)

# ===========================================================================
# Recommendation Functions
# ===========================================================================

def recommendation_ui():
    """Recommendation interface"""
    st.header("🎯 Movie Recommendations")
    st.markdown("---")
    
    # Check database status
    persistence = get_persistence_service()
    overview = persistence.get_system_overview()
    
    if overview['n_ratings'] == 0:
        st.warning("⚠️ Database is empty! Please initialize the database first.")
        return
    
    # Check if model is trained
    if 'rec_service' not in st.session_state:
        with st.spinner("Loading recommendation model..."):
            try:
                rec_service = get_recommendation_service(persistence)
                st.session_state['rec_service'] = rec_service
            except Exception as e:
                st.error(f"❌ Unable to load recommendation model: {str(e)}")
                st.info("Please train the model first")
                return
    
    rec_service = st.session_state['rec_service']
    
    # User selection method
    st.subheader("User Selection")
    user_mode = st.radio(
        "Select User Mode",
        ["Existing User", "New User"],
        horizontal=True
    )
    
    if user_mode == "Existing User":
        # Existing user recommendations
        user_id = st.number_input(
            "User ID",
            min_value=1,
            max_value=overview['n_users'],
            value=1,
            step=1
        )
        
        limit = st.slider("Number of Recommendations", min_value=5, max_value=50, value=10, step=5)
        
        if st.button("🚀 Get Recommendations", type="primary", use_container_width=True):
            with st.spinner("Calculating recommendations..."):
                try:
                    recommendations = rec_service.recommend_for_user(
                        user_id=user_id,
                        limit=limit,
                        exclude_rated=True
                    )
                    
                    if recommendations:
                        display_recommendations(recommendations, user_id)
                    else:
                        st.warning("No recommendations found")
                except Exception as e:
                    st.error(f"❌ Recommendation failed: {str(e)}")
    
    else:
        # New user recommendations
        st.subheader("User Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_age = st.selectbox("Age Group", list(AGE_MAP.values()))
            age_id = [k for k, v in AGE_MAP.items() if v == selected_age][0]
        with col2:
            gender = st.selectbox("Gender", ["M", "F"])
        with col3:
            selected_occ = st.selectbox("Occupation", list(OCCUPATION_MAP.values()))
            occ_id = [k for k, v in OCCUPATION_MAP.items() if v == selected_occ][0]
        
        limit = st.slider("Number of Recommendations", min_value=5, max_value=50, value=10, step=5)
        
        if st.button("🚀 Get Recommendations", type="primary", use_container_width=True):
            # Create temporary user
            temp_user_id = 999999
            
            # Check/create user
            session = persistence.db.get_session()
            try:
                existing_user = session.query(UserModel).filter_by(user_id=temp_user_id).first()
                if not existing_user:
                    session.add(UserModel(
                        user_id=temp_user_id,
                        gender=gender,
                        age=age_id,
                        occupation_id=occ_id,
                        zip_code="00000"
                    ))
                    session.commit()
                else:
                    existing_user.gender = gender
                    existing_user.age = age_id
                    existing_user.occupation_id = occ_id
                    session.commit()
            finally:
                session.close()
            
            with st.spinner("Calculating recommendations using enhanced algorithm (includes demographic-genre preference mapping)..."):
                try:
                    # Enhanced recommendation will automatically detect new users and use enhanced algorithm
                    recommendations = rec_service.recommend_for_user(
                        user_id=temp_user_id,
                        limit=limit,
                        exclude_rated=False,
                        preferred_genres=None  # Can add user preferred genres here
                    )
                    
                    if recommendations:
                        display_recommendations(recommendations, f"New User ({selected_age}, {gender}, {selected_occ})")
                        # Display enhanced recommendation tip
                        st.info("💡 This recommendation used the enhanced new user recommendation algorithm, combining demographic-genre preference mapping and similar user analysis")
                    else:
                        st.warning("No recommendations found")
                except Exception as e:
                    st.error(f"❌ Recommendation failed: {str(e)}")
                    import traceback
                    st.exception(e)

def display_recommendations(recommendations, user_info):
    """Display recommendation results"""
    st.subheader(f"🎬 Recommended Movies for You")
    st.caption(f"User: {user_info}")
    
    # Display recommendation list
    for i, rec in enumerate(recommendations, 1):
        with st.container():
            col1, col2, col3 = st.columns([4, 1, 1])
            with col1:
                st.markdown(f"**{i}. {rec['title']}** ({rec.get('release_year', 'N/A')})")
                genres_str = ', '.join(rec.get('genres', []))
                st.caption(f"Genres: {genres_str}")
            with col2:
                score = rec.get('predicted_rating', 0)
                st.metric("Predicted Rating", f"{score:.2f}")
            with col3:
                st.progress(min(score / 5.0, 1.0))
            st.markdown("---")
    
    # Visualization analysis
    if len(recommendations) > 0:
        st.subheader("📊 Recommendation Analysis")
        
        rec_df = pd.DataFrame(recommendations)
        
        tab1, tab2, tab3 = st.tabs(["Rating Distribution", "Genre Distribution", "Year Distribution"])
        
        with tab1:
            if 'predicted_rating' in rec_df.columns:
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.hist(rec_df['predicted_rating'], bins=20, edgecolor='white', color='#4CAF50', alpha=0.8)
                ax.set_xlabel('Predicted Rating')
                ax.set_ylabel('Number of Movies')
                ax.set_title('Recommended Movies Rating Distribution')
                st.pyplot(fig)
        
        with tab2:
            all_genres = []
            for genres in rec_df.get('genres', []):
                if isinstance(genres, list):
                    all_genres.extend(genres)
            
            if all_genres:
                genre_counts = pd.Series(all_genres).value_counts()
                st.bar_chart(genre_counts)
            else:
                st.info("No genre data")
        
        with tab3:
            if 'release_year' in rec_df.columns:
                rec_df['release_year'] = pd.to_numeric(rec_df['release_year'], errors='coerce')
                valid_years = rec_df['release_year'].dropna()
                
                if len(valid_years) > 0:
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.hist(valid_years, bins=15, edgecolor='white', color='#2196F3', alpha=0.7)
                    ax.set_xlabel('Year')
                    ax.set_ylabel('Number of Movies')
                    ax.set_title('Recommended Movies Year Distribution')
                    st.pyplot(fig)
                else:
                    st.info("No year data")

# ===========================================================================
# System Status
# ===========================================================================

def check_binary_data_ui():
    """Check database binary data"""
    st.subheader("🔍 Database Data Check")
    
    db_path = "movielens.db"
    if not os.path.exists(db_path):
        st.warning("Database file does not exist")
        return False
    
    import sqlite3
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Check for binary data
        cursor.execute("SELECT COUNT(*) FROM ratings WHERE typeof(rating) = 'blob'")
        binary_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM ratings")
        total_count = cursor.fetchone()[0]
        
        if binary_count > 0:
            st.warning(f"⚠️ Found {binary_count} ratings in binary format (out of {total_count} total)")
            st.info("💡 Recommended to run fix tool: `python fix_binary_data.py`")
            return True
        else:
            st.success(f"✅ Database data format is normal (total {total_count} ratings)")
            return False
    except Exception as e:
        st.error(f"Check failed: {str(e)}")
        return False
    finally:
        conn.close()

def system_status_ui():
    """System status interface"""
    st.header("📊 System Status")
    st.markdown("---")
    
    persistence = get_persistence_service()
    overview = persistence.get_system_overview()
    
    # Database status
    st.subheader("Database Status")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Users", overview['n_users'])
    with col2:
        st.metric("Movies", overview['n_movies'])
    with col3:
        st.metric("Ratings", overview['n_ratings'])
    
    # Data format check
    has_binary = check_binary_data_ui()
    
    # Model status
    st.subheader("Model Status")
    if 'rec_service' in st.session_state:
        st.success("✅ Recommendation model loaded")
        if st.session_state['rec_service']._is_trained:
            st.info("✅ Model trained")
        else:
            st.warning("⚠️ Model not trained")
    else:
        st.warning("⚠️ Recommendation model not loaded")
    
    if 'mf_model' in st.session_state:
        st.success("✅ Matrix factorization model trained")
    
    # Database file information
    st.subheader("Database File")
    db_path = "movielens.db"
    if os.path.exists(db_path):
        file_size = os.path.getsize(db_path) / (1024 * 1024)  # MB
        st.info(f"📁 Database file: {db_path} ({file_size:.2f} MB)")
        
        if has_binary:
            st.markdown("---")
            st.subheader("🔧 Data Repair")
            st.info("""
            **Binary Data Issue Detected**
            
            If you encounter recommendation or training issues, you can run the following command to fix:
            ```bash
            python fix_binary_data.py
            ```
            
            Or re-initialize the database.
            """)
    else:
        st.warning(f"⚠️ Database file does not exist: {db_path}")

# ===========================================================================
# Main Application
# ===========================================================================

def main():
    """Main function"""
    # Configure Streamlit
    configure_streamlit()
    
    # Sidebar
    st.sidebar.title("🎬 Movie Recommendation System")
    st.sidebar.markdown("---")
    
    # Automatically check database status on startup (with error handling)
    try:
        persistence = get_persistence_service()
        overview = persistence.get_system_overview()
        db_empty = overview['n_ratings'] == 0
    except Exception as e:
        error_msg = str(e)
        if "database is locked" in error_msg.lower() or "locked" in error_msg.lower():
            st.error("🔒 **Database is locked!**")
            st.warning("""
            **Possible causes:**
            1. Another process is using the database (e.g., `init_db.py` is running)
            2. Previous Streamlit instance was not properly closed
            3. Database connection was not properly released
            
            **Solutions:**
            1. Wait for `init_db.py` to complete (if running)
            2. Close all Streamlit instances and restart
            3. If problem persists, try deleting the database file and re-initializing
            """)
            st.stop()
        else:
            st.error(f"❌ Database access error: {error_msg}")
            st.exception(e)
            st.stop()
        return
    
    # If database is empty, show prompt
    if db_empty and 'db_init_prompt_shown' not in st.session_state:
        st.session_state['db_init_prompt_shown'] = True
        st.warning("⚠️ **Database is empty!** Please initialize the database first to use recommendation features.")
        st.info("💡 Tip: You can:\n1. Click 「📊 Database Initialization」 in the left menu to initialize\n2. Or run `python init_db.py` in the command line")
        st.markdown("---")
    
    # Navigation menu
    page = st.sidebar.radio(
        "Navigation",
        ["🏠 System Status", "📊 Database Initialization", "🤖 Model Training", "🎯 Movie Recommendations"],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    
    # Display database status (sidebar)
    st.sidebar.subheader("📊 Database Status")
    if db_empty:
        st.sidebar.error("❌ Database is empty")
    else:
        st.sidebar.success(f"✅ {overview['n_users']} users")
        st.sidebar.success(f"✅ {overview['n_movies']} movies")
        st.sidebar.success(f"✅ {overview['n_ratings']} ratings")
    
    st.sidebar.markdown("---")
    
    # Control panel
    st.sidebar.subheader("🔧 Control Panel")
    if st.sidebar.button("🔄 Clear Cache and Reload", use_container_width=True):
        st.cache_resource.clear()
        if 'rec_service' in st.session_state:
            del st.session_state['rec_service']
        if 'mf_model' in st.session_state:
            del st.session_state['mf_model']
        if 'db_init_prompt_shown' in st.session_state:
            del st.session_state['db_init_prompt_shown']
        st.rerun()
    
    # Display system information
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📖 Usage Instructions")
    st.sidebar.info("""
    1. **Database Initialization**: Initialize database on first use
    2. **Model Training**: Train recommendation model (optional)
    3. **Movie Recommendations**: Get personalized recommendations
    """)
    
    # Display content based on selected page
    if page == "🏠 System Status":
        system_status_ui()
    elif page == "📊 Database Initialization":
        init_database_ui()
    elif page == "🤖 Model Training":
        train_model_ui()
    elif page == "🎯 Movie Recommendations":
        recommendation_ui()

if __name__ == "__main__":
    # Support command line arguments: python app_streamlit.py --init-db [data_path] [--drop-existing]
    # This allows initializing database without starting Streamlit interface
    if len(sys.argv) > 1 and sys.argv[1] == "--init-db":
        data_path = None
        drop_existing = False
        
        for arg in sys.argv[2:]:
            if arg == "--drop-existing":
                drop_existing = True
            elif not arg.startswith("--"):
                data_path = arg
        
        print("=" * 70)
        print("CLI Mode: Database Initialization")
        print("=" * 70)
        print("\nNote: This feature uses the same initialization logic as the Streamlit Web interface")
        print("(based on MovieLensSystem class in preprocessing.py)\n")
        
        if drop_existing:
            print("⚠️  Warning: Will delete existing database and recreate\n")
        
        success = init_database_cli(data_path=data_path, drop_existing=drop_existing)
        
        if success:
            print("\n" + "=" * 70)
            print("✅ Database initialization complete!")
            print("=" * 70)
            print("\nYou can now run the Streamlit application:")
            print("  streamlit run app_streamlit.py")
        else:
            print("\n" + "=" * 70)
            print("❌ Database initialization failed!")
            print("=" * 70)
            sys.exit(1)
    else:
        # Normal Streamlit application startup
        main()

