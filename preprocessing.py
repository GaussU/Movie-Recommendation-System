#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy import func

from abc import ABC, abstractmethod
from typing import List, Dict, Set
from datetime import datetime


# # ORM model

# In[2]:


Base = declarative_base()

movie_genre_association = Table(
    'movie_genre_association',
    Base.metadata,
    Column('movie_id', Integer, ForeignKey('movies.movie_id'), primary_key=True),
    Column('genre_id', Integer, ForeignKey('genres.genre_id'), primary_key=True)
)

class UserModel(Base):
    __tablename__ = 'users'
    user_id = Column(Integer, primary_key=True)
    gender = Column(String(1))
    age = Column(Integer)
    occupation_id = Column(Integer, ForeignKey('occupations.occupation_id'))
    zip_code = Column(String(10))
    ratings = relationship("RatingModel", back_populates="user")
    occupation = relationship("OccupationModel", back_populates="users")

class OccupationModel(Base):
    __tablename__ = 'occupations'
    occupation_id = Column(Integer, primary_key=True)
    occupation_name = Column(String(50))
    users = relationship("UserModel", back_populates="occupation")

class MovieModel(Base):
    __tablename__ = 'movies'
    movie_id = Column(Integer, primary_key=True)
    title = Column(String(100))
    release_year = Column(Integer)
    ratings = relationship("RatingModel", back_populates="movie")
    genres = relationship("GenreModel", secondary=movie_genre_association, back_populates="movies")

class GenreModel(Base):
    __tablename__ = 'genres'
    genre_id = Column(Integer, primary_key=True, autoincrement=True)
    genre_name = Column(String(50), unique=True)
    movies = relationship("MovieModel", secondary=movie_genre_association, back_populates="genres")

class RatingModel(Base):
    __tablename__ = 'ratings'
    rating_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.user_id'), index = True)
    movie_id = Column(Integer, ForeignKey('movies.movie_id'), index = True)
    rating = Column(Integer, index = True)
    timestamp = Column(Integer, index = True)
    user = relationship("UserModel", back_populates="ratings")
    movie = relationship("MovieModel", back_populates="ratings")


# # Class

# In[3]:


class User:
    def __init__(self, user_id: int, gender: str, age: int, occupation_id: int, zip_code: str):
        self.user_id = user_id
        self.gender = gender
        self.age = age
        self.occupation_id = occupation_id
        self.zip_code = zip_code
        self._ratings: List['Rating'] = []

    def age_group(self) -> str:
        age_map = {
            1: "Under 18", 18: "18-24", 25: "25-34",
            35: "35-44", 45: "45-49", 50: "50-55", 56: "56+"
        }
        return age_map.get(self.age, "Unknown")
    
    def add_rating(self, rating: 'Rating'):
        self._ratings.append(rating)
    
    def get_ratings(self) -> List['Rating']:
        return self._ratings

class Movie:
    def __init__(self, movie_id: int, title: str, release_year: int):
        self.movie_id = movie_id
        self.title = title
        self.release_year = release_year
        self._genres: Set['Genre'] = set()
        self._ratings: List['Rating'] = []
    
    def add_genre(self, genre: 'Genre'):
        self._genres.add(genre)
    
    def get_genres(self) -> Set['Genre']:
        return self._genres
    
    def add_rating(self, rating: 'Rating'):
        self._ratings.append(rating)
    
    def get_ratings(self) -> List['Rating']:
        return self._ratings

class Genre:
    def __init__(self, genre_id: int, genre_name: str):
        self.genre_id = genre_id
        self.genre_name = genre_name
        self._movies: Set[Movie] = set()
    
    def add_movie(self, movie: Movie):
        self._movies.add(movie)
    
    def get_movies(self) -> Set[Movie]:
        return self._movies

class Rating:
    def __init__(self, rating_id: int, user: User, movie: Movie, rating: int, timestamp: int):
        self.rating_id = rating_id
        self.user = user
        self.movie = movie
        self.rating = rating
        self.timestamp = timestamp
    
    def rating_date(self) -> datetime:
        if isinstance(self.timestamp, bytes):
            timestamp_int = int.from_bytes(self.timestamp, byteorder='big')
            return datetime.fromtimestamp(timestamp_int)
        return datetime.fromtimestamp(self.timestamp)

class Occupation:
    def __init__(self, occupation_id: int, occupation_name: str):
        self.occupation_id = occupation_id
        self.occupation_name = occupation_name
        self._users: List[User] = []
    
    def add_user(self, user: User):
        self._users.append(user)
    
    def get_users(self) -> List[User]:
        return self._users


# In[4]:


class DatabaseManager:
    """database management"""
    
    def __init__(self, db_url: str = 'sqlite:///movielens.db'):
        self.engine = create_engine(db_url)
        self.Session = sessionmaker(bind=self.engine)
        self.Base = Base
    
    def get_session(self):
        """get new database conversation"""
        return self.Session()
    
    def init_db(self, drop_existing):
        """database initialization"""
        if drop_existing:
            self.Base.metadata.drop_all(self.engine)
        self.Base.metadata.create_all(self.engine)
    
    def domain_to_orm(self, domain_obj):
        """change into ORM"""
        if isinstance(domain_obj, User):
            return UserModel(
                user_id=domain_obj.user_id,
                gender=domain_obj.gender,
                age=domain_obj.age,
                occupation_id=domain_obj.occupation_id,
                zip_code=domain_obj.zip_code
            )
        elif isinstance(domain_obj, Movie):
            return MovieModel(
                movie_id=domain_obj.movie_id,
                title=domain_obj.title,
                release_year=domain_obj.release_year
            )
        elif isinstance(domain_obj, Genre):
            return GenreModel(
                genre_name=domain_obj.genre_name
            )
        elif isinstance(domain_obj, Occupation):
            return OccupationModel(
                occupation_id=domain_obj.occupation_id,
                occupation_name=domain_obj.occupation_name
            )
        elif isinstance(domain_obj, Rating):
            return RatingModel(
                rating_id=domain_obj.rating_id,
                user_id=domain_obj.user.user_id,
                movie_id=domain_obj.movie.movie_id,
                rating=int(domain_obj.rating),
                timestamp=int(domain_obj.timestamp)
            )
        raise ValueError(f"Unknown domain object type: {type(domain_obj)}")

    def orm_to_domain(self, orm_obj):
        """get domain"""
        if isinstance(orm_obj, UserModel):
            return User(
                user_id=orm_obj.user_id,
                gender=orm_obj.gender,
                age=orm_obj.age,
                occupation_id=orm_obj.occupation_id,
                zip_code=orm_obj.zip_code
            )
        elif isinstance(orm_obj, MovieModel):
            movie = Movie(
                movie_id=orm_obj.movie_id,
                title=orm_obj.title,
                release_year=orm_obj.release_year
            )
            return movie
        elif isinstance(orm_obj, GenreModel):
            return Genre(
                genre_id=orm_obj.genre_id,
                genre_name=orm_obj.genre_name
            )
        elif isinstance(orm_obj, OccupationModel):
            return Occupation(
                occupation_id=orm_obj.occupation_id,
                occupation_name=orm_obj.occupation_name
            )
        elif isinstance(orm_obj, RatingModel):
            user = self.orm_to_domain(orm_obj.user)
            movie = self.orm_to_domain(orm_obj.movie)
            return Rating(
                rating_id=orm_obj.rating_id,
                user=user,
                movie=movie,
                rating=orm_obj.rating,
                timestamp=int(orm_obj.timestamp)
            )
        raise ValueError(f"Unknown ORM object type: {type(orm_obj)}")


# In[5]:


class DataAccessManager:
    """data"""

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    def get_ratings_by_user(self, user_id: int) -> List[Rating]:
        """get user ratings by id"""
        session = self.db_manager.get_session()
        try:
            ratings = session.query(RatingModel).filter_by(user_id=user_id).all()
            return [self.db_manager.orm_to_domain(rating) for rating in ratings]
        finally:
            session.close()
        
    def get_all_users(self) -> List[User]:
        """get all users information"""
        session = self.db_manager.get_session()
        try:
            users_orm = session.query(UserModel).all()
            return [self.db_manager.orm_to_domain(user) for user in users_orm]
        finally:
            session.close()
    
    def get_ratings_by_movie(self, movie_id: int) -> List[Rating]:
        """get movie ratings by id"""
        session = self.db_manager.get_session()
        try:
            ratings = session.query(RatingModel).filter_by(movie_id=movie_id).all()
            return [self.db_manager.orm_to_domain(rating) for rating in ratings]
        finally:
            session.close()

    def get_all_movies(self) -> List[Movie]:
        """get all movies information"""
        session = self.db_manager.get_session()
        try:
            movies_orm = session.query(MovieModel).all()
            return [self.db_manager.orm_to_domain(movie) for movie in movies_orm]
        finally:
            session.close()
    
    def get_all_ratings(self) -> List[Rating]:
        """get all ratings information"""
        session = self.db_manager.get_session()
        try:
            ratings_orm = session.query(RatingModel).all()
            return [self.db_manager.orm_to_domain(rating) for rating in ratings_orm]
        finally:
            session.close()

    def get_movie_genres(self, movie_id: int) -> List[Genre]:
        """get movie type by id"""
        session = self.db_manager.get_session()
        try:
            movie = session.query(MovieModel).filter_by(movie_id=movie_id).first()
            if movie and movie.genres:
                return [self.db_manager.orm_to_domain(genre) for genre in movie.genres]
            return []
        finally:
            session.close()


# In[6]:


class DataProcessor:
    """load data and clean"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
    
    def load_and_clean(self) -> dict:
        return {
            'users': self._load_users(),
            'movies': self._load_movies(),
            'ratings': self._load_ratings()
        }
    
    def _load_users(self) -> pd.DataFrame:
        """load and clean users data"""
        df = pd.read_table(
            f"{self.data_path}/users.dat", sep="::",
            header=None, names=["user_id", "gender", "age", "occupation", "zip_code"],
            engine='python', encoding='latin1'
        )
        
        df['gender'] = df['gender'].apply(lambda x: x if x in ['M', 'F'] else 'U')
        df['age'] = df['age'].apply(lambda x: x if x in [1, 18, 25, 35, 45, 50, 56] else 0)
        df['occupation'] = df['occupation'].apply(lambda x: x if 0 <= x <= 20 else 0)
        
        return df
    
    def _load_movies(self) -> pd.DataFrame:
        """load and clean movies data"""
        df = pd.read_table(
            f"{self.data_path}/movies.dat", sep="::",
            header=None, names=["movie_id", "title", "genres"],
            engine='python', encoding='latin1'
        )
        
        df['release_year'] = df['title'].str.extract(r'\((\d{4})\)')
        df['release_year'] = pd.to_numeric(df['release_year'])
        df['title'] = df['title'].str.replace(r'\(\d{4}\)', '', regex=True).str.strip()
        df['genres_list'] = df['genres'].str.split('|')
        
        return df
    
    def _load_ratings(self) -> pd.DataFrame:
        """load and clean ratings data"""
        df = pd.read_table(
            f"{self.data_path}/ratings.dat", sep="::",
            header=None, names=["user_id", "movie_id", "rating", "timestamp"],
            engine='python', encoding='latin1'
        )
        
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
        
        return df


# In[7]:


class DataAnalyzer:
    """data analysis"""
    
    def __init__(self, data_access: DataAccessManager):
        self.data_access = data_access


# In[8]:


class DataVisualizer:
    """data visualization"""
    
    def __init__(self, data_analyzer: DataAnalyzer):
        self.analyzer = data_analyzer


# In[9]:


class DataExporter:
    """export data"""
    
    def __init__(self, data_access: DataAccessManager):
        self.data_access = data_access
    
    def export_to_dataframes(self) -> Dict[str, pd.DataFrame]:
        """get df"""
        users = self.data_access.get_all_users()
        movies = self.data_access.get_all_movies()
        ratings = self.data_access.get_all_ratings()
        
        users_data = []
        for user in users:
            user_ratings = self.data_access.get_ratings_by_user(user.user_id)
            users_data.append({
                'user_id': user.user_id,
                'gender': user.gender,
                'age': user.age,
                'age_group': user.age_group(),
                'occupation_id': user.occupation_id,
                'zip_code': user.zip_code,
                'rating_count': len(user_ratings)
            })
        
        movies_data = []
        for movie in movies:
            movie_ratings = self.data_access.get_ratings_by_movie(movie.movie_id)
            movie_genres = self.data_access.get_movie_genres(movie.movie_id)
            movies_data.append({
                'movie_id': movie.movie_id,
                'title': movie.title,
                'release_year': movie.release_year,
                'genres': ', '.join([g.genre_name for g in movie_genres]),
                'rating_count': len(movie_ratings)
            })
        
        ratings_data = []
        for rating in ratings:
            ratings_data.append({
                'rating_id': rating.rating_id,
                'user_id': rating.user.user_id,
                'movie_id': rating.movie.movie_id,
                'rating': rating.rating,
                'timestamp': rating.timestamp,
                'rating_date': rating.rating_date()
            })
        
        return {
            'users': pd.DataFrame(users_data),
            'movies': pd.DataFrame(movies_data),
            'ratings': pd.DataFrame(ratings_data)
        }


# In[10]:


class MovieLensSystem:
    """MovieLens system main class"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.db_manager = DatabaseManager()
        self.data_processor = DataProcessor(data_path)
        self.data_access = DataAccessManager(self.db_manager)
        # self.data_analyzer = DataAnalyzer(self.data_access)
        # self.data_visualizer = DataVisualizer(self.data_analyzer)
        self.data_exporter = DataExporter(self.data_access)
        
        self._users: Dict[int, User] = {}
        self._movies: Dict[int, Movie] = {}
        self._genres: Dict[str, Genre] = {}
        self._occupations: Dict[int, Occupation] = {}
    
    def initialize(self, drop_existing_db):
        # initialize database
        self.db_manager.init_db(drop_existing=drop_existing_db)

        """initialization system"""
        self._users.clear()
        self._movies.clear()
        self._genres.clear()
        self._occupations.clear()
        
        # load and clean data
        data = self.data_processor.load_and_clean()
        
        # create tables
        self._create_occupations()
        self._create_users(data['users'])
        self._create_movies_and_genres(data['movies'])
        self._create_ratings(data['ratings'])
        
        # 4. save to database
        self._save_all_to_db()
    
    def _create_occupations(self):
        """create occupation"""
        occupations = [
        (0, "other"), (1, "academic/educator"), (2, "artist"), (3, "clerk"),
        (4, "college/grad student"), (5, "customer service"), (6, "doctor/health care"),
        (7, "executive/managerial"), (8, "farmer"), (9, "homemaker"), (10, "K-12 student"),
        (11, "lawyer"), (12, "programmer"), (13, "retired"), (14, "sales/marketing"),
        (15, "scientist"), (16, "self-employed"), (17, "technician/engineer"),
        (18, "tradesman/craftsman"), (19, "unemployed"), (20, "writer")
        ]
        
        for occ_id, occ_name in occupations:
            self._occupations[occ_id] = Occupation(occ_id, occ_name)
    
    def _create_users(self, users_df: pd.DataFrame):
        """create users"""
        for _, row in users_df.iterrows():
            user = User(
                user_id=row['user_id'],
                gender=row['gender'],
                age=row['age'],
                occupation_id=row['occupation'],
                zip_code=row['zip_code']
            )
            self._users[user.user_id] = user
            self._occupations[row['occupation']].add_user(user)
    
    def _create_movies_and_genres(self, movies_df: pd.DataFrame):
        """create movies and types"""
        all_genres = set()
        for _, row in movies_df.iterrows():
            all_genres.update(row['genres_list'])
        
        for genre_name in sorted(all_genres): 
            if genre_name not in self._genres:
                genre_id = len(self._genres) + 1
                self._genres[genre_name] = Genre(genre_id, genre_name)
        
        for _, row in movies_df.iterrows():
            movie = Movie(
                movie_id=row['movie_id'],
                title=row['title'],
                release_year=row['release_year']
            )
            
            for genre_name in row['genres_list']:
                genre = self._genres[genre_name]
                movie.add_genre(genre)
                genre.add_movie(movie)
            
            self._movies[movie.movie_id] = movie
    
    def _create_ratings(self, ratings_df: pd.DataFrame):
        """create ratings"""
        rating_id = 1

        for _, row in ratings_df.iterrows():
            user = self._users.get(row['user_id'])
            movie = self._movies.get(row['movie_id'])
            
            if user and movie:
                rating = Rating(
                    rating_id=rating_id,
                    user=user,
                    movie=movie,
                    rating=row['rating'],
                    timestamp=row['timestamp']
                )
                user.add_rating(rating)
                movie.add_rating(rating)
                rating_id += 1
    
    def _save_all_to_db(self):
        """save all data to database"""
        session = self.db_manager.get_session()
    
        try:
            for occupation in self._occupations.values():
                existing_occupation = session.query(OccupationModel).filter_by(
                    occupation_id=occupation.occupation_id
                ).first()
                if not existing_occupation:
                    session.add(self.db_manager.domain_to_orm(occupation))
            
            for genre in self._genres.values():
                existing_genre = session.query(GenreModel).filter_by(
                    genre_name=genre.genre_name
                ).first()
                if not existing_genre:
                    session.add(self.db_manager.domain_to_orm(genre))
            
            session.flush()
            
            for user in self._users.values():
                existing_user = session.query(UserModel).filter_by(
                    user_id=user.user_id
                ).first()
                if not existing_user:
                    session.add(self.db_manager.domain_to_orm(user))
            
            session.flush()
            
            for movie in self._movies.values():
                existing_movie = session.query(MovieModel).filter_by(
                    movie_id=movie.movie_id
                ).first()
                if not existing_movie:
                    movie_orm = self.db_manager.domain_to_orm(movie)
                    # 先添加到 session，然后再添加关联
                    session.add(movie_orm)
                    session.flush()  # 确保 movie_orm 在 session 中
                    
                    for genre in movie.get_genres():
                        genre_orm = session.query(GenreModel).filter_by(
                            genre_name=genre.genre_name
                        ).first()
                        if genre_orm and genre_orm not in movie_orm.genres:
                            movie_orm.genres.append(genre_orm)
            
            session.flush()
            
            for user in self._users.values():
                for rating in user.get_ratings():
                    existing_rating = session.query(RatingModel).filter_by(
                        rating_id=rating.rating_id
                    ).first()
                    if not existing_rating:
                        session.add(self.db_manager.domain_to_orm(rating))
            
            session.commit()
            print("Data saved successfully.")
        except Exception as e:
            session.rollback()
            print(f"Data saved failed.: {e}")
            raise
        finally:
            session.close()


# # Application code
# 
# 注意：以下代码已被注释，避免在导入时自动执行
# 如果需要测试，请手动运行这些代码
#
# import os
# if os.path.exists('movielens.db'):
#     os.remove('movielens.db')
#
# system = MovieLensSystem("../movie_dataset/")
#
# # initialize and load data
# system.initialize(drop_existing_db=True)
#
#
# df = system.data_exporter.export_to_dataframes()
#
#
# print("=== Users table ===")
# print(df['users'].head())  
#
# print("=== Movies table ===")
# print(df['movies'].head())
#
# print("=== Ratings table ===")
# print(df['ratings'].head())
#
#
# df2 = df['ratings']
# df2['rating_id'].nunique()

