#!/usr/bin/env python3
"""
完整的模型训练流程示例
结合 movie_system_simplified.py 和 mapping&recommend code.ipynb 的所有功能
包括数据加载、用户/电影分析、特征工程、聚类、推荐模型训练、评估等
"""

from movie_system_simplified import (
    PersistenceService,
    RecommendationService,
    DataIngestionService,
    DomainRegistry,
    ProfilingService,
    NewUserHandler,
    EnhancedNewUserHandler,
    ModelEvaluator,
    DataAnalyzer,
    UserModel,
    MovieModel,
    RatingModel,
    GenreModel,
    OccupationModel,
    setup_chinese_font
)
import os
import numpy as np
import pandas as pd
from sqlalchemy import func
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error, mean_absolute_error, silhouette_score
import warnings
warnings.filterwarnings('ignore')

# ===========================================================================
# 辅助函数：解析二进制字段
# ===========================================================================

def parse_binary_int(value):
    """将二进制字段解析为整数"""
    if isinstance(value, bytes):
        return int.from_bytes(value[:8], byteorder='little', signed=False)
    return int(value)

# ===========================================================================
# 矩阵分解模型 (来自 mapping notebook)
# ===========================================================================

class MatrixFactorizationModel:
    """基于矩阵分解的推荐模型 - 使用梯度下降训练"""
    
    def __init__(self, n_factors=50, learning_rate=0.005, reg_param=0.02, n_epochs=20):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.reg_param = reg_param
        self.n_epochs = n_epochs
        self.user_factors = None
        self.item_factors = None
        self.global_bias = None
        self.user_biases = None
        self.item_biases = None
        self.train_losses = []
        self.val_losses = []
        
    def fit(self, ratings, val_ratings=None, verbose=True):
        """训练矩阵分解模型"""
        print("=== 开始训练矩阵分解模型 ===")
        
        n_users, n_items = ratings.shape
        self.global_bias = np.nanmean(ratings)
        
        # 初始化参数
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        self.user_biases = np.zeros(n_users)
        self.item_biases = np.zeros(n_items)
        
        # 获取非零评分索引
        train_indices = np.argwhere(~np.isnan(ratings))
        
        if val_ratings is not None:
            val_indices = np.argwhere(~np.isnan(val_ratings))
        
        print(f"训练样本数: {len(train_indices)}")
        
        for epoch in range(self.n_epochs):
            epoch_loss = 0
            np.random.shuffle(train_indices)
            
            for idx in train_indices:
                i, j = idx
                true_rating = ratings[i, j]
                
                # 预测评分
                pred_rating = self._predict_single(i, j)
                
                # 计算误差
                error = true_rating - pred_rating
                
                # 更新参数
                self.user_biases[i] += self.learning_rate * (error - self.reg_param * self.user_biases[i])
                self.item_biases[j] += self.learning_rate * (error - self.reg_param * self.item_biases[j])
                
                # 更新因子
                user_factor_grad = error * self.item_factors[j] - self.reg_param * self.user_factors[i]
                item_factor_grad = error * self.user_factors[i] - self.reg_param * self.item_factors[j]
                
                self.user_factors[i] += self.learning_rate * user_factor_grad
                self.item_factors[j] += self.learning_rate * item_factor_grad
                
                epoch_loss += error ** 2
            
            # 计算平均损失
            avg_loss = epoch_loss / len(train_indices)
            self.train_losses.append(avg_loss)
            
            # 计算验证损失
            if val_ratings is not None and len(val_indices) > 0:
                val_loss = 0
                for idx in val_indices:
                    i, j = idx
                    true_rating = val_ratings[i, j]
                    pred_rating = self._predict_single(i, j)
                    val_loss += (true_rating - pred_rating) ** 2
                self.val_losses.append(val_loss / len(val_indices))
            
            if verbose and (epoch + 1) % 5 == 0:
                if val_ratings is not None and self.val_losses:
                    print(f"Epoch {epoch+1}/{self.n_epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {self.val_losses[-1]:.4f}")
                else:
                    print(f"Epoch {epoch+1}/{self.n_epochs}, Train Loss: {avg_loss:.4f}")
        
        print("矩阵分解模型训练完成!")
        return self
    
    def _predict_single(self, user_idx, item_idx):
        """预测单个评分"""
        return (self.global_bias + 
                self.user_biases[user_idx] + 
                self.item_biases[item_idx] + 
                np.dot(self.user_factors[user_idx], self.item_factors[item_idx]))
    
    def predict(self, user_idx, item_idx):
        """预测评分"""
        return self._predict_single(user_idx, item_idx)
    
    def get_user_embedding(self, user_idx):
        """获取用户嵌入向量"""
        return self.user_factors[user_idx]
    
    def get_item_embedding(self, item_idx):
        """获取物品嵌入向量"""
        return self.item_factors[item_idx]

# ===========================================================================
# 增强的用户画像构建器 (来自 mapping notebook)
# ===========================================================================

class EnhancedUserProfileBuilder:
    """增强的用户画像构建器 - 包含类型偏好和最佳聚类数选择"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)
        self.kmeans = None
        self.is_fitted = False
        self.silhouette_scores = {}
        self.inertia_values = []
        
    def build_user_features(self, users_df, ratings_df, movies_df):
        """构建用户特征矩阵 - 包含类型偏好"""
        print("=== 开始构建用户特征 ===")
        
        # 基本特征
        users_df = users_df.copy()
        users_df['gender_encoded'] = users_df['gender'].map({'M': 0, 'F': 1, 'U': 0.5}).fillna(0.5)
        
        # 用户行为特征
        user_stats = ratings_df.groupby('user_id')['rating'].agg([
            ('rating_count', 'count'),
            ('avg_rating', 'mean'),
            ('rating_std', 'std'),
        ]).reset_index()
        user_stats['rating_std'] = user_stats['rating_std'].fillna(0)
        
        # 类型偏好特征
        try:
            ratings_with_genres = pd.merge(ratings_df, movies_df[['movie_id', 'genres']], 
                                         on='movie_id', how='left')
            ratings_with_genres['genres_list'] = ratings_with_genres['genres'].str.split('|')
            exploded_ratings = ratings_with_genres.explode('genres_list')
            
            genre_preferences = exploded_ratings.groupby(['user_id', 'genres_list'])['rating'].mean().unstack(fill_value=0)
            genre_preferences.columns = [f'genre_{col}' for col in genre_preferences.columns]
            
            user_features = pd.merge(users_df, user_stats, on='user_id', how='left')
            user_features = pd.merge(user_features, genre_preferences, on='user_id', how='left')
        except Exception as e:
            print(f"类型偏好特征构建失败，使用基础特征: {e}")
            user_features = pd.merge(users_df, user_stats, on='user_id', how='left')
        
        # 选择数值型特征
        feature_columns = ['gender_encoded', 'age', 'occupation', 'rating_count', 'avg_rating', 'rating_std']
        genre_columns = [col for col in user_features.columns if col.startswith('genre_')]
        feature_columns.extend(genre_columns)
        
        available_columns = [col for col in feature_columns if col in user_features.columns]
        user_features[available_columns] = user_features[available_columns].fillna(0)
        
        print(f"用户特征维度: {user_features[available_columns].shape}")
        return user_features[available_columns], user_features
    
    def reduce_dimensionality(self, features):
        """特征降维"""
        features = features.astype(float)
        features_scaled = self.scaler.fit_transform(features)
        features_reduced = self.pca.fit_transform(features_scaled)
        print(f"降维后特征维度: {features_reduced.shape}")
        print(f"保留的方差比例: {np.sum(self.pca.explained_variance_ratio_):.4f}")
        return features_reduced
    
    def find_optimal_clusters(self, features_reduced, max_k=10):
        """使用肘部法则和轮廓系数确定最佳聚类数"""
        print("步骤: 寻找最佳聚类数...")
        
        if len(features_reduced) < 2:
            return 2
        
        inertias = []
        silhouette_scores = []
        k_range = range(2, min(max_k + 1, len(features_reduced)))
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features_reduced)
            inertias.append(kmeans.inertia_)
            
            if len(np.unique(cluster_labels)) > 1:
                if len(features_reduced) > 1000:
                    sample_indices = np.random.choice(len(features_reduced), 
                                                    size=min(1000, len(features_reduced)), 
                                                    replace=False)
                    sample_features = features_reduced[sample_indices]
                    sample_labels = cluster_labels[sample_indices]
                    silhouette_avg = silhouette_score(sample_features, sample_labels)
                else:
                    silhouette_avg = silhouette_score(features_reduced, cluster_labels)
            else:
                silhouette_avg = -1
            
            silhouette_scores.append(silhouette_avg)
            print(f"聚类数 {k}: Inertia = {kmeans.inertia_:.2f}, Silhouette = {silhouette_avg:.4f}")
        
        self.inertia_values = inertias
        self.silhouette_scores = dict(zip(k_range, silhouette_scores))
        
        valid_scores = {k: score for k, score in zip(k_range, silhouette_scores) if score > 0}
        if valid_scores:
            optimal_k = max(valid_scores, key=valid_scores.get)
            print(f"基于轮廓系数选择最佳聚类数: {optimal_k}")
        else:
            optimal_k = 5  # 默认值
        
        return optimal_k
    
    def build_user_profiles(self, features_reduced, n_clusters=None, auto_select_clusters=True, max_k=8):
        """构建用户画像聚类"""
        if auto_select_clusters and n_clusters is None:
            n_clusters = self.find_optimal_clusters(features_reduced, max_k)
        elif n_clusters is None:
            n_clusters = 5
        
        print(f"使用聚类数: {n_clusters}")
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        user_clusters = self.kmeans.fit_predict(features_reduced)
        
        if len(np.unique(user_clusters)) > 1:
            final_silhouette = silhouette_score(features_reduced, user_clusters)
            print(f"最终聚类轮廓系数: {final_silhouette:.4f}")
        
        self.is_fitted = True
        return user_clusters

# ===========================================================================
# 增强的电影画像构建器 (来自 mapping notebook)
# ===========================================================================

class EnhancedMovieProfileBuilder:
    """增强的电影画像构建器 - 包含类型特征"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)
        self.kmeans = None
        
    def build_movie_features(self, movies_df, ratings_df):
        """构建电影特征矩阵 - 包含类型特征"""
        print("=== 开始构建电影特征 ===")
        
        movies_df = movies_df.copy()
        
        # 类型特征
        try:
            movies_df['genres_list'] = movies_df['genres'].str.split('|')
            all_genres = set()
            for genres in movies_df['genres_list'].dropna():
                all_genres.update(genres)
            
            genre_features = []
            for _, row in movies_df.iterrows():
                movie_genres = row['genres_list'] if isinstance(row.get('genres_list'), list) else []
                genre_row = {f'genre_{genre}': 1 if genre in movie_genres else 0 for genre in all_genres}
                genre_row['movie_id'] = row['movie_id']
                genre_features.append(genre_row)
            
            genre_features_df = pd.DataFrame(genre_features)
        except Exception as e:
            print(f"类型特征构建失败: {e}")
            genre_features_df = pd.DataFrame({'movie_id': movies_df['movie_id']})
        
        # 评分特征
        try:
            movie_stats = ratings_df.groupby('movie_id')['rating'].agg([
                ('rating_count', 'count'),
                ('avg_rating', 'mean'),
                ('rating_std', 'std'),
            ]).reset_index()
            movie_stats['rating_std'] = movie_stats['rating_std'].fillna(0)
        except Exception as e:
            print(f"评分特征构建失败: {e}")
            movie_stats = pd.DataFrame({'movie_id': movies_df['movie_id'].unique()})
        
        # 合并特征
        movie_features = pd.merge(movies_df[['movie_id', 'release_year']], 
                                genre_features_df, on='movie_id', how='left')
        movie_features = pd.merge(movie_features, movie_stats, on='movie_id', how='left')
        
        # 选择数值型特征
        feature_columns = ['release_year'] + \
                         [col for col in movie_features.columns if col.startswith('genre_')] + \
                         ['avg_rating', 'rating_count', 'rating_std']
        
        available_columns = [col for col in feature_columns if col in movie_features.columns]
        movie_features[available_columns] = movie_features[available_columns].fillna(0)
        
        print(f"电影特征维度: {movie_features[available_columns].shape}")
        return movie_features[available_columns], movie_features
    
    def reduce_dimensionality(self, features):
        """特征降维"""
        features = features.astype(float)
        features_scaled = self.scaler.fit_transform(features)
        features_reduced = self.pca.fit_transform(features_scaled)
        print(f"降维后特征维度: {features_reduced.shape}")
        print(f"保留的方差比例: {np.sum(self.pca.explained_variance_ratio_):.4f}")
        return features_reduced
    
    def build_movie_profiles(self, features_reduced, n_clusters=8):
        """构建电影画像聚类"""
        print(f"使用聚类数: {n_clusters}")
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        movie_clusters = self.kmeans.fit_predict(features_reduced)
        return movie_clusters

# ===========================================================================
# 增强的新用户处理器 (从 movie_system_simplified 导入)
# ===========================================================================
# EnhancedNewUserHandler 已从 movie_system_simplified 导入
# 该类包含：
# - 人口统计学-类型偏好映射
# - 智能相似用户查找（余弦相似度）
# - 混合推荐策略

# ===========================================================================
# 1. 数据加载阶段
# ===========================================================================

def load_data_to_database(data_path=None):
    """
    将数据加载到数据库
    
    参数:
        data_path (str, optional): 数据文件夹路径，默认为 "../movie_dataset"
    
    返回:
        bool: 成功返回 True，失败返回 False
    """
    print("=" * 70)
    print("STEP 1: 数据加载和数据库初始化")
    print("=" * 70)
    
    persistence = PersistenceService()
    overview = persistence.get_system_overview()
    
    if overview['n_ratings'] > 0:
        print(f"\n✓ 数据库已有数据:")
        print(f"  用户: {overview['n_users']}人")
        print(f"  电影: {overview['n_movies']}部")
        print(f"  评分: {overview['n_ratings']}条")
        return True
    
    print("\n数据库为空，开始加载数据...")
    
    # 使用默认路径或提供的路径
    if data_path is None:
        data_path = "../movie_dataset"
    if not os.path.exists(data_path):
        print(f"✗ 错误: 数据路径不存在: {data_path}")
        print("请确保 movie_dataset 文件夹存在")
        return False
    
    print("\n步骤1.1: 加载原始数据到内存...")
    registry = DomainRegistry()
    ingestion = DataIngestionService(data_path, registry)
    ingestion.load()
    
    print(f"  ✓ 加载完成: {len(registry.users)}用户, {len(registry.movies)}电影")
    
    print("\n步骤1.2: 保存数据到数据库...")
    session = persistence.db.get_session()
    try:
        for occ in registry.occupations.values():
            existing = session.query(OccupationModel).filter_by(occupation_id=occ.occupation_id).first()
            if not existing:
                session.add(OccupationModel(occupation_id=occ.occupation_id, occupation_name=occ.occupation_name))
        
        session.flush()
        
        for genre in registry.genres.values():
            existing = session.query(GenreModel).filter_by(genre_name=genre.genre_name).first()
            if not existing:
                session.add(GenreModel(genre_name=genre.genre_name))
        
        session.flush()
        
        for user in registry.users.values():
            existing = session.query(UserModel).filter_by(user_id=user.user_id).first()
            if not existing:
                session.add(UserModel(
                    user_id=user.user_id,
                    gender=user.gender,
                    age=user.age,
                    occupation_id=user.occupation_id,
                    zip_code=user.zip_code
                ))
        
        session.flush()
        
        for movie in registry.movies.values():
            existing = session.query(MovieModel).filter_by(movie_id=movie.movie_id).first()
            if not existing:
                movie_orm = MovieModel(
                    movie_id=movie.movie_id,
                    title=movie.title,
                    release_year=movie.release_year
                )
                session.add(movie_orm)
                session.flush()
                
                for genre in movie.genres:
                    genre_orm = session.query(GenreModel).filter_by(genre_name=genre.genre_name).first()
                    if genre_orm and genre_orm not in movie_orm.genres:
                        movie_orm.genres.append(genre_orm)
        
        session.flush()
        
        for user in registry.users.values():
            for rating in user.ratings:
                existing = session.query(RatingModel).filter_by(
                    user_id=rating.user.user_id,
                    movie_id=rating.movie.movie_id
                ).first()
                if not existing:
                    max_id = session.query(func.max(RatingModel.rating_id)).scalar() or 0
                    # 确保 rating 以整数格式存储，避免二进制格式问题
                    rating_value = int(rating.rating) if rating.rating is not None else 3
                    session.add(RatingModel(
                        rating_id=max_id + 1,
                        user_id=rating.user.user_id,
                        movie_id=rating.movie.movie_id,
                        rating=rating_value,  # 明确使用整数
                        timestamp=rating.timestamp
                    ))
        
        session.commit()
        print("  ✓ 数据保存成功")
        
        overview = persistence.get_system_overview()
        print(f"\n✓ 数据库状态:")
        print(f"  用户: {overview['n_users']}人")
        print(f"  电影: {overview['n_movies']}部")
        print(f"  评分: {overview['n_ratings']}条")
        
        return True
        
    except Exception as e:
        session.rollback()
        print(f"  ✗ 保存失败: {e}")
        return False
    finally:
        session.close()

# ===========================================================================
# 2. 构建评分矩阵
# ===========================================================================

def build_ratings_matrix():
    """构建用户-电影评分矩阵"""
    print("\n" + "=" * 70)
    print("STEP 2: 构建评分矩阵")
    print("=" * 70)
    
    persistence = PersistenceService()
    session = persistence.db.get_session()
    
    try:
        # 获取所有用户和电影
        users = session.query(UserModel).all()
        movies = session.query(MovieModel).all()
        
        user_ids = [u.user_id for u in users]
        movie_ids = [m.movie_id for m in movies]
        
        user_id_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
        movie_id_to_idx = {mid: idx for idx, mid in enumerate(movie_ids)}
        
        # 初始化评分矩阵
        n_users = len(user_ids)
        n_movies = len(movie_ids)
        ratings_matrix = np.full((n_users, n_movies), np.nan)
        
        # 填充评分矩阵
        for rating in session.query(RatingModel).all():
            user_idx = user_id_to_idx.get(rating.user_id)
            movie_idx = movie_id_to_idx.get(rating.movie_id)
            
            if user_idx is not None and movie_idx is not None:
                ratings_matrix[user_idx, movie_idx] = parse_binary_int(rating.rating)
        
        sparsity = 1 - np.sum(~np.isnan(ratings_matrix)) / (n_users * n_movies)
        print(f"\n评分矩阵形状: {ratings_matrix.shape}")
        print(f"矩阵稀疏度: {sparsity:.4f}")
        
        return ratings_matrix, user_ids, movie_ids
        
    finally:
        session.close()

# ===========================================================================
# 3. 用户和电影特征工程 (增强版)
# ===========================================================================

def build_enhanced_user_and_movie_features():
    """构建增强的用户和电影特征"""
    print("\n" + "=" * 70)
    print("STEP 3: 增强的用户和电影特征工程")
    print("=" * 70)
    
    persistence = PersistenceService()
    session = persistence.db.get_session()
    
    try:
        # 获取数据
        users_data = []
        movies_data = []
        ratings_data = []
        
        for user in session.query(UserModel).all():
            users_data.append({
                'user_id': int(user.user_id),
                'gender': str(user.gender),
                'age': int(user.age),
                'occupation': int(user.occupation_id),
            })
        
        for movie in session.query(MovieModel).all():
            genres_str = ', '.join([g.genre_name for g in movie.genres])
            movies_data.append({
                'movie_id': int(movie.movie_id),
                'release_year': int(movie.release_year) if movie.release_year else 2000,
                'genres': genres_str,
            })
        
        for rating in session.query(RatingModel).all():
            ratings_data.append({
                'user_id': int(rating.user_id),
                'movie_id': int(rating.movie_id),
                'rating': parse_binary_int(rating.rating),
            })
        
        users_df = pd.DataFrame(users_data)
        movies_df = pd.DataFrame(movies_data)
        ratings_df = pd.DataFrame(ratings_data)
        
        print(f"\n用户数据: {len(users_df)} 条")
        print(f"电影数据: {len(movies_df)} 条")
        print(f"评分数据: {len(ratings_df)} 条")
        
    finally:
        session.close()
    
    # 构建增强的用户特征
    print("\n步骤3.1: 构建增强的用户特征（包含类型偏好）...")
    user_builder = EnhancedUserProfileBuilder()
    user_features_df, user_features_full = user_builder.build_user_features(users_df, ratings_df, movies_df)
    user_features_X = user_features_df.values.astype(float)
    print(f"  ✓ 用户特征矩阵形状: {user_features_X.shape}")
    
    # 特征降维
    print("\n步骤3.2: 用户特征降维...")
    user_features_reduced = user_builder.reduce_dimensionality(user_features_X)
    
    # 构建增强的电影特征
    print("\n步骤3.3: 构建增强的电影特征（包含类型特征）...")
    movie_builder = EnhancedMovieProfileBuilder()
    movie_features_df, movie_features_full = movie_builder.build_movie_features(movies_df, ratings_df)
    movie_features_X = movie_features_df.values.astype(float)
    print(f"  ✓ 电影特征矩阵形状: {movie_features_X.shape}")
    
    # 特征降维
    print("\n步骤3.4: 电影特征降维...")
    movie_features_reduced = movie_builder.reduce_dimensionality(movie_features_X)
    
    return (user_builder, movie_builder, user_features_X, movie_features_X,
            user_features_reduced, movie_features_reduced, user_features_full, movies_df, ratings_df)

# ===========================================================================
# 4. 用户聚类 (增强版)
# ===========================================================================

def cluster_users_enhanced(user_builder, user_features_reduced, n_clusters=None, auto_select=True):
    """对用户进行聚类 - 自动选择最佳聚类数"""
    print("\n" + "=" * 70)
    print("STEP 4: 增强的用户聚类")
    print("=" * 70)
    
    user_clusters = user_builder.build_user_profiles(
        user_features_reduced, 
        n_clusters=n_clusters,
        auto_select_clusters=auto_select,
        max_k=8
    )
    
    print(f"\n✓ 用户聚类完成！")
    print(f"  聚类数: {len(np.unique(user_clusters))}")
    print(f"  各簇用户分布:")
    for cluster_id in np.unique(user_clusters):
        count = np.sum(user_clusters == cluster_id)
        print(f"    簇 {cluster_id}: {count} 个用户")
    
    return user_clusters

# ===========================================================================
# 5. 电影聚类 (增强版)
# ===========================================================================

def cluster_movies_enhanced(movie_builder, movie_features_reduced, n_clusters=8):
    """对电影进行聚类"""
    print("\n" + "=" * 70)
    print(f"STEP 5: 增强的电影聚类 (k={n_clusters})")
    print("=" * 70)
    
    movie_clusters = movie_builder.build_movie_profiles(movie_features_reduced, n_clusters=n_clusters)
    
    print(f"\n✓ 电影聚类完成！")
    print(f"  聚类数: {len(np.unique(movie_clusters))}")
    print(f"  各簇电影分布:")
    for cluster_id in np.unique(movie_clusters):
        count = np.sum(movie_clusters == cluster_id)
        print(f"    簇 {cluster_id}: {count} 部电影")
    
    return movie_clusters

# ===========================================================================
# 6. 训练测试集划分
# ===========================================================================

def train_test_split_ratings(ratings_matrix, test_size=0.2):
    """划分训练集和测试集"""
    print("\n" + "=" * 70)
    print("STEP 6: 训练测试集划分")
    print("=" * 70)
    
    train_matrix = ratings_matrix.copy()
    test_matrix = np.full_like(ratings_matrix, np.nan)
    
    non_nan_indices = np.argwhere(~np.isnan(ratings_matrix))
    n_test = int(len(non_nan_indices) * test_size)
    
    np.random.seed(42)
    test_indices = non_nan_indices[np.random.choice(len(non_nan_indices), n_test, replace=False)]
    
    for idx in test_indices:
        i, j = idx
        test_matrix[i, j] = train_matrix[i, j]
        train_matrix[i, j] = np.nan
    
    print(f"\n训练集大小: {np.sum(~np.isnan(train_matrix))}")
    print(f"测试集大小: {np.sum(~np.isnan(test_matrix))}")
    
    return train_matrix, test_matrix

# ===========================================================================
# 7. 矩阵分解模型训练
# ===========================================================================

def train_matrix_factorization_model(train_ratings, val_ratings=None, n_factors=50, 
                                     learning_rate=0.005, reg_param=0.02, n_epochs=20):
    """训练矩阵分解模型（梯度下降）"""
    print("\n" + "=" * 70)
    print("STEP 7: 矩阵分解模型训练（梯度下降）")
    print("=" * 70)
    
    mf_model = MatrixFactorizationModel(
        n_factors=n_factors,
        learning_rate=learning_rate,
        reg_param=reg_param,
        n_epochs=n_epochs
    )
    
    mf_model.fit(train_ratings, val_ratings, verbose=True)
    
    print(f"\n✓ 模型训练成功！")
    print(f"  用户因子矩阵形状: {mf_model.user_factors.shape}")
    print(f"  电影因子矩阵形状: {mf_model.item_factors.shape}")
    print(f"  全局平均评分: {mf_model.global_bias:.4f}")
    
    return mf_model

# ===========================================================================
# 8. SVD推荐模型训练 (保留原有功能)
# ===========================================================================

def train_svd_recommendation_model(n_factors=50):
    """训练SVD推荐模型"""
    print("\n" + "=" * 70)
    print("STEP 8: SVD 推荐模型训练")
    print("=" * 70)
    
    persistence = PersistenceService()
    overview = persistence.get_system_overview()
    
    if overview['n_ratings'] == 0:
        print("✗ 数据库为空，无法训练模型")
        return None
    
    print(f"\n数据准备:")
    print(f"  用户数: {overview['n_users']}")
    print(f"  电影数: {overview['n_movies']}")
    print(f"  评分数: {overview['n_ratings']}")
    
    recommendation = RecommendationService(persistence, n_factors=n_factors)
    recommendation.train()
    
    if recommendation._is_trained:
        print("\n✓ 模型训练成功！")
        print(f"  用户因子矩阵形状: {recommendation._user_factors.shape}")
        print(f"  电影因子矩阵形状: {recommendation._movie_factors.shape}")
        print(f"  全局平均评分: {recommendation._global_mean:.4f}")
        return recommendation
    else:
        print("\n✗ 模型训练失败")
        return None

# ===========================================================================
# 9. 模型评估 (增强版)
# ===========================================================================

def evaluate_matrix_factorization_model(mf_model, test_ratings):
    """评估矩阵分解模型性能"""
    print("\n" + "=" * 70)
    print("STEP 9: 矩阵分解模型评估")
    print("=" * 70)
    
    predictions = []
    actuals = []
    
    test_indices = np.argwhere(~np.isnan(test_ratings))
    print(f"\n测试样本数: {len(test_indices)}")
    
    for idx in test_indices:
        i, j = idx
        pred = mf_model.predict(i, j)
        actual = test_ratings[i, j]
        predictions.append(float(pred))
        actuals.append(float(actual))
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # 计算评估指标
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)
    
    if len(actuals) > 1 and np.var(actuals) > 0:
        explained_variance = 1 - np.var(actuals - predictions) / np.var(actuals)
    else:
        explained_variance = 0
    
    predicted_rounded = np.round(predictions)
    accuracy = np.mean(predicted_rounded == actuals)
    
    high_rating_pred = (predictions >= 4)
    high_rating_actual = (actuals >= 4)
    precision = np.sum(high_rating_pred & high_rating_actual) / np.sum(high_rating_pred) if np.sum(high_rating_pred) > 0 else 0
    recall = np.sum(high_rating_pred & high_rating_actual) / np.sum(high_rating_actual) if np.sum(high_rating_actual) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics = {
        'rmse': rmse,
        'mae': mae,
        'explained_variance': explained_variance,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    print("\n✓ 评估完成！性能指标:")
    for metric, value in metrics.items():
        print(f"  {metric.upper()}: {value:.4f}")
    
    return metrics

# ===========================================================================
# 10. 新用户处理 (增强版)
# ===========================================================================

def handle_new_user_recommendations_enhanced(mf_model, user_features_X, user_ids):
    """处理新用户推荐 - 使用增强版新用户处理器（包含人口统计学-类型偏好映射）"""
    print("\n" + "=" * 70)
    print("STEP 10: 增强的新用户推荐处理")
    print("=" * 70)
    
    if user_features_X.shape[0] == 0:
        print("✗ 没有用户数据")
        return None
    
    # 获取用户基本特征（性别、年龄、职业）
    user_basic_features = user_features_X[:, :3] if user_features_X.shape[1] >= 3 else user_features_X
    
    # 初始化增强版新用户处理器（优化参数）
    persistence = PersistenceService()
    new_user_handler = EnhancedNewUserHandler(
        persistence=persistence,
        n_neighbors=15,
        min_similarity=0.2
    )
    
    # 训练增强版处理器（会自动构建人口统计学-类型偏好映射）
    new_user_handler.fit(user_basic_features)
    
    print("✓ 增强版新用户处理器已训练")
    print("  - 包含人口统计学-类型偏好映射")
    print("  - 使用余弦相似度查找相似用户")
    print("  - 支持混合推荐策略")
    
    # 测试新用户处理
    if len(user_basic_features) > 0:
        test_user_idx = 0
        test_user_features = user_basic_features[test_user_idx]
        
        # 获取测试用户信息
        session = persistence.db.get_session()
        try:
            test_user_id = user_ids[test_user_idx]
            test_user = session.query(UserModel).filter_by(user_id=test_user_id).first()
            if test_user:
                # 测试推荐功能
                gender_encoded = test_user_features[0]
                age = int(test_user_features[1])
                occupation_id = int(test_user_features[2])
                gender = 'M' if gender_encoded == 0 else 'F'
                
                similar_users = new_user_handler.find_similar_users(test_user_features)
                print(f"\n测试用户 {test_user_id} ({gender}, {age}岁, 职业{occupation_id}) 的相似用户 (Top 5):")
                for i, (user_id, score) in enumerate(similar_users[:5]):
                    print(f"  用户 {user_id}: 相似度 {score:.4f}")
                
                # 测试类型偏好
                genre_prefs = new_user_handler.get_genre_preferences(age, gender, occupation_id)
                if genre_prefs:
                    top_genres = sorted(genre_prefs.items(), key=lambda x: x[1], reverse=True)[:5]
                    print(f"\n  类型偏好 (Top 5):")
                    for genre, bias in top_genres:
                        print(f"    {genre}: {bias:+.3f}")
        finally:
            session.close()
    
    return new_user_handler

# ===========================================================================
# 主函数
# ===========================================================================

def main():
    """主函数 - 执行完整的模型训练流程"""
    print("\n" + "=" * 70)
    print("完整的电影推荐系统模型训练流程")
    print("结合 movie_system_simplified.py 和 mapping&recommend code.ipynb")
    print("=" * 70)
    
    # 设置中文字体
    setup_chinese_font()
    
    # STEP 1: 加载数据
    if not load_data_to_database():
        return
    
    # STEP 2: 构建评分矩阵
    ratings_matrix, user_ids, movie_ids = build_ratings_matrix()
    
    # STEP 3: 增强的特征工程
    (user_builder, movie_builder, user_features_X, movie_features_X,
     user_features_reduced, movie_features_reduced, user_features_full, movies_df, ratings_df) = \
        build_enhanced_user_and_movie_features()
    
    # STEP 4: 用户聚类（自动选择最佳聚类数）
    user_clusters = cluster_users_enhanced(user_builder, user_features_reduced, auto_select=True)
    
    # STEP 5: 电影聚类
    movie_clusters = cluster_movies_enhanced(movie_builder, movie_features_reduced, n_clusters=8)
    
    # STEP 6: 训练测试集划分
    train_ratings, test_ratings = train_test_split_ratings(ratings_matrix, test_size=0.2)
    
    # STEP 7: 矩阵分解模型训练（梯度下降）
    mf_model = train_matrix_factorization_model(
        train_ratings, 
        val_ratings=None,
        n_factors=50,
        learning_rate=0.005,
        reg_param=0.02,
        n_epochs=20
    )
    
    # STEP 8: SVD推荐模型训练（保留原有功能）
    svd_recommendation = train_svd_recommendation_model(n_factors=50)
    
    # STEP 9: 模型评估
    if mf_model:
        metrics = evaluate_matrix_factorization_model(mf_model, test_ratings)
    
    # STEP 10: 新用户处理
    if mf_model:
        new_user_handler = handle_new_user_recommendations_enhanced(mf_model, user_features_X, user_ids)
    
    # 完成
    print("\n" + "=" * 70)
    print("✓ 完整训练流程完成！")
    print("=" * 70)
    print("\n已完成的训练功能:")
    print("  ✓ 数据加载和数据库初始化")
    print("  ✓ 评分矩阵构建")
    print("  ✓ 增强的用户特征工程（包含类型偏好）")
    print("  ✓ 增强的电影特征工程（包含类型特征）")
    print("  ✓ 特征降维（PCA）")
    print("  ✓ 用户聚类（K-Means，自动选择最佳聚类数）")
    print("  ✓ 电影聚类（K-Means）")
    print("  ✓ 训练测试集划分")
    print("  ✓ 矩阵分解模型训练（梯度下降）")
    print("  ✓ SVD 推荐模型训练")
    print("  ✓ 模型性能评估（RMSE, MAE, 准确率等）")
    print("  ✓ 新用户相似度计算（KNN + 用户嵌入）")
    print("=" * 70)

if __name__ == "__main__":
    main()
