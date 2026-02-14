import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import warnings
warnings.filterwarnings('ignore')


class SephoraRecommenderSystem:
    """
    Hybrid Recommender System for Sephora Skincare Products
    
    This system combines:
    1. Collaborative Filtering (Matrix Factorization using SVD)
    2. Content-Based Filtering (TF-IDF on product attributes)
    3. Popularity-Based recommendations for cold start
    """
    def __init__(self, productsDf, reviewsDf):
        self.productsDf = productsDf.copy()
        self.reviewsDf = reviewsDf.copy()
        
        #model components
        self.userItemMatrix = None
        self.predictionMatrix = None
        self.contentSimilarityMatrix = None
        self.tfidfVectorizor=  None
        self.popularProducts = None
        
        #mapping
        self.userToIdx = {}
        self.idxToUser = {}
        self.productToIdx = {}
        self.idxToProduct = {}
        
        print(f"Product length = {len(self.productsDf)}")
        print(f"reviews Length = {len(self.reviewsDf)}")
        
        
    def preprocess_data(self):
        self.reviewsDf = self.reviewsDf.dropna(subset=['author_id', 'product_id', 'rating'])
        self.productsDf = self.productsDf.dropna(subset=['product_id'])
        
        self.reviewsDf['rating'] = pd.to_numeric(self.reviewsDf['rating'] , errors = 'coerce')
        self.reviewsDf = self.reviewsDf[self.reviewsDf['rating'].between(1,5)]
        
        #mapping
        uniqueUsers = self.reviewsDf['author_id'].unique()
        uniqueProd = self.reviewsDf['product_id'].unique()
        
        self.userToIdx = {user: idx for idx, user in enumerate(uniqueUsers)}
        self.idxToUser = {idx: user for user, idx in self.userToIdx.items()}
        self.productToIdx = {prod: idx for idx, prod in enumerate(uniqueProd)}
        self.idxToProduct = {idx: prod for prod, idx in self.productToIdx.items()}
        
        print(f"unique users = {len(uniqueUsers)}")
        print(f"unique products ={len(uniqueProd)}")
        
        #user item interaction matrix
        self.buildUserItemMatrix()
        
        #build content features
        self.buildContentFeatures()
        
        #calculate popular prod
        self.calculatePopularProducts()
            
            
    def buildUserItemMatrix(self):
        n_users = len(self.userToIdx)
        n_products = len(self.productToIdx)
        
        user_indices = self.reviewsDf['author_id'].map(self.userToIdx).values
        product_indices = self.reviewsDf['product_id'].map(self.productToIdx).values
        ratings = self.reviewsDf['rating'].values
        
        #creating sparse matrix
        self.userItemMatrix = csr_matrix(
            (ratings  ,(user_indices ,product_indices)),
            shape=(n_users , n_products)
        )
        
        sparcity = 1.0 - (self.userItemMatrix.nnz / (n_users*n_products))
        
        print(f"   Matrix shape: {self.userItemMatrix.shape}")
        print(f"   Sparsity: {sparcity:.4f} ({sparcity*100:.2f}%)")
        
    
    def buildContentFeatures(self):
        text_features = []
        
        for idx , row in self.productsDf.iterrows():
            features = []
            
            # Add product name (higher weight)
            if pd.notna(row.get('product_name')):
                features.append(str(row['product_name']) * 3)
            
            # Add brand (higher weight)
            if pd.notna(row.get('brand_name')):
                features.append(str(row['brand_name']) * 2)
            
            # Add category
            if pd.notna(row.get('primary_category')):
                features.append(str(row['primary_category']) * 2)
            
            if pd.notna(row.get('secondary_category')):
                features.append(str(row['secondary_category']))
            
            # Add ingredients (if available)
            if pd.notna(row.get('ingredients')):
                features.append(str(row['ingredients']))
            
            # Add skin type/concerns (if available)
            if pd.notna(row.get('skin_type')):
                features.append(str(row['skin_type']) * 2)
                
            text_features.append(' '.join(features))
            
            
        
        #tf-idf
        self.tfidfVectorizor = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1,2),
            stop_words='english',
            min_df=2
        )
        
        tfidfmatrix = self.tfidfVectorizor.fit_transform(text_features)
        
        self.contentSimilarityMatrix = linear_kernel(tfidfmatrix , tfidfmatrix)
        
        print(f" Content similarity matrix shape: {self.contentSimilarityMatrix.shape}")
        
        

    def calculatePopularProducts(self):
        productStats = self.reviewsDf.groupby('product_id').agg({'rating':['mean' , 'count']}).reset_index()
        
        productStats.columns= ['product_id', 'avg_rating', 'num_ratings']
        # Calculate weighted rating (IMDB formula)
        # WR = (v/(v+m)) * R + (m/(v+m)) * C
        # Where:
        # R = average rating
        # v = number of ratings
        # m = minimum ratings threshold
        # C = mean rating across all products
        
        m = productStats['num_ratings'].quantile(0.70)
        c = productStats['avg_rating'].mean()
        
        
        productStats['weighted_rating'] = (productStats['num_ratings']/(productStats['num_ratings']+m)* productStats['avg_rating']+(m/(productStats['num_ratings']+m))*c)
        
        self.popularProducts = productStats.merge(
            self.productsDf[['product_id', 'product_name', 'brand_name', 'price_usd','primary_category', 'variation_value']],
            on = 'product_id',
            how='left'
        ).sort_values('weighted_rating', ascending=False)
        
        print(f"   Popular products calculated: {len(self.popularProducts)}")
        
    
    def trainCollaborativeFiltering(self , n_factors = 50):
        userRatingMean = np.array(self.userItemMatrix.mean(axis=1)).flatten()
        matrix_centered = self.userItemMatrix.toarray() - userRatingMean.reshape(-1,1)
        
        #apply SVD
        U , sigma , Vt = svds(matrix_centered, k=n_factors)
        
        sigma = np.diag(sigma)
        
        self.predictionMatrix = np.dot(np.dot(U,sigma),Vt) + userRatingMean.reshape(-1,1)
        
        self.predictionMatrix = np.clip(self.predictionMatrix, 1,5)      
        print("Collaborative filtering model trained successfully!")
        print(f"Prediction matrix shape: {self.predictionMatrix.shape}")  
        
    
    def getCollaborativeRecommendations(self , userid , n_recommendations = 10):
        if userid not in self.userToIdx:
            return []
        
        userIDX = self.userToIdx[userid]
        userPredictions = self.predictionMatrix[userIDX, :]
        
        ratedProducts = set(self.reviewsDf[self.reviewsDf['author_id'] == userid]['product_id'].values)
        
        recommendations = []
        
        for prodIDX , predictedRatings in enumerate(userPredictions):
            productID = self.idxToProduct[prodIDX]
            
            if productID in ratedProducts:
                continue
            
            prodInfo = self.productsDf[self.productsDf['product_id'] == productID]
            prodName = prodInfo['product_name'].values[0] if len(prodInfo)>0 else 'Unknown'
            
            recommendations.append((productID , predictedRatings , prodName))
            
        recommendations.sort(key=lambda x:x[1], reverse=True)
        
        return recommendations[:n_recommendations]   
    
    
    def getContentBasedRecommendations(self , productID, n_recommendations = 10):
        prodcutIndices = self.productsDf[self.productsDf['product_id'] == productID].index
        
        if len(prodcutIndices)==0:
            return []
        
        productIDX = prodcutIndices[0]
        
        similarityScores = list(enumerate(self.contentSimilarityMatrix[productIDX]))
        
        similarityScores = sorted(similarityScores , key=lambda x:x[1] , reverse=True)
        
        recommendations = []
        
        for idx , score in similarityScores[1:n_recommendations+1]:
            similarProduct = self.productsDf.iloc[idx]
            
            recommendations.append((
                similarProduct['product_id'],
                score,
                similarProduct['product_name'],
                similarProduct['rating'],
                similarProduct['size'],
                similarProduct['brand_name'],
                similarProduct['price_usd'],
                similarProduct['primary_category']
                
                
            ))    
            
        return recommendations
    
    
    
    
    def getHybridRecommendations(self , userid, n_recommmendations=10, cf_weights=0.6 , cb_weights=0.4):
        cf_recs =self.getCollaborativeRecommendations(userid , n_recommmendations*2)
        
        if not cf_recs:
            return self.coldStartRecommendations(n_recommmendations)
        
        userReviews = self.reviewsDf[self.reviewsDf['author_id']==userid]
        userTopProducts = userReviews.nlargest(5, 'rating')['product_id'].values
        
        hybridScores = {}
        
        for productid , rating , name in cf_recs:
            normalizedRating = (rating-1)/4
            hybridScores[productid]={
                'cf_score': normalizedRating * cf_weights,
                'cb_score': 0,
                'name': name
            }
        
        for topProducts in userTopProducts:
            cbRecs = self.getContentBasedRecommendations(topProducts, n_recommmendations)
            
            for item in cbRecs:
                product_ID = item[0]
                similarity =item[1]
                name=item[1]
                if product_ID not in hybridScores:
                    hybridScores[product_ID] = {
                        'cf_score': 0,
                        'cb_score': 0,
                        'name': name
                    }

                hybridScores[product_ID]['cb_score'] += similarity * cb_weights/len(userTopProducts)
        
        
        result = []
        
        for product_id , scores , in hybridScores.items():
            finalScore = scores['cf_score'] + scores['cb_score']
            print(scores)
            productInfo = self.productsDf[self.productsDf['product_id'] == product_id]
            print(productInfo.info())
            if len(productInfo)>0:
                result.append({
                    'product_id': product_id,
                    'rating' :productInfo['rating'].values[0] if 'rating' in productInfo else 'Unknown',
                    'size' :productInfo['size'].values[0] if 'size' in productInfo else 'Unknown',
                    'loves_count' :productInfo['loves_count'].values[0] if 'loves_count' in productInfo else 'Unknown',
                    'product_name': productInfo['product_name'].values[0] if 'product_name' in productInfo else 'Unknown',
                    'brand': productInfo['brand_name'].values[0] if 'brand_name' in productInfo else 'Unknown',
                    'price': productInfo['price_usd'].values[0] if 'price_usd' in productInfo else 0,
                    'category':productInfo['primary_category'].values[0] if 'primary_category' in productInfo else 'Unknown',
                    'hybrid_score': finalScore,
                    'cf_score': scores['cf_score'],
                    'cb_score': scores['cb_score']
                })
        
        resultsDF = pd.DataFrame(result).sort_values('hybrid_score' , ascending=False)

        return resultsDF.head(n_recommmendations)
    
    
    
    def coldStartRecommendations(self, n_recommendations):
        return self.popularProducts.head(n_recommendations)[[
            'product_id', 'product_name', 'brand_name', 'price_usd',
            'avg_rating', 'num_ratings', 'weighted_rating' , 'primary_category', 'variation_value'
        ]].rename(columns={'brand_name': 'brand_name','price_usd': 'price'})
    
    
    
    
    
    
    def explainRecommendations(self , userid , productid):
        explanation = {'product_id': productid,'reasons': []}

    
        product_Info = self.productsDf[self.productsDf['product_id'] == productid]
        if len(product_Info)==0:
            return explanation
        
        product_name = product_Info['product_name'].values[0]
        explanation['product_name'] = product_name
        
        
        if userid in self.userToIdx:
            userReviews = self.reviewsDf[self.reviewsDf['author_id']== userid]
            
            if len(userReviews)>0:
                user_idx = self.userToIdx[userid]
                product_idx = self.productToIdx.get(productid)
                
                if product_idx is not None:
                    predicted_rating = self.predictionMatrix[user_idx , product_idx]    
                    explanation['reasons'].append(
                        f"Users with similar preferences rated this {predicted_rating:.1f}/5.0"
                    )
                
                user_top_products = userReviews.nlargest(3, 'rating')
                
                for _ , review in user_top_products.iterrows():
                    top_product_id = review['product_id']
                    cb_recs = self.getContentBasedRecommendations(top_product_id, 20)
                    
                    for rec_id, similarity, _ in cb_recs:
                        if rec_id == productid:
                            top_product_name = self.productsDf[
                                self.productsDf['product_id'] == top_product_id
                            ]['product_name'].values[0]
                            
                            explanation['reasons'].append(
                                f"Similar to '{top_product_name}' which you rated {review['rating']:.1f}/5.0 "
                                f"(similarity: {similarity:.2f})"
                            )
                            break
        
        productStats = self.popularProducts[self.popularProducts['product_id'] == productid]
        
        if len(productStats) > 0:
            avg_rating = productStats['avg_rating'].values[0]
            num_ratings = int(productStats['num_ratings'].values[0])
            
            explanation['reasons'].append(
                f"Highly rated by the community: {avg_rating:.1f}/5.0 based on {num_ratings} reviews"
            )
        
        return explanation
    
    
    
    
    
    
    
    
    def getUserProfile(self, userid):
        
        if userid not in self.userToIdx:
            return None
        
        userReviews = self.reviewsDf[self.reviewsDf['author_id'] == userid]
        
        if len(userReviews) == 0:
            return None
        
        # Calculate statistics
        profile = {
            'user_id': userid,
            'total_reviews': len(userReviews),
            'avg_rating_given': userReviews['rating'].mean(),
            'min_rating': userReviews['rating'].min(),
            'max_rating': userReviews['rating'].max(),
            'rating_std': userReviews['rating'].std()
        }
        
        # Get favorite products (rated 4 or 5)
        # print("troubleshooting brand")
        # print(self.productsDf.head())
        favorite_products = userReviews[userReviews['rating'] >= 4].merge(
            self.productsDf[['product_id', 'product_name', 'brand_name', 'primary_category']],
            on='product_id',
            how='left',
            suffixes=('', '_product')
        )
        
        if len(favorite_products) > 0:
            # Favorite brands
            brand_counts = favorite_products['brand_name_product'].value_counts()
            profile['favorite_brands'] = brand_counts.head(3).to_dict()
            
            # Favorite categories
            category_counts = favorite_products['primary_category'].value_counts()
            profile['favorite_categories'] = category_counts.head(3).to_dict()
            
            # Top rated products
            top_products = favorite_products.nlargest(5, 'rating')[
                ['product_name', 'brand_name', 'rating']
            ].to_dict('records')
            profile['top_rated_products'] = top_products
        
        return profile
    
    
    
    
class dataloder:
    def load_sephora_data(products_path, reviews_path):
        try:
            products_df = pd.read_csv(products_path)
            print(f"✓ Products loaded: {len(products_df)} rows")
            
            reviews_df = pd.read_csv(reviews_path)
            print(f"✓ Reviews loaded: {len(reviews_df)} rows")
            
            
            print("DATA VALIDATION")
            
            required_product_cols = ['product_id', 'product_name', 'brand_name']
            missing_product_cols = [col for col in required_product_cols if col not in products_df.columns]
            if missing_product_cols:
                print(f"⚠ Warning: Missing product columns: {missing_product_cols}")
            else:
                print(f"✓ All required product columns present")
            
            
            # Check required columns for reviews
            required_review_cols = ['author_id', 'product_id', 'rating']
            missing_review_cols = [col for col in required_review_cols if col not in reviews_df.columns]
            if missing_review_cols:
                print(f"⚠ Warning: Missing review columns: {missing_review_cols}")
            else:
                print(f"✓ All required review columns present")
                
                
                
            print("\n" + "-" * 80)
            print("DATA STATISTICS")
            print("-" * 80)
            print(f"Unique products in catalog: {products_df['product_id'].nunique()}")
            print(f"Unique products with reviews: {reviews_df['product_id'].nunique()}")
            print(f"Unique users (reviewers): {reviews_df['author_id'].nunique()}")
            print(f"Rating range: {reviews_df['rating'].min()} - {reviews_df['rating'].max()}")
            print(f"Average rating: {reviews_df['rating'].mean():.2f}")
            print(f"Total reviews: {len(reviews_df)}")
            
            n_users = reviews_df['author_id'].nunique()
            n_products = reviews_df['product_id'].nunique()
            n_interactions = len(reviews_df)
            sparsity = 1 - (n_interactions / (n_users * n_products))
            print(f"Matrix sparsity: {sparsity:.4f} ({sparsity*100:.2f}%)")
            
            print("\n✓ Data loaded successfully!")
            print("=" * 80)
            
            
            return products_df, reviews_df
            
        except FileNotFoundError as e:
            print(f"\n❌ Error: File not found - {e}")
            print("\nPlease make sure the file paths are correct:")
            print(f"  - Products: {products_path}")
            print(f"  - Reviews: {reviews_path}")
            raise
        except Exception as e:
            print(f"\n❌ Error loading data: {e}")
            raise






class recommenderclass:
    def get_user_recommendations(recommender, userid, n_recommendations=10, recommendation_type='hybrid'):
        print(f"GETTING RECOMMENDATIONS FOR USER: {userid}")
        print("=" * 80)
        print(f"Recommendation type: {recommendation_type}")
        print(f"Number of recommendations: {n_recommendations}")
        print()
        
        user_exists = userid in recommender.userToIdx
        #print(recommender.userToIdx)
        if not user_exists and recommendation_type != 'popular':
            print(f"⚠ Warning: User '{userid}' not found in the system!")
            print("  → Using popular products instead (cold start)")
            return "Unknown"
            #recommendation_type = 'popular'
        
        
        if user_exists:
            profile = recommender.getUserProfile(userid)
            if profile:
                print(f"Total reviews: {profile['total_reviews']}")
                print(f"Average rating given: {profile['avg_rating_given']:.2f}")
                if 'favorite_brands' in profile:
                    print(f"Favorite brands: {list(profile['favorite_brands'].keys())}")
                if 'favorite_categories' in profile:
                    print(f"Favorite categories: {list(profile['favorite_categories'].keys())}")
                print()
                
        
        if recommendation_type == 'hybrid':
            recommendations = recommender.getHybridRecommendations(userid, n_recommendations)
            
        elif recommendation_type == 'collaborative':
            recs_list = recommender.getCollaborativeRecommendations(userid, n_recommendations)
            # Convert to DataFrame
            recommendations = pd.DataFrame([
                {
                    'product_id': pid,
                    'predicted_rating': rating,
                    'product_name': name
                }
                for pid, rating, name in recs_list
            ])
            
        
        elif recommendation_type == 'content':
            # Get user's top product and find similar items
            user_reviews = recommender.reviewsDf[recommender.reviewsDf['author_id'] == userid]
            if len(user_reviews) > 0:
                top_product = user_reviews.nlargest(1, 'rating')['product_id'].values[0]
                recs_list = recommender.getContentBasedRecommendations(top_product, n_recommendations)
                recommendations = pd.DataFrame([
                    {
                        'product_id': pid,
                        'similarity_score': score,
                        'product_name': name
                    }
                    for pid, score, name in recs_list
                ])
            else:
                recommendations = recommender.coldStartRecommendations(n_recommendations)
                
                
        elif recommendation_type == 'popular':
            recommendations = recommender.coldStartRecommendations(n_recommendations)
            
            
        else:
            raise ValueError(f"Invalid recommendation_type: {recommendation_type}. "
                            f"Must be 'hybrid', 'collaborative', 'content', or 'popular'")
        
        print(recommendations.to_string(index=False))
        
        
        return recommendations
    
    
    
    def get_similar_products(recommender, product_id, n_recommendations=10):
        print(f"FINDING SIMILAR PRODUCTS")
        
        product_info = recommender.productsDf[recommender.productsDf['product_id'] == product_id]
        if len(product_info) == 0:
            print(f"❌ Error: Product '{product_id}' not found!")
            return []
        
        product_name = product_info['product_name'].values[0]
        product_brand = product_info['brand_name'].values[0]
        
        print(f"Base product: {product_name}")
        print(f"Brand: {product_brand}")
        print()
        similar = recommender.getContentBasedRecommendations(product_id, n_recommendations)
        
        print("-" * 80)
        print(f"TOP {n_recommendations} SIMILAR PRODUCTS")
        print("-" * 80)
        print(similar[0])
        # for i, (pid, similarity, name) in enumerate(similar, 1):
        #     print(f"{i}. {name}")
        #     print(f"   Similarity: {similarity:.3f}")
        #     print()
        
        print("=" * 80)
        
        return similar