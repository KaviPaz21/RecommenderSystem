from fastapi import FastAPI,HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from recommendersys import SephoraRecommenderSystem , dataloder ,recommenderclass
import numpy as np
import pandas as pd

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# A simple Pydantic model for our POST request
class Item(BaseModel):
    userid: str

PRODUCTS_PATH = 'data/product_info.csv'
REVIEWS_PATH = 'data/reviews_250-500.csv'

products_df, reviews_df = dataloder.load_sephora_data(PRODUCTS_PATH, REVIEWS_PATH)

recommender = SephoraRecommenderSystem(products_df, reviews_df)
recommender.preprocess_data()
recommender.trainCollaborativeFiltering(n_factors=50)

# GET Endpoint: Returns a simple message
@app.get("/")
def read_root():
    return {"message": "Welcome to my FastAPI server!"}






@app.post("/user-profile")
def get_user_profile(request: Item):
    user_id = request.userid
    user_df = reviews_df[reviews_df["author_id"] == user_id]

    # If user not found
    if user_df.empty:
        return {
        "user_id": user_id,
        "love_count": 0,
        "total_reviews": 0,
        "top_categories": [],
        "top_brands": []
    }

    # Love count (sum of positive feedback)
    love_count = user_df["rating"].fillna(0).sum()

    # Total reviews
    total_reviews =  user_df["total_feedback_count"].fillna(0).sum()

    # Top 6 preferred primary categories
    top_categories = (
        user_df["product_name"]
        .dropna()
        .value_counts()
        .head(6)
        .index
        .tolist()
    )

    # Top 6 preferred brands
    top_brands = (
        user_df["brand_name"]
        .dropna()
        .value_counts()
        .head(6)
        .index
        .tolist()
    )

    return {
        "user_id": user_id,
        "love_count": int(love_count),
        "total_reviews": int(total_reviews),
        "top_categories": top_categories,
        "top_brands": top_brands
    }





# POST Endpoint: Receives JSON data and returns it
@app.post("/Recommendations/userchoice")
def create_item(item: Item):
    userid = item.userid
    hybrid_recs = recommenderclass.get_user_recommendations(
    recommender,
    userid=userid,
    n_recommendations=10,
    recommendation_type='hybrid'
    )
    if isinstance(hybrid_recs, str):
        if hybrid_recs == "Unknown":
            return {"userid": userid, "recommendations": []}
        
    hybrid_recs = hybrid_recs.replace([np.inf, -np.inf], np.nan)
    hybrid_recs = hybrid_recs.fillna(0)
    recommendations_json = hybrid_recs.to_dict(orient="records")
    
    return {"message":"ok", "items":recommendations_json}




@app.post("/Recommendations/toppicks")
def gettoppicks(item: Item):
    userid = item.userid
    hybrid_recs = recommenderclass.get_user_recommendations(
    recommender,
    userid=userid,
    n_recommendations=10,
    recommendation_type='popular'
    )
    if isinstance(hybrid_recs, str):
        if hybrid_recs == "Unknown":
            return {"userid": userid, "recommendations": []}
        
    hybrid_recs = hybrid_recs.replace([np.inf, -np.inf], np.nan)
    hybrid_recs = hybrid_recs.fillna(0)
    recommendations_json = hybrid_recs.to_dict(orient="records")
    
    return {"message":"ok", "items":recommendations_json}


import math
def sanitize_float(val):
    try:
        val = float(val)
        if math.isnan(val) or math.isinf(val):
            return 0.0
        return val
    except:
        return 0.0

@app.post("/Recommendations/similaritems")
def gettoppicks(item: Item):
    userid = item.userid

    # Call your recommender
    similar_products = recommenderclass.get_similar_products(
        recommender,
        product_id=userid,
        n_recommendations=20
    )

    # If recommender returns a string (error)
    if isinstance(similar_products, str):
        return {"message": "ok", "items": []}

    formatted_items = []
    
    # Iterate safely over recommender output
    for prod in similar_products:
        # Ensure it has at least 8 elements
        if not isinstance(prod, (list, tuple)) or len(prod) < 8:
            continue

        pid, similarity, product_name, rating, size, brand, price, attribute = prod

        formatted_items.append({
            "product_id": str(pid),
            "similarity": sanitize_float(similarity),
            "product_name": str(product_name),
            "avg_rating": sanitize_float(rating),
            "variation_value": str(size),
            "brand_name": str(brand),
            "price": sanitize_float(price),
            "primary_category": str(attribute)
        })

    return {"message": "ok", "items": formatted_items}