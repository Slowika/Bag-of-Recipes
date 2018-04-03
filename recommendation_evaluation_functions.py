import numpy as np
import pandas as pd
import statistics


def recommend_ingredients(partial_recipes, user_item_matrix, k = 10, similarity_measure = "cosine", 
                          similarity_matrix = None, n_recommendations = 10, alpha = 0.2):
    """Recommend ingredients to (partial) recipes based on the similarity between ingredients.
    
    Inputs:
        - partial_recipes:    pandas dataframe of recipes that ingredient recommendations are produced for. Should be
                              of the shape recipes x ingredients.
                           
        - user_item_matrix:   pandas dataframe of training recipes. Should be of the shape recipes x ingredients.
        
        - k:                  number of neighbours (ingredients) used when calculating the ingredient 
                              recommendation scores.
        
        - similarity_measure: the measure used for calculating the similarity between ingredients. One of
                              'cosine', 'asymmetric_cosine', 'jaccard', 'pmi'.
                              
        - similarity_matrix:  the precomputed matrix of ingredient similarities. If not given, this will be
                              computed by the function.
                              
        - n_recommendations:  the desired number of recommended ingredients per recipe.
        
    Outputs a matrix of the recommended ingredients (columns) for the given partial recipes (rows).
        
    """
    
    # Calculate the similarity matrix if none was given as input.
    if np.all(similarity_matrix == None):
        
        if similarity_measure == "cosine":
            from sklearn.metrics.pairwise import cosine_similarity
            similarity_matrix = cosine_similarity(user_item_matrix.T)
            
        elif similarity_measure == "asymmetric_cosine":
            from similarity_functions import asymmetric_cosine
            similarity_matrix = asymmetric_cosine(user_item_matrix, alpha)
            
        elif similarity_measure == "jaccard":
            from similarity_functions import jaccard
            similarity_matrix = jaccard(user_item_matrix)
            
        elif similarity_measure == "pmi":
            from similarity_functions import pmi
            similarity_matrix = pmi(user_item_matrix)
        
        else: 
            raise ValueError("The similarity measure must be one of: 'cosine', 'asymmetric_cosine', 'jaccard', 'pmi'.")
    
    # Set similarity to self to zero.
    np.fill_diagonal(similarity_matrix, 0)     
    
    # For each ingredient, find the ingredients that are not among the k most similar and set similarity to zero.
    for i in range(np.shape(similarity_matrix)[0]):
        not_kNN = similarity_matrix[i, ] < similarity_matrix[i, np.argpartition(similarity_matrix[i, ], -k)[-k]]
        similarity_matrix[i, not_kNN] = 0

    # Calculate the ingredient scores.
    ingredient_scores = np.matmul(similarity_matrix, partial_recipes.T) / np.sum(abs(similarity_matrix), axis = 1)[:, None]
    ingredient_scores = ingredient_scores.T
    
    # Set ingredient scores of already present ingredients to zero.
    ingredient_scores[partial_recipes == 1] = 0
   
    # For each recipe, get the indices of the *n_recommendations* highest-scoring ingredients in order.
    recommendations_idx = np.argsort(-ingredient_scores, axis = 1)[:, :n_recommendations]
    
    # Convert recommendation indices to ingredient names.
    recommendations = user_item_matrix.columns[recommendations_idx]
    
    return recommendations



def held_out_recommendation(user_item_matrix, model_config = [10, "cosine", None, 10]):
    """Return a list of held out ingredients and a list of corresponding recommendations.
    
    """
    
    held_out_ingredients = []
    recommendations = []

    for index, row in user_item_matrix.iterrows():
        # Current training data: exclude the recipe tested
        X_curr = user_item_matrix.copy()
        X_curr.drop(index, inplace=True)
        
        # Current testing example: remove one ingredient
        recipe = row.copy()
        ing = recipe[recipe==1].sample(axis=0, random_state = 1).index.values[0]
        recipe[ing] = 0

        # Model tested
        k = model_config[0]
        similarity_measure = model_config[1]
        similarity_matrix = model_config[2]
        n_recommendations = model_config[3]
        
        # Get recommendations
        recommendation = recommend_ingredients(pd.DataFrame(recipe).T, X_curr, k, 
                                               similarity_measure, similarity_matrix, 
                                               n_recommendations)[0]
        
        # Store the removed ingredient and corresponding recommendations
        held_out_ingredients.append(ing)
        recommendations.append(recommendation)
        
    return (held_out_ingredients, recommendations)



def metric_1(missing_ingredients, recommendations):
    """Return the percentage of recipes for which the missing ingredient
    is among the top-10 recommended ingredients. (Mean Precision @ 10)
    
    """    
    
    matches = [1 for i in range(len(missing_ingredients)) if missing_ingredients[i] in recommendations[i][:10]]
    
    return len(matches)/len(missing_ingredients)



def metric_2(missing_ingredients, recommendations):
    """Mean rank of the missing ingredients in the list of recommended ingredients.
    
    """
    
    ranks = [np.where(missing_ingredients[i] == recommendations[i])[0][0]
             for i in range(len(missing_ingredients)) if missing_ingredients[i] in recommendations[i]]

    return sum(ranks)/len(ranks)



def metric_3(missing_ingredients, recommendations):
    """Median rank of the missing ingredients in the list of recommended ingredients.
    
    """
    
    ranks = [np.where(missing_ingredients[i] == recommendations[i])[0][0]
             for i in range(len(missing_ingredients)) if missing_ingredients[i] in recommendations[i]]

    return statistics.median(sorted(ranks))



def calculate_metrics(missing_ingredients, recommendations, model_config):
    """Calculate three evaluation metrics of recommendations made.
    
    Inputs:
        - missing_ingredients: list of the held-out ingredients.
        - recommendations: list of arrays with corresponding recommendations.
        - model_config: model settings used to make the recommendations.
        
    Outputs a dataframe with:
        - crucial model settings.
        - percentage of recipes for which the missing ingredient is among the top-10 recommended ingredients.
        - mean rank of the missing ingredients in the list of recommended ingredients.
        - median rank of the missing ingredients in the list of recommended ingredients.
        
    """
    
    # from recommendation_evaluation_functions import metric_1, metric_2, metric_3
    
    metrics = pd.DataFrame(columns = ["k", "similarity_measure", "top10_presence", "mean_rank", "median_rank"])
    metrics.loc[0, "k"]                  = model_config[0]
    metrics.loc[0, "similarity_measure"] = model_config[1]
    metrics.loc[0, "top10_presence"]     = metric_1(missing_ingredients, recommendations) 
    metrics.loc[0, "mean_rank"]          = metric_2(missing_ingredients, recommendations)
    metrics.loc[0, "median_rank"]        = metric_3(missing_ingredients, recommendations)
    
    return metrics