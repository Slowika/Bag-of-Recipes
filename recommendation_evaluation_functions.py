import numpy as np
import pandas as pd
import statistics
from sklearn.decomposition import PCA


def recommend_ingredients(partial_recipes, user_item_matrix, k = 10, similarity_measure = "cosine", 
                          n_recommendations = 10, alpha = 0.05):
    """Recommend ingredients to (partial) recipes based on the similarity between ingredients.
    
    Inputs:
        - partial_recipes:    pandas dataframe of recipes that ingredient recommendations are produced for. Should be
                              of the shape recipes x ingredients.
                           
        - user_item_matrix:   pandas dataframe of training recipes. Should be of the shape recipes x ingredients.
        
        - k:                  number of neighbours (ingredients) used when calculating the ingredient 
                              recommendation scores.
        
        - similarity_measure: the measure used for calculating the similarity between ingredients. One of
                              'cosine', 'asymmetric_cosine', 'jaccard', 'pmi'.
                              
        - n_recommendations:  the desired number of recommended ingredients per recipe.
        
        - alpha:              tuning parameter for asymmetric cosine similarity.
        
    Outputs a matrix of the recommended ingredients (columns) for the given partial recipes (rows).
        
    """
    
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
    
    recommendations = {}
    
    # For each ingredient, find the ingredients that are not among the k most similar and set similarity to zero.
    if isinstance(k, int):
        k = [k]
    
    for elem in k:
        
        sim = similarity_matrix.copy()
        
        for i in range(np.shape(similarity_matrix)[0]):
            not_kNN = sim[i, ] < sim[i, np.argpartition(sim[i, ], -elem)[-elem]]
            sim[i, not_kNN] = 0

        # Calculate the ingredient scores.
        ingredient_scores = np.matmul(sim, partial_recipes.T) / np.sum(abs(sim), axis = 1)[:, None]
        ingredient_scores = ingredient_scores.T
    
        # Set ingredient scores of already present ingredients to zero.
        ingredient_scores[partial_recipes == 1] = 0
   
        # For each recipe, get the indices of the *n_recommendations* highest-scoring ingredients in order.
        recommendations_idx = np.argsort(-ingredient_scores, axis = 1)[:, :n_recommendations]
    
        # Convert recommendation indices to ingredient names.
        recommendations[elem] = user_item_matrix.columns[recommendations_idx]
    
    return recommendations


def held_out_recommendation(user_item_matrix, model_config=[10, "cosine", 10], usePCA = False, alpha = 0.2):
    """Return a list of held out ingredients and a list of corresponding recommendations.
    
    """
    
    held_out_ingredients = []
    recommendations      = {}
    
    if isinstance(model_config[0], int):
        model_config[0] = [model_config[0]]
    
    for k in model_config[0]:
        recommendations[k] = []
    
    # If PCA has to be applied on the user-item matrix
    if usePCA == True:
        n       = user_item_matrix.shape[0]
        pca     = PCA(n_components = n)
        X_pca_T = pca.fit_transform(user_item_matrix.T)
        X_curr  = pd.DataFrame(X_pca_T.T, columns = user_item_matrix.columns)
    
    for index, row in user_item_matrix.iterrows():
        
        # Current training data: exclude the recipe tested
        if usePCA == False:
            X_curr = user_item_matrix.copy()
            X_curr.drop(index, inplace=True)
        
        # Current testing example: remove one ingredient
        recipe      = row.copy()
        ing         = recipe[recipe==1].sample(axis=0, random_state = 1).index.values[0]
        recipe[ing] = 0 
        
        # Get recommendations
        dict_ = recommend_ingredients(pd.DataFrame(recipe).T, X_curr, model_config[0] , model_config[1],
                                               n_recommendations = model_config[2], alpha=alpha)
        
        # Append this list of recommendations for different recipes to the dictionary for with key k
        for k, recs in dict_.items():
            recommendations[k].append(recs[0])
        
        # Store the removed ingredient and corresponding recommendations
        held_out_ingredients.append(ing)
        
    return (held_out_ingredients, recommendations)



def metric_1(missing_ingredients, recommendations):
    """Return the percentage of recipes for which the missing ingredient
    is among the top-10 recommended ingredients (recall@10).
    
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



def calculate_metrics(missing_ingredients, recommendations, k, sim):
    """Calculate three evaluation metrics of recommendations made.
    
    Inputs:
        - missing_ingredients: list of the held-out ingredients.
        - recommendations: list of arrays with corresponding recommendations.
        - k: number of neighbours used to make the recommendations.
        - sim: similarity matrix used to make the recommendations.
        
    Outputs a dataframe with:
        - crucial model settings.
        - percentage of recipes for which the missing ingredient is among the top-10 recommended ingredients.
        - mean rank of the missing ingredients in the list of recommended ingredients.
        - median rank of the missing ingredients in the list of recommended ingredients.
        
    """    
    
    metrics = pd.DataFrame(columns = ["k", "similarity_measure", "top10_presence", "mean_rank", "median_rank"])
    metrics.loc[0, "k"]                  = k
    metrics.loc[0, "similarity_measure"] = sim
    metrics.loc[0, "top10_presence"]     = metric_1(missing_ingredients, recommendations) 
    metrics.loc[0, "mean_rank"]          = metric_2(missing_ingredients, recommendations)
    metrics.loc[0, "median_rank"]        = metric_3(missing_ingredients, recommendations)
    
    return metrics