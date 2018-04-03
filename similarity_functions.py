import numpy as np


def jaccard(X):
    """ Function to compute Jaccard similarity between two sets of elements.
        In this case, we have a list of recipes that contain each ingredient in the pair
        ingredients we are evaluating.

        Inputs:
            X: dataframe of recipes x ingredients
    """
    
    # Find the intersection of each ingredient pair (co-occurrences).
    inters = X.T.dot(X) 
    
    # Find the union of each ingredient pair (number of recipes with either a or b or both).
    union = np.add.outer(np.diag(inters), np.diag(inters)) - inters 
    
    # Check if any unions are zero.
    if np.any(union == 0):
        asims = np.zeros(X.shape[1])
        print("Some ingredients were not present in any recipe.")
    
    # Calculate Jaccard similarities.
    jacs = inters / union

    return np.array(jacs)



def asymmetric_cosine(X, alpha = 0.2):
    """ Function to compute asymmetric cosine similarity between two sets of elements
        In this case, we have a list of recipes that contain each ingredient in the pair
        ingredients we are evaluating.

        Inputs:
            X: dataframe of recipes x ingredients
    """
    
     # Find the intersection of each ingredient pair (co-occurrences).
    inters = X.T.dot(X)
    
    # Find the denominator (|U(i)|^alpha * |U(j)|^(1-alpha).
    denom = np.outer(np.diag(inters)**alpha, np.diag(inters)**(1-alpha))
    
    # Check if any unions are zero.
    if np.any(denom == 0):
        asims = np.zeros(X.shape[1])
        print("Some ingredients were not present in any recipe.")
    
    # Calculate asymmetric cosine similarities.
    asims = inters / denom

    return np.array(asims)



def pmi(X):
    """Calculate Pointwise Mutual Information (PMI) between all columns in a binary dataframe.
    
       Inputs:
           X: dataframe of recipes x ingredients
    """
    
    cooc = X.T.dot(X) / X.shape[0] # Get co-occurrence matrix.
    pmi = cooc / np.outer(np.diag(cooc), np.diag(cooc).T) # Calculate PMIs.
    pmi.values[[range(pmi.shape[0])]*2] = 0 # Set self-PMI to zero.
    
    return np.array(pmi)