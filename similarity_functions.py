import numpy as np

def jaccard(X):

    """ Function to compute Jaccard similarity between two sets of elements
        In this case, we have a list of recipes that contain each ingredient in the pair
        ingredients we are evaluating

        Inputs:
            X: dataframe of recipes x ingredients
    """
    # Create the set of recipes that contain each ingredient
    recipe_set = [ np.array(X.index[X[ingr] != 0]) for ingr in X.columns]

    # Initialise matrix of asymmetric similarity
    jacs = np.zeros((X.shape[1], X.shape[1]))

    # Compute similarity for each pair of ingredients
    for ii, ingri in enumerate(X.columns):
        for ij, ingrj in enumerate(X.columns):
            if ij > ii:
                intersect = len(set.intersection(*[set(recipe_set[ii]), set(recipe_set[ij])]))
                union     = len(set.union(*[set(recipe_set[ii]), set(recipe_set[ij])]))
                jacs[ii, ij] = jacs[ij, ii] = intersect/float(union)

    return jacs

def asymmetric_cosine(X, alpha = 0.2):

    """ Function to compute assymetric cosine similarity between two sets of elements
        In this case, we have a list of recipes that contain each ingredient in the pair
        ingredients we are evaluating

        Inputs:
            X: dataframe of recipes x ingredients
    """
    # Create the set of recipes that contain each ingredient
    recipe_set = [ np.array(X.index[X[ingr] != 0]) for ingr in X.columns]

    # Initialise matrix of asymmetric similarity
    asims = np.zeros((X.shape[1], X.shape[1]))

    # Compute similarity for each pair of ingredients
    for ii, ingri in enumerate(X.columns):
        for ij, ingrj in enumerate(X.columns):
            intersect = len(set.intersection(*[set(recipe_set[ii]), set(recipe_set[ij])]))
            union     = len(set.union(*[set(recipe_set[ii]), set(recipe_set[ij])]))
            asims[ii, ij] = intersect/ ( len(recipe_set[ii])**alpha * len(recipe_set[ij])**(1-alpha) )

    np.fill_diagonal(asims, 0)

    return asims
