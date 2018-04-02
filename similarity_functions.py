import numpy as np

def jaccard(X):

    """ Function to compute Jaccard similarity between two sets of elements
        In this case, we have a list of recipes that contain each ingredient in the pair
        ingredients we are evaluating

        Inputs:
            X: dataframe of recipes x ingredients
    """
    # Initialise count of pairs of ingredients that have zero recipes
    nullCount = 0

    # List to store ingredients with zero count
    problematicIngr = []

    # Create the set of recipes that contain each ingredient
    recipe_set = [ np.array(X.index[X[ingr] != 0]) for ingr in X.columns]

    # Initialise matrix of jaccard similarity
    jacs = np.zeros((X.shape[1], X.shape[1]))

    # Compute similarity for each pair of ingredients
    for ii, ingri in enumerate(X.columns):
        for ij, ingrj in enumerate(X.columns):
            if ij > ii:
                intersect = len(set.intersection(*[set(recipe_set[ii]), set(recipe_set[ij])]))
                union     = len(set.union(*[set(recipe_set[ii]), set(recipe_set[ij])]))

                if len(recipe_set[ii]) == 0 or len(recipe_set[ij])==0: pass

                # Pairs of ingredients with empty union
                if len(recipe_set[ii]) == 0 and ingri not in problematicIngr:
                    problematicIngr.append(ingri)
                    nullCount += 1
                if len(recipe_set[ij]) == 0 and ingrj not in problematicIngr:
                    problematicIngr.append(ingrj)
                    nullCount += 1

                jacs[ii, ij] = jacs[ij, ii] = intersect/float(union)

    if nullCount > 0:
        print("{:d} ingredients with zero frequency happened, similarities involving this ingredient = 0".format(nullCount))
        print(problematicIngr)

    # Empty diagonal
    np.fill_diagonal(jacs, 0)

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

    # Initialise count of pairs of ingredients that have zero recipes
    nullCount = 0

    # List to store ingredients with zero count
    problematicIngr = []

    # Compute similarity for each pair of ingredients
    for ii, ingri in enumerate(X.columns):
        for ij, ingrj in enumerate(X.columns):
            intersect = len(set.intersection(*[set(recipe_set[ii]), set(recipe_set[ij])]))
            denom     = len(recipe_set[ii])**alpha * len(recipe_set[ij])**(1-alpha)

            # Pairs of ingredients with empty union
            if denom == 0:
                if len(recipe_set[ii]) == 0 and ingri not in problematicIngr:
                    problematicIngr.append(ingri)
                    nullCount += 1
                if len(recipe_set[ij]) == 0 and ingrj not in problematicIngr:
                    problematicIngr.append(ingrj)
                    nullCount += 1
            else:
                asims[ii, ij] = intersect/ denom

    if nullCount > 0:
        print("{:d} ingredients with zero frequency happened, similarity 0 returned".format(nullCount))
        print(problematicIngr)

    # Empty diagonal
    np.fill_diagonal(asims, 0)

    return asims
