import pandas as pd
import pickle

def pickle_save(object, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(object, f)

def main():
    df = pd.read_csv(f'./data/recipe/filtered_data.csv', delimiter=',')
    recipes = set(df.loc[:, 'recipe_id'].tolist())
    df = pd.read_csv(f'./data/recipe/filtered_data.csv', delimiter=',')
    recipe_category = dict()
    categories = set()
    for i in recipes:
        category = df.loc[df.RecipeId == i, "Category"].tolist()
        if len(category) > 0:
            cat = category[0]
            recipe_category[i] = cat
            categories.add(cat)
    print(len(categories))
    pickle_save(recipe_category, f'./data/recipe/recipe_category.pkl')


if __name__ == '__main__':
    main()