import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch.nn as nn
#import pytorch_lightning as pl
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import pairwise_distances
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from wordcloud import WordCloud
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from PIL import Image

st.set_page_config(
    page_title="Custom Streamlit Styling",
    page_icon=":smiley:",
    layout="centered",
    initial_sidebar_state="expanded",
)
st.set_option('deprecation.showPyplotGlobalUse', False)

def Preprocessing(behaviour_data):
    behaviour_data['click_history'].fillna('', inplace=True)
    
    ## Indexize users
    uniqueIds = behaviour_data['userId'].unique()

    ind2user = {idx +1: itemid for idx, itemid in enumerate(uniqueIds)}
    user2ind = {itemid : idx for idx, itemid in ind2user.items()}
    print(f"We have {len(user2ind)} unique users in the dataset")

    # Create a new column with userIdx:
    behaviour_data['userIdx'] = behaviour_data['userId'].map(lambda x: user2ind.get(x,0))
    return behaviour_data


def home():
    st.title("Article Recommendation")
    st.write("By 21PD22 - Nilavini")
    st.write("  21PD27 - Raja Neha")
    st.write("  21PD39 - Varsha")
    st.header("About the dataset")
    
    behaviour_data = pd.read_csv('C:\\Users\\birth\\ML package\\behaviors.tsv', sep="\t",names=["impressionId","userId","timestamp","click_history","impressions"])
    st.write("Behaviors data")
    st.write(behaviour_data)
    st.write("Data after Preprocessing")
    behaviour_data = Preprocessing(behaviour_data)
    st.write(behaviour_data)
    
    
    news = pd.read_csv('C:\\Users\\birth\\ML package\\news.tsv', sep="\t",names=["itemId","category","subcategory","title","abstract","url","title_entities","abstract_entities"])
    # Build index of items
    ind2item = {idx +1: itemid for idx, itemid in enumerate(news['itemId'].values)}
    item2ind = {itemid : idx for idx, itemid in ind2item.items()}
    st.write("News data")
    st.write(news)
    
    news.fillna('', inplace=True)
    
    data=behaviour_data
    def process_click_history(s):
        list_of_strings = str(s).split(" ")
        return [item2ind.get(l, 0) for l in list_of_strings]

    data['click_history_idx'] = behaviour_data.click_history.map(lambda s:  process_click_history(s))
    
    def process_impression(s):
        list_of_strings = s.split(" ")
        itemid_rel_tuple = [l.split("-") for l in list_of_strings]
        noclicks = []
        for entry in itemid_rel_tuple:
            if entry[1] =='0':
                noclicks.append(entry[0])
            if entry[1] =='1':
                click = entry[0]
        return noclicks, click

    data['noclicks'], data['click'] = zip(*behaviour_data['impressions'].map(process_impression))
    # We can then indexize these two new columns:
    data['noclicks'] =data['noclicks'].map(lambda list_of_strings: [item2ind.get(l, 0) for l in list_of_strings])
    data['noclick'] = data['noclicks'].map(lambda x : x[0])
    data['click'] = data['click'].map(lambda x: item2ind.get(x,0))
    
    data['epochhrs'] = pd.to_datetime(data['timestamp']).values.astype(np.int64)/(1e6)/1000/3600
    data['epochhrs'] = data['epochhrs'].round()
    data[['click','epochhrs']].groupby("click").min("epochhrs").reset_index()
    
    dataR =data[['epochhrs','userIdx','click_history_idx','noclick','click']]
    
    temp = data[['epochhrs','userIdx','noclick','click']]
    
    X_temp=X_temp = temp.drop('click', axis=1)
    y_temp=temp['click']
    # Assuming 'X_temp' is your feature matrix and 'y_temp' is your target variable
    selector = SelectKBest(score_func=f_classif, k=2)  # Select the top 2 features using ANOVA F-value
    X_temp_selected = selector.fit_transform(X_temp, y_temp)

    # Get the selected feature indices
    selected_indices = selector.get_support(indices=True)

    # Extract the selected feature names
    selected_features_content_based = X_temp.columns[selected_indices]
    st.write('The number of articles before processing :',len(news))
    news.drop_duplicates(subset=['title'],inplace=True)
    st.write('The number of articles after processing :',len(news))
    
    X_news = news[['category', 'subcategory', 'title', 'abstract', 'title_entities', 'abstract_entities']]
    y_user = behaviour_data['click_history']
    
    df2=news.copy()
    
    st.header("Bagging method")
    headline_vectorizer = TfidfVectorizer()
    headline_features = headline_vectorizer.fit_transform(news['title'].values)

    def Euclidean_Distance_based_model(row_index, num_similar_items):
        cate = news['category'][row_index]
        name = news['title'][row_index]
        cate_data = news[news['category'] == cate]

        row_index2 = cate_data[cate_data['title'] == name].index
        couple_dist = pairwise_distances(headline_features, headline_features[row_index2])

        num_similar_items = min(num_similar_items, len(cate_data) - 1)

        indices = np.argsort(couple_dist.ravel())[:num_similar_items]

        indices = [i for i in indices if i < len(cate_data)]

        df = pd.DataFrame({
            'headline': cate_data['title'].values[indices],
            'Category': cate_data['category'].values[indices],
            'Abstract': cate_data['abstract'].values[indices],
            'Euclidean similarity with the queried article': couple_dist[indices].ravel()
        })
        st.write("=" * 30, "News Article Name", "=" * 30)
        st.write('News Headline : ', name)
        st.write("\n", "=" * 30, "Recommended News : ", "=" * 30)
        return df.iloc[1:, :]

    name = st.text_input("News Title For Recommendation:")
    if st.button('Enter'):
        ind = news[news['title'] == name].index[0]
        dd = Euclidean_Distance_based_model(ind, min(10, len(news)))  # Adjust the number of similar items as needed
        st.write(dd.head(10))
    
    st.header("TF-IDF")
    
    tfidf_headline_vectorizer = TfidfVectorizer(min_df = 0)
    
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(data['click_history'])

    from tabulate import tabulate

    def TFIDF_based_model(row_index, num_similar_items):
        user_data = data.iloc[row_index]
        user_click_history = user_data['click_history']

        couple_dist = pairwise_distances(tfidf_matrix, tfidf_matrix[user_data.name])
        indices = np.argsort(couple_dist.ravel())[0:num_similar_items]

        recommended_articles = data.iloc[indices]

        st.write("=" * 30, "User Click History", "=" * 30)
        st.write('User Click History : ', user_click_history)
        st.write("\n", "=" * 26, "Recommended Articles Using TF-IDF: ", "=" * 26)

        table_data = recommended_articles[['impressionId', 'userId', 'timestamp', 'click_history', 'impressions']]
        st.text(tabulate(table_data, headers='keys', tablefmt='fancy_grid'))

    user_idx = st.number_input("Enter the user id", value=None, placeholder="Type the user id...")
    
    if st.button('Search'):
        user_idx = int(user_idx)
        recommended_articles = TFIDF_based_model(user_idx, 10)
    
    def TFIDF_based_model(row_index, num_similar_items):
        article_data = news.iloc[row_index]
        article_title = article_data['title']
        category_data = news[news['category'] == article_data['category']]

        row_index2 = category_data[category_data['title'] == article_title].index
        headline_features = tfidf_vectorizer.transform(category_data['title'].values)
        couple_dist = pairwise_distances(headline_features, headline_features[row_index2])
        indices = np.argsort(couple_dist.ravel())[0:num_similar_items]

        recommended_articles = category_data.iloc[indices]

        st.write("=" * 30, "Article Title", "=" * 30)
        st.write('Article Title : ', article_data['title'])
        st.write("\n", "=" * 26, "Recommended Articles Using TF-IDF: ", "=" * 26)

        table_data = recommended_articles[['title', 'category', 'abstract', 'url']]
        st.text(tabulate(table_data, headers='keys', tablefmt='fancy_grid'))

    article_idx = st.number_input("Enter the article id", value=None, placeholder="Type the article id...")
    
    if st.button('Go'):
        article_idx = int(article_idx)
        TFIDF_based_model(article_idx, 10)
    
    st.header("Category Recommendation System")
    
    X=news['title'].values
    y=news['category'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21)

    tfidf_vectorizer = TfidfVectorizer(min_df=4)

    svd = TruncatedSVD(n_components=400, n_iter=8, random_state=42)
    dt_classifier = DecisionTreeClassifier()

    pipeline = make_pipeline(tfidf_vectorizer, svd, dt_classifier)

    pipeline.fit(X_train, y_train)

    def print_report(pipe):
        y_pred = pipe.predict(X_test)
        labels = np.unique(y)
        report = metrics.classification_report(y_test, y_pred, target_names=labels)
        st.text(report)
        st.text("Accuracy: {:0.3f}".format(metrics.accuracy_score(y_test, y_pred)))


    print_report(pipeline)

    st.header("Pipeline using TruncatedSVD and XGBClassifier")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21)

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    tfidf_vectorizer = TfidfVectorizer()

    svd = TruncatedSVD(n_components=10, n_iter=1, random_state=42)

    xgb_classifier = XGBClassifier()

    pipeline = make_pipeline(tfidf_vectorizer, svd, xgb_classifier)
    

    pipeline.fit(X_train, y_train_encoded)

    y_pred_encoded = pipeline.predict(X_test)
    report = classification_report(y_test_encoded, y_pred_encoded)
    st.text(report)
    
    st.header("Pipeline using TruncatedSVD and LGBMClassifier")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21)

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    tfidf_vectorizer = TfidfVectorizer()

    svd = TruncatedSVD(n_components=100, n_iter=1, random_state=42)

    lgbm_classifier = LGBMClassifier()

    pipeline = make_pipeline(tfidf_vectorizer, svd, lgbm_classifier)

    pipeline.fit(X_train, y_train_encoded)


    y_pred_encoded = pipeline.predict(X_test)
    report = classification_report(y_test_encoded, y_pred_encoded)
    st.text(report)
            
def main():
    
    home()
        

if __name__ == "__main__":
    
    main()