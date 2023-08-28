import pickle
import streamlit as st
import numpy as np 

st.header('Books Recomendations using Machine Learning')
st.subheader('Original code from: DSwithBappy. \n Tutorial: https://www.youtube.com/playlist?list=PLkz_y24mlSJa37r2xNDyEgt0Z4ilHtJ07')

model = pickle.load(open('artifacts/model.pkl','rb'))
books_name= pickle.load(open('artifacts/books_name.pkl','rb'))
final_rating=pickle.load(open('artifacts/final_ranting.pkl', 'rb'))
book_pivot= pickle.load(open('artifacts/book_pivot.pkl','rb'))

def fecth_poster(suggestion):
    books_name=[]
    ids_index=[]
    poster_url =[]

    for book_id in suggestion:
        books_name.append(book_pivot.index[book_id])

    for name in books_name[0]:
        ids = np.where(final_rating['title'] == name)[0][0]
        ids_index.append(ids)

    for idx in ids_index:
        url = final_rating.iloc[idx]['img_url']
        poster_url.append(url)

    return poster_url


def recommend_books(book_name):
    book_list=[]
    book_id=np.where(book_pivot.index== book_name)[0][0]
    distance, suggestion = model.kneighbors(book_pivot.iloc[book_id,:].values.reshape(1,-1), n_neighbors=6)

    poster_url = fecth_poster(suggestion)

    for i in range(len(suggestion)):
        books= book_pivot.index[suggestion[i]]
        for j in books:
            book_list.append(j)

    return book_list,poster_url

selected_books = st.selectbox(
    'Type or select a book',
    books_name
)

if st.button('Show Reccommedation'):
    recommendation_books, poster_url = recommend_books(selected_books)
    tab1,tab2,tab3,tab4,tab5 = st.tabs(['First','Second','Third','Fourth','Fifth'])

    with tab1:
        st.title(recommendation_books[1])
        st.image(poster_url[1])

    with tab2:
        st.title(recommendation_books[2])
        st.title(poster_url[2])

    with tab3:
        st.title(recommendation_books[3])
        st.image(poster_url[3])
    
    with tab4:
        st.title(recommendation_books[4])
        st.image(poster_url[4])

    with tab5:
        st.title(recommendation_books[5])
        st.image(poster_url[5])

st.caption('Dataset: https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset')