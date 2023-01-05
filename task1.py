import streamlit as st
import pandas as pd
import time
from streamlit_option_menu import option_menu
import plotly.express as px
import matplotlib
import numpy as np
from streamlit_lottie import st_lottie
st.set_page_config(page_title="Homework Streamlite", page_icon="chart_with_upwards_trend", layout="wide")

st.title("Streamlit")
mapping = {'Ankieta' : 0, 'Staty' : 1}

selected = option_menu(
       menu_title=None, # wymagane, None oznacza, że go po prostu nie będzie
       options = ['Ankieta', 'Staty'], # wymagane
       menu_icon = 'cast',
       default_index=0, # na której stronie będziemy na początku
       orientation='horizontal'
    )
with st.sidebar:
    selected = option_menu(
        menu_title='Main Menu', # wymagane
         options = ['Ankieta', 'Staty'], # wymagane,
         menu_icon = 'cast',
         default_index=mapping[selected] # na której stronie będziemy na początku
    )

if selected == 'Ankieta':
    st.title(f'You have selected {selected}')
    name = st.text_input("Please, enter your name", "Type here...")
    surname = st.text_input("Please, enter your surname", "Type here...")
    if st.button("Submit"):
        result = name.title() + " " + surname.title()
        st.success(result)
        message = st.success("kwestionariusz jest poprawny")

if selected == 'Staty':
    st.title(f'You have selected {selected}')
    data = st.file_uploader("Upload your dataset", type=['csv'])

    if data is not None:
        my_bar = st.progress(0)
        for p in range(100):
            time.sleep(0.01)
            my_bar.progress(p + 1)
        df = pd.read_csv(data)
        st.dataframe(df.head(10))



        st.set_option('deprecation.showPyplotGlobalUse', False)
        all_columns_names = list(df.select_dtypes(include=[np.number]).columns.values) #df.columns.to_list()
        selected_column_names = st.multiselect("Select numeric columns to plot", all_columns_names)
        selected_chart_types = st.multiselect("Select numeric columns to plot", ['bar', 'line', 'area'])
        plot_data = df[selected_column_names]

        if 'bar' in selected_chart_types:
            st.bar_chart(plot_data)

        if 'area' in selected_chart_types:
            st.area_chart(plot_data)

        if 'line' in selected_chart_types:
            st.line_chart(plot_data)

     