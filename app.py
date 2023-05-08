import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


st.title('Welcome to Streamlit!')

st.title('Diamonds EDA nd Price Prediction App')
st.sidebar.header('User Input')

uploaded_file = st.sidebar.file_uploader('Upload DiamondsDataset (CSV Format)', type=['csv'])



if uploaded_file:
    diamonds_data = pd.read_csv(uploaded_file)
    st.write('### Dataset Preview')
    st.dataframe(diamonds_data.head(10))
    st.write('### Summary Statistics')
    summary_stats = diamonds_data.describe().T
    st.write(summary_stats)
    st.write('### Summary Statistics Table')
    st.table(summary_stats)

    st.sidebar.subheader('Visualization Options')
    columns_to_plot = st.sidebar.multiselect('Select Columns to Visualize:', diamonds_data.columns)

    if columns_to_plot:
        for column in columns_to_plot:
            fig, ax = plt.subplots()
            sns.histplot(data=diamonds_data, x=column, bins=30, ax=ax)
            ax.set_title(f'Histogram of {column}')
            st.pyplot(fig)
else:
    st.write('Please upload a dataset')





