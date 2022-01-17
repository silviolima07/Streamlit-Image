import streamlit as st
import os


def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Selecione o dataset para avaliação', filenames)
    return os.path.join(folder_path, selected_filename)

filename = file_selector()
st.write('Dataset`%s`' % filename)

