import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import plotly.express as px

# ------------------------------
# PAGE STYLE
# ------------------------------
st.set_page_config(page_title="Iris Classification", layout="wide")
st.markdown("""
<style>
    .main { background-color: #F8F9F9; }
    .title-text { color: #2E86C1; font-size: 40px; font-weight: bold; text-align: center; }
    .section-box { background-color: #FFFFFF; padding: 20px; border-radius: 12px; box-shadow: 2px 2px 10px #D5D8DC; }
    .sub-header { text-align:center; font-size: 20px; font-weight:bold; color:#1F618D; }
</style>
""", unsafe_allow_html=True)

# ------------------------------
# HEADER
# ------------------------------
st.markdown('<p class="title-text">โปรแกรมจำลองข้อมูลดอกไม้ Iris</p>', unsafe_allow_html=True)
st.image("./img/kim.jpeg", width=250)

# ------------------------------
# FLOWER IMAGES SECTION
# ------------------------------
st.markdown("<div class='sub-header'>ประเภทของดอกไม้ Iris</div>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.header("Versicolor")
    st.image("./img/iris1.jpg")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.header("Virginica")
    st.image("./img/iris2.jpg")
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.header("Setosa")
    st.image("./img/iris3.jpg")
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------
# STATISTICS SECTION
# ------------------------------
st.markdown("<div class='sub-header'>สถิติข้อมูลดอกไม้</div>", unsafe_allow_html=True)

dt = pd.read_csv("./data/iris.csv")
st.dataframe(dt.head(10), use_container_width=True)

# SUMMARY DATA
numeric_cols = ['petal.length','petal.width','sepal.length','sepal.width']
summary = dt[numeric_cols].sum().reset_index()
summary.columns = ["Feature", "Total"]

# show chart
if st.button("แสดงการจินตทัศน์ข้อมูล"):
    fig = px.bar(summary, x="Feature", y="Total", title="ภาพรวมข้อมูลดอกไม้ Iris", text_auto=True)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("ไม่แสดงข้อมูล")

# ------------------------------
# PREDICTION SECTION
# ------------------------------
st.markdown("<div class='sub-header'>ทำนายข้อมูลดอกไม้</div>", unsafe_allow_html=True)

pt_len = st.slider("เลือกข้อมูล petal.length", 0.0, 7.0, 1.0)
pt_wd = st.slider("เลือกข้อมูล petal.width", 0.0, 3.0, 1.0)
sp_len = st.number_input("เลือกข้อมูล sepal.length", 0.0, 10.0, 5.0)
sp_wd = st.number_input("เลือกข้อมูล sepal.width", 0.0, 5.0, 2.0)

if st.button("ทำนายผล"):
    X = dt.drop('variety', axis=1)
    y = dt['variety']

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X, y)

    x_input = np.array([[pt_len, pt_wd, sp_len, sp_wd]])
    prediction = model.predict(x_input)

    st.success(f"ผลการทำนาย: {prediction[0]}")

    if prediction[0] == 'Setosa':
        st.image("./img/iris1.jpg")
    elif prediction[0] == 'Versicolor':
        st.image("./img/iris2.jpg")
    else:
        st.image("./img/iris3.jpg")
else:
    st.info("ไม่ทำนาย")