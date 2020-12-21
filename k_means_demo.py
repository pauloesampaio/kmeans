from sklearn.cluster import KMeans
from utils.kmeans_utils import (
    open_image,
    resize_image,
    img_to_list,
    get_elbow_curve,
    create_cluster_image,
    plot_elbow_curve,
)
import streamlit as st

st.title("Image k-means demo")
k = st.slider("Number of clusters k", min_value=1, max_value=64, value=4, step=1)
path = st.text_input("Image URL", "Enter image url")

st.sidebar.title("Options")
add_legend = st.sidebar.checkbox(label="Display legend", value=False)
plot_elbow = st.sidebar.checkbox(label="Display elbow curve", value=False)

if path != "Enter image url":
    current_image = open_image(path, remote=True)
    resized_image = resize_image(current_image, max_side=128)
    w, h = resized_image.size[:2]
    current_image_hsv = resized_image.convert("HSV")
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(img_to_list(current_image_hsv))
    cluster_img = create_cluster_image(kmeans, (h, w), add_legend)
    col1, col2 = st.beta_columns(2)
    col1.header("Original")
    col1.image(resized_image, use_column_width=True)
    col2.header(f"{k} clusters reconstruction")
    col2.pyplot(cluster_img)
    if plot_elbow:
        elbow = get_elbow_curve(img_to_list(current_image_hsv))
        initial_inertia = elbow[0][1]
        chart = plot_elbow_curve(elbow)
        st.subheader("Elbow curve")
        st.altair_chart(chart, use_container_width=True)
