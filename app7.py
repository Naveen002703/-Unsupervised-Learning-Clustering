import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="K-Means Clustering", layout="wide")

st.title("🔵 Lab Program 7: Unsupervised Learning – Clustering")
st.subheader("K-Means Clustering with Elbow Method")

# ------------------------------------------------
# Upload Dataset
# ------------------------------------------------
st.sidebar.header("Upload Dataset")
file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

if file is not None:
    df = pd.read_csv(file)

    st.subheader("📌 Dataset Preview")
    st.write(df.head())
    st.write("Shape:", df.shape)

    # ------------------------------------------------
    # Select Numeric Features
    # ------------------------------------------------
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    st.subheader("🎯 Select Features for Clustering")

    selected_features = st.multiselect(
        "Choose Numeric Features",
        numeric_cols,
        default=numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols
    )

    if len(selected_features) >= 2:

        X = df[selected_features]

        # ------------------------------------------------
        # Standardize Data
        # ------------------------------------------------
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # ------------------------------------------------
        # Elbow Method
        # ------------------------------------------------
        st.subheader("📉 Elbow Method")

        k_range = range(1, 11)
        inertia = []

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertia.append(kmeans.inertia_)

        fig1, ax1 = plt.subplots()
        ax1.plot(k_range, inertia, marker='o')
        ax1.set_xlabel("Number of Clusters (K)")
        ax1.set_ylabel("Inertia (WCSS)")
        ax1.set_title("Elbow Method")
        st.pyplot(fig1)

        # ------------------------------------------------
        # Select K
        # ------------------------------------------------
        k_value = st.slider("Select Number of Clusters (K)", 2, 10, 3)

        if st.button("Run K-Means"):

            kmeans = KMeans(n_clusters=k_value, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_scaled)

            df["Cluster"] = clusters

            st.success("Clustering Completed!")

            # ------------------------------------------------
            # Cluster Visualization (2D)
            # ------------------------------------------------
            st.subheader("📊 Cluster Visualization")

            fig2, ax2 = plt.subplots()
            scatter = ax2.scatter(
                X_scaled[:, 0],
                X_scaled[:, 1],
                c=clusters
            )
            ax2.set_xlabel(selected_features[0])
            ax2.set_ylabel(selected_features[1])
            ax2.set_title("Cluster Plot")
            st.pyplot(fig2)

            # ------------------------------------------------
            # Show Clustered Data
            # ------------------------------------------------
            st.subheader("📌 Clustered Dataset Preview")
            st.write(df.head())

            # Download Option
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Clustered Dataset",
                data=csv,
                file_name="clustered_data.csv",
                mime="text/csv"
            )

    else:
        st.warning("Please select at least 2 numeric features.")

else:
    st.info("Please upload a CSV dataset to begin.")

# ------------------------------------------------
# Theory Section
# ------------------------------------------------
st.markdown("---")
st.subheader("📘 Theory Explanation")

st.markdown("""
### 🔹 K-Means Clustering

K-Means is an unsupervised learning algorithm that groups data into K clusters.

Steps:
1️⃣ Choose number of clusters (K)  
2️⃣ Initialize centroids  
3️⃣ Assign points to nearest centroid  
4️⃣ Update centroids  
5️⃣ Repeat until convergence  

---

### 🔹 Elbow Method

Used to determine optimal K.

- Plot K vs Inertia (WCSS)
- Look for the "elbow point"
- After elbow → diminishing returns

---

### 🔹 Inertia (WCSS)

\[
WCSS = \sum (distance\ to\ centroid)^2
\]

Lower inertia → tighter clusters  

---

### 📊 Applications

- Customer Segmentation  
- Market Analysis  
- Image Compression  
- Pattern Recognition  
""")