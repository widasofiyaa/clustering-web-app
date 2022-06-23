import numpy as np
from sklearn import cluster, datasets
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Select, Slider
from bokeh.palettes import Spectral6
from bokeh.plotting import figure
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
np.random.seed(0)
def clustering(X, algorithm, n_clusters):
    bandwidth = cluster.estimate_bandwidth(X, quantile=0.3)
    if algorithm=='KMeans':
        model = cluster.KMeans(n_clusters=n_clusters)
    elif algorithm=='AgglomerativeClustering':
        model = cluster.AgglomerativeClustering(linkage="average",
                                                affinity="cityblock",
                                                n_clusters=n_clusters)
    model.fit(X)
    if hasattr(model, 'labels_'):
            y_pred = model.labels_.astype(int)
    else:
            y_pred = model.predict(X)
    return X, y_pred

def get_dataset(dataset, n_samples):
    if dataset == 'Marketing Campaign':
        df = pd.read_csv("myapp/marketing_campaign.csv", sep="\t", nrows=n_samples)
        df = df.dropna()
        df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"])
        dates = []
        for i in df["Dt_Customer"]:
            i = i.date()
            dates.append(i)
        days = []
        d1 = max(dates)  # taking it to be the newest customer
        for i in dates:
            delta = d1 - i
            days.append(delta)
        df["Customer_For"] = days
        df["Customer_For"] = pd.to_numeric(df["Customer_For"], errors="coerce")
        df["Age"] = 2021 - df["Year_Birth"]
        df["Spent"] = df["MntWines"] + df["MntFruits"] + df["MntMeatProducts"] + df["MntFishProducts"] + df[
            "MntSweetProducts"] + df["MntGoldProds"]
        df["Living_With"] = df["Marital_Status"].replace(
            {"Married": "Partner", "Together": "Partner", "Absurd": "Alone", "Widow": "Alone", "YOLO": "Alone",
             "Divorced": "Alone", "Single": "Alone", })
        df["Children"] = df["Kidhome"] + df["Teenhome"]
        df["Family_Size"] = df["Living_With"].replace({"Alone": 1, "Partner": 2}) + df["Children"]
        df["Is_Parent"] = np.where(df.Children > 0, 1, 0)
        df["Education"] = df["Education"].replace(
            {"Basic": "Undergraduate", "2n Cycle": "Undergraduate", "Graduation": "Graduate", "Master": "Postgraduate",
             "PhD": "Postgraduate"})
        df = df.rename(
            columns={"MntWines": "Wines", "MntFruits": "Fruits", "MntMeatProducts": "Meat", "MntFishProducts": "Fish",
                     "MntSweetProducts": "Sweets", "MntGoldProds": "Gold"})
        to_drop = ["Marital_Status", "Dt_Customer", "Z_CostContact", "Z_Revenue", "Year_Birth", "ID"]
        df = df.drop(to_drop, axis=1)
        df = df[(df["Age"] < 90)]
        df = df[(df["Income"] < 600000)]
        s = (df.dtypes == 'object')
        object_cols = list(s[s].index)
        LE = LabelEncoder()
        for i in object_cols:
            df[i] = df[[i]].apply(LE.fit_transform)
        ds = df.copy()
        cols_del = ['AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2', 'Complain',
                    'Response']
        ds = ds.drop(cols_del, axis=1)
        scaler = StandardScaler()
        scaler.fit(ds)
        scaled_ds = pd.DataFrame(scaler.transform(ds), columns=ds.columns)
        pca = PCA(n_components=2)
        pca.fit(scaled_ds)
        PCA_ds = pd.DataFrame(pca.transform(scaled_ds), columns=(["col1", "col2"]))
        return PCA_ds.values, None
    elif dataset == 'Blobs':
        return datasets.make_blobs(n_samples=n_samples, random_state=8)

n_samples = 1500
n_clusters = 2
algorithm = 'KMeans'
dataset = 'Blobs'

X, y = get_dataset(dataset, n_samples)
X, y_pred = clustering(X, algorithm, n_clusters)
spectral = np.hstack([Spectral6] * 20)
colors = [spectral[i] for i in y]
plot = figure(toolbar_location=None, title=algorithm)
source = ColumnDataSource(data=dict(x=X[:, 0], y=X[:, 1], colors=colors))
plot.circle('x', 'y', fill_color='colors', line_color=None, source=source)
clustering_algorithms= [
    'KMeans',
    'AgglomerativeClustering'
]
datasets_names = [
    'Marketing Campaign',
    'Blobs'
]

algorithm_select = Select(value='KMeans',
                          title='Pilih Algoritma:',
                          width=200,
                          options=clustering_algorithms)

dataset_select = Select(value='Blobs',
                        title='Pilih dataset:',
                        width=200,
                        options=datasets_names)

samples_slider = Slider(title="Jumlah sampel",
                        value=1500.0,
                        start=1000.0,
                        end=3000.0,
                        step=100,
                        width=400)

clusters_slider = Slider(title="Jumlah cluster",
                         value=2.0,
                         start=2.0,
                         end=10.0,
                         step=1,
                         width=400)

def update_algorithm_or_clusters(attrname, old, new):
    global X
    algorithm = algorithm_select.value
    n_clusters = int(clusters_slider.value)
    X, y_pred = clustering(X, algorithm, n_clusters)
    colors = [spectral[i] for i in y_pred]
    source.data = dict(colors=colors, x=X[:, 0], y=X[:, 1])
    plot.title.text = algorithm
def update_samples_or_dataset(attrname, old, new):
    global X, y
    dataset = dataset_select.value
    algorithm = algorithm_select.value
    n_clusters = int(clusters_slider.value)
    n_samples = int(samples_slider.value)
    X, y = get_dataset(dataset, n_samples)
    X, y_pred = clustering(X, algorithm, n_clusters)
    colors = [spectral[i] for i in y_pred]
    source.data = dict(colors=colors, x=X[:, 0], y=X[:, 1])
algorithm_select.on_change('value', update_algorithm_or_clusters)
clusters_slider.on_change('value_throttled', update_algorithm_or_clusters)
dataset_select.on_change('value', update_samples_or_dataset)
samples_slider.on_change('value_throttled', update_samples_or_dataset)
selects = row(dataset_select, algorithm_select)
input = row (samples_slider, clusters_slider)
inputs = column(selects, input)
curdoc().add_root(column(plot,inputs, height=50, sizing_mode="stretch_width"))
curdoc().title = "Visualisasi Clustering"