
from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA



app = Flask(__name__)
app.config["DEBUG"] = True

def prepross(d):
    d.drop_duplicates(keep='first')
    d.city.fillna(d.city.mean(), inplace=True) # x1 example
    le = preprocessing.LabelEncoder() #label encoder
    le.fit(d.nbre)
    d.nbre=le.transform(d.nbre)
    min_max_scaler = preprocessing.MinMaxScaler() #feature scaling
    scaled_array = min_max_scaler.fit_transform(d)
    d = pd.DataFrame(scaled_array,columns=d.columns)
    return d

def pca(alll):

    pca = PCA()
    pca.fit(alll)
    pca_samples = pca.transform(alll)
    return pca_samples




@app.route('/', methods=['GET'])
def home():
        return jsonify('good')
        #render_template('index.html')

@app.route('/clustering')
def predict():
    data= pd.read_csv('Dataset.csv')
    data=prepross(data)
    data=pca(data)
    clusterer = KMeans(n_clusters=4,random_state=42,n_init=10).fit(data)
    centers = clusterer.cluster_centers_
    labels= clusterer.predict(data)

    return jsonify('done')

if __name__ == '__main__':
      app.run(debug='true' )
