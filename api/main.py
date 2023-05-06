from flask import Flask, request, jsonify
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# read data from CSV file
df = pd.read_csv('Merged file.csv')

# select the relevant columns
X = df[['artist', 'acousticness','danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']]

numerical_cols = X.select_dtypes(include=['number','float','string','int']).columns

# select the numerical columns from the X DataFrame
X = X[numerical_cols]

# standardize the data
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# train a KNN model for content-based filtering
model_cb = NearestNeighbors(n_neighbors=10, algorithm='ball_tree')
model_cb.fit(X_std)

# train a KNN model for hybrid filtering
X_hybrid = df[['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']]
scaler_hybrid = StandardScaler()
X_hybrid_std = scaler_hybrid.fit_transform(X_hybrid)
model_hybrid = NearestNeighbors(n_neighbors=5, algorithm='ball_tree')
model_hybrid.fit(X_hybrid_std)

app = Flask(__name__)

# define a route for content-based filtering
@app.route('/cb/<song_name>', methods=['GET'])
def get_similar_songs_cb(song_name):
    # lookup the song in the dataframe
    song_row = df.loc[df['name'] == song_name]
    if song_row.empty:
        return jsonify({'error': 'Song not found'})

    # get the index of the song
    song_index = song_row.index[0]

    # get the standardized features of the song
    song_features = X_std[song_index]

    # find the nearest neighbors
    distances, indices = model_cb.kneighbors([song_features])

    # return the names of the similar songs
    similar_songs = df.iloc[indices[0]][['name', 'artist','preview_url']]
    return jsonify(similar_songs.to_dict('records'))

# define a route for hybrid filtering
@app.route('/hybrid/<song_name>', methods=['GET'])
def get_similar_songs_hybrid(song_name):
    
    # lookup the song in the dataframe
    song_row = df.loc[df['name'] == song_name]
    if song_row.empty:
        return jsonify({'error': 'Song not found'})

    # get the index of the song
    song_index = song_row.index[0]

    # get the standardized features of the song
    song_features = X_hybrid_std[song_index]

    # find the nearest neighbors
    distances, indices = model_hybrid.kneighbors([song_features])

    # return the names of the similar songs
    similar_songs = df.iloc[indices[0]][['name', 'artist','preview_url']]
    return jsonify(similar_songs.to_dict('records'))

if __name__ == '__main__':
    app.run(debug=True)