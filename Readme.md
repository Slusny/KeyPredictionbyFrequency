# Musical Key Prediction and PCA on Spotify Audio Data Using Fast Fourier Transform

In this project, the musical keys of piano records are predicted based on their frequency spectrum, calculated using Fast Fourier Transform (FFT). After collecting 30-second audio clips and key labels from the Spotify Web API and applying FFT, we use multinomial logistic regression for multi-class classification and achieve a cross-validation accuracy of 72.8% when predicting the relative key (i.e., keys with identical scales are categorized as one class). By visualizing the weights of the classifier, we recover the musical scale that belongs to each key. Furthermore, we provide insight into the systematic errors the classifier tends to make, namely that it sometimes misclassifies keys by a perfect fourth or perfect fifth. In addition, we perform PCA on our piano music dataset, which results in principal components that one can listen to, providing a new and interesting perspective on PCA.

## How to reproduce our results:

### Get the data:

1. Create an "app" on the Spotify Developer platform
https://developer.spotify.com/

2. In the src folder, create a credentials.py file with your app's login credentials:  
client_id =[your id]  
client_secret = [your secret]  
redirect_uri = 'http://localhost/'  

3. Run the src/SpotipyDataGetter.py script to collect the data. For this, you will need to install Spotipy (https://spotipy.readthedocs.io/en/2.19.0/) which interfaces with the Spotify API.  
By default, this will download the larger dataset used for classification.
To get the dataset for PCA, change the last line in the script to the following:  
val = sd.get_dataset_from_playlist(piano_playlist_url, '../data/piano')

3. To convert the .mp3 files to .WAV, install ffmpeg: https://ffmpeg.org/.
Execute bash script ./mp3_to_wav.sh with the first argument beeing the path to the directory which mp3 files should be converted. The second argument is the desired sample rate. It should be 4000 for classification or 2000 for PCA.
The folder where you downloaded the mp3 clips should be 'data/piano' for PCA or 'data/dl_more_piano' for classification.

4. Run src/append_fft_to_Dataset.py to perform FFT on the audio clips. The resulting dataframe with frequency components will be saved to the folder specified in the script.

### PCA

Run src/PCA.py. The arguments needed are explained in the script / the argparser.
Note that computing PCA requires at least around 16GB of RAM already for the smaller dataset with a sampling rate of 2000Hz, and more for larger datasets or higher sampling rates.

### Key classification

Run the Jupyter notebook exp/Key_classification.ipynb. Use the BINNING, BIN_OVER_OCTAVES and RELATIVE_KEY_PREDICTION options to specify exactly which experiment you want to execute.

The Jupyter notebook exp/DatasetExploration.ipynb containts the code to plot the key distribution of the dataset.
