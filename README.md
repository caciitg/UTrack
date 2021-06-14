<img src="https://github.com/Arsh2k01/UTrack/blob/main/UTrack.jpg" width="650" height="620"> <br />
Project under [Consulting and Analytics Club, IITG](https://github.com/caciitg)

## 1. Technologies Used

1. Tweepy API
2. NLTK
3. BERT Model
4. Tensorflow
6. Seaborn
5. Streamlit

## 2. Project Description
### 2.1 Data Extraction and Preprocessing
We scraped data for each illness using the Tweepy API, based on keywords for each category.
Additionally, we scraped tweets that didn't contain these keywords. This data acted as the ‘neutral’ data.
The data was cleaned using libraries like regex, NLTK. Links, emojis, emoticons, and symbols were removed. 

### 2.2 DL Model
We explored Transformer models and found that BERT(Bidirectional Encoder Representations from Transformers) was better-suited for sentiment analysis. We used a pretrained BERT model and fine-tuned it on our training data. We trained a model for each class. <br />

### 2.3 Visualisation and Deployment
We used Seaborn to display the caculated level of Loneliness, Stress, and Anxiety for each user across time, thus enabling us to see how the user's mental state varied over time.
Additonally, you can also view each specific tweet and its scores.
Deployment was done using Sreamlit. 

## 3. Files
* **`Cleaning Tweets.py`** - Script to clean scraped tweets
* **`Extracting Targeted Tweets.py`** - Script to scrape a user's Twitter information
* **`Streamlit Deployment.py`** - Script to deploy the project
* **`Streamlit Deployment.ipynb`** - Jupyter Notebook to deploy the project
* **Extracted Tweets** - Training Data
* **Training Models:**
   * **`Anxiety Model.py`**
   * **`Lonely Model.py`**
   * **`Stress Model.py`**

## 4. Usage
To use UTrack, first add [this folder](https://tinyurl.com/utrackmodels) to your Google Drive.  <br />
Then run **`Streamlit Deployment.ipynb`** on Google Colab. Click on the **ngrok** link. <br />

Once you go to the localhost, use the following video as a reference:
  
  ![demo video](https://github.com/Arsh2k01/UTrack/blob/main/UTrack_Working.webp)


## 5. Team
* [Arsh Kandroo](https://github.com/Arsh2k01)
* [Franchis Saikia](https://github.com/Francode007)
* [Narmin Kauser](https://github.com/narmin24)
* [Roshan Shaji](https://github.com/roshan-shaji)
* [Jaswanth Gudiseva](https://github.com/jaswanth-gudiseva)
* [Atharva Shrawge](https://github.com/haxer-max)

## 6. References
* [Bidirectional Encoder Representations from Transformers (BERT): A sentiment analysis odyssey](https://arxiv.org/abs/2007.01127)

## 7. License
[MIT](https://choosealicense.com/licenses/mit/)
