This is a repository of the DataCamp projects that I have done in python.

### Project: a_visual_history_of_nobel_prize_winners
##### Used standard input and data manipulation techniques such as sorting and filtering to extract informational statistics from the nobel prize dataset.
#### Techniques used
 - Read CSV
 - Data Sorting Filtering
 - Graphing
 - Lambda
#### Libraries and Functions used
 - Pandas
   - .read_csv()
   - df.value_counts()
   - df.head()
   - df.nlargest()
   - df.nsmallerst()
   - df.groupby()
   - df.astype()
   - df.filter()
   - df.mean()
 - Numpy (qux)
   - np.floor()
 - Seaborn
   - .lineplot()
   - .lmplot()
 - Matplotlib
   - .ticker.PercentFormatter()
 
### Project: android.app.project
##### Used standard importing and data manipulation to analyze app data and reviews to look for the relationships in price and quality.
#### Techniques used
- Importing
- Cleaning data
  - Drop duplicates
  - Remove characters
- Data manipulation
- Probability and statistics
- Data visualization
- Sentiment analysis
#### Libraries and Functions used
- Pandas
  - .read_csv()
  - df.drop_duplicates()
  - df.index
  - df.values
  - df.sample()
  - df.value_counts()
  - df.sort_values()
  - df.unique()
  - df.mean()
  - df.isin()
  - .merge()
  - df.dropna()
- Plotly
  - .offline.init_notebook_mode()
  - .graph_bjs.bar()
  - .offline.iplot()
  - .graph_bjs.Histogram()
  - .graph_bjs.Bos()
- Seaborn
  - .set_style()
  - .jointplot()
  - .stripplot()
  - .boxplot()
- Warnings
  - .filterwarnings()
- Matplotlib.pyplot
  - .subplots()

### Project: Classify Song Genres from Audio Data
##### Used basic data manipulation to prepare a dataset to use machine learning algorithms to determine song genre from audio files.
#### Techniques used
- Data manipulation
- Data visualization
- Machine learning
- Importing and data cleaning
#### Libraries and Functions used
- Pandas
  - .read_csv()
  - .read_json()
  - .merge()
  - df.info()
  - df.corr()
  - df.background_gradient()
  - df.drop()
  - df.sample()
  - df.concat()
- Scikit-learn
  - .preprocessing
    - StandardScaler()
      - .fit_transform()
  - .decomposistion
    - PCA()
      - .fit()
      - .transform()
      - .explained_variance_ratio_
  - .tree
    - DecisionTreeClassifier()
      - .fit()
      - . predict()
  - .linear_model
    - LogisticRegression()
      - .fit()
      - .predict()
  - .metrics
    - classification_report()
  - .model_selection
    - KFold()
    - cross_val_score()
- Matplotlib.pyplot
  - .subplots()
  - .bar()
  - .plot()
- Numpy
  - .cumxum()
  - .mean()

### Project: Dr. Semmelweis and the Discovery of Handwashing
##### Used basic importing and cleaning techniques to analyze data and show that patients do better when doctors wash their hands
#### Techniques used
- Data Manipulation
- Data Visualization
- Probability and Statistics
- Importing and cleaning data
#### Libraries and Functions used
- Pandas
  - .read_csv()
  - df.plot()
  - df.head()
  - .to_datetime()
  - .sample()
  - .append()
- Numpy
  - percentile()

### Project: Exploring 67 years of LEGO
##### Used basic data importing and manipulation to explore the colors and themes of LEGOs.
#### Techniques Used
- Data Manipulation
- Data Visualization
- Importing and cleaning data
#### Libraries and Functions used
-Pandas
  - .read_csv()
  - df.head()
  - df.groupby()
  - df.count()
  - df.mean()
  - df.plot()
  - df.agg()

### Project: Exploring the Bitcoin Cryptocurrency Market
##### Used basic importing and data cleaning to analyze the cryptocurrency market to see its growth and valatility.
#### Techniques Used
- Data manipulation
- Data visualization
- Importing and cleaning data
- Lambda
#### Libraries and Functions used
- Pandas
  - .read_csv()
  - df.count()
  - df.query()
  - df.head()
  - df.set_index()
  - df.assign()
  - df.plot.bar()
  - df.sort_values()
- Matplotlib.pyplot
  - .style.use()
  - .subplots()
  - .bar()

### Project: Generating Keywords for Google Ads
##### Used basic data manipulation to generate lists of market sales keywords.
#### Techniques Used
- Data manipulation
#### Libraries and Functions used
- list.append()
- Pandas
  - .DataFrame.from_records()
  - df.head()
  - df.copy()
  - df.append()
  - df.to_csv()
  - df.groupby()
  
### Project: Mobile Games A:B Testing with Cookie Cats
##### Used basic import and data manipulation with general statistics to determine the difference in retention rate between two version of a mobile game.
#### Techniques Used
- Data Manipulation
- Data Visualization
- Probability and Statistics
- Importing and cleaning data
- Bootstrapping
#### Libraries and Functions used
- Pandas
  - .read_csv()
  - df.head()
  - df.groupby()
  - df.count()
  - df.plot()
  - df.sum()
  - df.mean()
  - df.sample()
  - df.append()
  - .DataFrame()
  
### Project: Predicting Credit Card Approvals
##### Used basic importing and data cleaning techniques to set up for a machine learning algorithm that will automatically pridict credit card aproval.
#### Techniques Used
- Data manipulation
- Machine Learning
- Importing and cleaning data
- Applied Finance
#### Libraries and Functions used
- Pandas
  - pd.read_csv()
  - df.head()
  - df.describe()
  - df.info()
  - df.tail()
  - df.replace()
  - df.fillna()
  - df.isnull()
  - df.values()
  - df.sum()
  - df.mean()
  - df.columns
  - df.dtypes
  - df.index[]
  - df.drop()
- Numpy
  - .NaN
- Scikit-learn
  - preprocessing
    - LabelEncoder()
      - .fit_transform()
    - MinMaxScaler()
      - .fit_transform()
  - train_test_split()
  - linear_model
    - LogisticRegression()
      - .fit()
      - .predict()
      - .score()
  - metrics
    - confusion_matrix()
  - model_selection
    - GridSearchCV()
      - .fit()
      - .best_score_
      - .best_params_

### Project: Real-time Insights from Social Media Data
##### Used basic importing and data manipulation to analyze 'Twitter-storm' data to analyze a 'hot-trend'.
#### Techniques Used
- Data manipulation
- Data visualization
- Probability and statistics
- Importing and cleaning data
- Pretty-printing JSON
#### Libraries and Functions used
- JSON
  - .loads()
  - .dumbss()
- collections
  - Counter()
- Matplotlib.pyplot
  - .hist()
- Pandas
  - .DataFrame()
  - df.style.background_gradient()

### Project: Reducing Traffic Mortality in the USA
##### Used basic data manipulation to look at the demographics of those in traffic accidents to use machine learning to determine which groups need to be focused on in order to reduce traffic accidents.
#### Techniques Used
- Data manipulation
- Data Visualization
- Machine learning
- Importing and cleaning data
#### Libraries and Functions used
- Pandas
  - .read_csv()
  - df.shape
  - df.info()
  - df.tail()
  - df.describe()
  - df.corr()
  - .melt()
  - .merge()
  - df.groupby().agg()
- Seaborn
  - .pairplot()
  - .violinplot()
  - .barplot()
- Scikit-learn
  - linear_model
    - LinearRegression()
      - .fit()
      - .coef_
  - preprocessing
    - StandardScaler()
      - .fit_transform()
  - decomposition
    - PCA()
      - .fit()
      - .explained_variance_ratio_.cumsum()
  - cluster
    - KMeans()
      - .fit()
      - .inertia_
      - .fit_transform()
      - .labels_
- Matplotlib.pyplot
  - .bar()
  - .plot()
  - .scatter()

### Project: THE GITHUB HISTORY OF THE SCALA LANGUAGE
##### Used basic data import and cleaning techniques to analyze GitHub data to see who had the most influence on a project and who the experts were. 
#### Techniques Used
- Data manipulation
- Data vizualizatioin
- Importing and cleaning data
#### Libraries and Functions used
- Pandas
  - .read_csv()
  - df.append()
  - .to_datetime()
  - .merge()
  - .df['date'].dt.year
  - df.groupby().count()
  - df.plot()
  - df.groupby().agg()
  - df.hist()
  - df.nlargest()
  - df.reset_index()
  - df.pivot_table()
  - df.head()
  - df.isin()

### Project: tv.halftimeshows.big.game.project
##### Used basic data manipulation and visualization to see how game scores and halftime shows has affected viewership over the years.
#### Techniques Used
- Data manipulation
- Data visualization
- Importing and cleaning data
#### Libraries and Functions used
- Pandas
  - .read_csv()
  - df.head()
  - df.info()
  - .merge()
  - df.groupby().count()
  - df.reset_index()
  - df.sort_values()
  - df.values()
  - df.dropna()
- Matplotlib.pyplot
  - .style.use()
  - .hist()
  - .subplot()
  - .tight_layout()
- Seaborn
  - .regplot()

### Project: Which Debts Are Worth the Bank's Effort?
##### Used data manipulation and visualization to determine which debts the bank should try to recover and which the bank should walkaway from using regression modeling.
#### Techniques Used
- Data manipulation
- Data visualization
- Probability and statistics
- Importing and cleaning data
- Regression modeling
#### Libraries and Functions used
- Pandas
  - .read_csv()
  - df.head()
  - df.describe()
  - df.unstack()
  - .crosstab()
- Numpy
  - .where()
- Matplotlib.pyplot
  - .scatter()
- Scipy
  - stats.kruskal()
  - stats.chi2_contingency()
- Statsmodels.api
  - .add_constant()
  - .OLS().fit()
  - .predict()
  - .summary()
  - .add_constant()

### Project: Who Is Drunk and When in Ames, Iowa?
##### Used basic data manipulation and visualization to create timelines to answer who was drunk and where in Ames, Iowa.
#### Techniques Used
- Data manipulation
- Data visualization
- Importing and cleaning data
#### Libraries and Functions used
- Pandas
  - .read_csv()
  - df.head()
  - df.value_counts()
  - df.groupby().size()
  - df.plot.bar()
  - df.dropna()
  - df.assign()
  - df.boxplot()
  - df.shape
  - .to_datetime()
  - df.dt.week
  - df.groupby().count()
  - df.unstack()
  - df.plot()
  
### Project: A Network Analysis of Game of Thrones
##### Used basic data manipulation and network analysis to determine which characters were the most important in the books from start to finish.
#### Techniques Used
- Data manipulation
- Network analysis
#### Libraries and Functions used
- Pandas
  - .read_csv()
  - df.head()
  - df.iterrows()
  - .DataFrame.from_records()
  - df.plot()
  - df.corr()
  - df.idxmax()
- Networkx
  - .Graph()
  - .add_edge()
  - .degree_centrality()
  - .betweenness_centrality()
  - .pagerank()
