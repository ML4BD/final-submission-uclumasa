## Folder Structure
```
final_submission_uclumasa
│
├── cluster_csv
├── cluster_regression_scores.md
├── cluster.py
├── images_german
├── images_math
├── lernnavi_clustering_efforts.ipynb
├── lernnavi_preprocess.py
├── lstm_functions.py
├── m4_lernnavi_uclumasa.ipynb
└── utils.py
```
## Pre processing:

To run the code provided in this repository, the following Python libraries should be installed in your environment:

- **pandas**: A powerful data manipulation library that offers data structures to quickly analyze and organize data.
- **importlib**: A built-in Python module for using import statements and managing modules dynamically.
- **numpy**: A library for numerical operations. It offers support for arrays, matrices, and many mathematical functions.
- **matplotlib**: A popular plotting and data visualization library.
- **lernnavi_preprocess**: A custom library (provided in this repository as `lernnavi_preprocess.py`) for preprocessing data specific to this project.
- **tqdm**: A library providing fast, extensible progress bars for loops and other iterative tasks. Here we use its `notebook` module for displaying progress bars in Jupyter notebooks.
- **cluster**: This is likely a custom module (found in this repository as `cluster.py`) used for the clustering feature in the datasets.
- **tslearn**: A machine learning library for time series analysis. Here, `cdist_dtw` is used for computing Dynamic Time Warping (DTW) distances.
- **scikit-learn (sklearn)**: A comprehensive machine learning library. We use several modules, including model selection tools (`train_test_split`, `GridSearchCV`), linear models (`LinearRegression`, `Ridge`, `Lasso`), metrics (`mean_squared_error`, `make_scorer`, `mean_absolute_error`), and preprocessing tools (`MinMaxScaler`).
- **os**: A built-in Python module for interacting with the operating system.
- **torch**: PyTorch, a popular open-source library used for machine learning and deep learning tasks.
- **seaborn**: A statistical data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics.
- **pathlib**: A built-in Python library for dealing with filesystem paths in a platform-agnostic way.
- **colorama**: A library used for adding colored terminal text and enabling ANSI escape character interpretation on Windows machines.

Please ensure all these libraries are installed to correctly execute the provided code. If you're using pip, you can install these libraries with the command `pip install pandas importlib numpy matplotlib tqdm tslearn scikit-learn torch seaborn pathlib colorama`. Note that for custom libraries (like `lernnavi_preprocess` and `cluster`), make sure they are available in the Python path.

### 1. Parsing the mastery level

We have used the following code to get the mastery level from NAVIGATE_DASHBOARD task. We have decided not to use the
even ACCEPT/REJECT_PROGRESS as we have found that the mastery levels retrieved from these actions are a lot less in number
compared to our previous method.

```python
# Get the total mastery as the sum of the mastery of all topics
import json
from tqdm import notebook as vis
from numpy import mean as mean
from numpy import sum as sum
rows = []
total = events[events['action']=='NAVIGATE_DASHBOARD'].shape[0]
for index,row in vis.tqdm(events[events['action']=='NAVIGATE_DASHBOARD'].iterrows(),
                          total=total,
                          desc="Processing records"):
    json_loaded = json.loads(row['tracking_data'])
    if(json_loaded['trackingDataType'] != 'DASHBOARD_VIEW_DATA'):
        continue
    title = json_loaded['dashboard']['title']
    if(len(json_loaded['dashboard']['topics']) == 0):
        continue
    topics = json_loaded['dashboard']['topics']
    total_mastery = []
    for topic in topics:
        children = topic['children']
        for child in children:
            total_mastery.append(topic['userData']['mastery'])
    user_id = row['user_id']
    start_time = row.event_date
    # add the row to the list
    rows.append([user_id, title, sum(total_mastery), start_time])
```

Each event that has the action of NAVIGATE_DASHBOARD has a JSON object serialized to a string in the tracking_data column
In this JSON object, under the dashboard field we have the title field, which states if the topics under this JSON object belongs
to Mathematics or German. Then under the topics field we have the actual topics, and under them we have the subtopics. We consider the
mastery level of a user to be the summation of all his mastery of his subtopics.

**Note there is also the diligence level right next to the mastery level in the json object. We currently do not consider this field,
but it could be interesting to later look at it.**

---

### 2. Creating timeseries data

#### a. Creating the initial data

Now that we had the mastery level of each student at specific
time intervals. (See [parsing the mastery section](#1.-parsing-the-mastery-level))
We'll be creating one dataframe for each of the topics.
We had to create a data in the following format:

```
user_id | week | mastery
```

After creating dataframes by using the
data we have parsed we notice the following:

```python
print('Orthografie: ' + str(mastery_df[mastery_df.title == 'Orthografie'].shape))
print('Mathematik: ' + str(mastery_df[mastery_df.title == 'Mathematik'].shape))
print('Deutsch: ' + str(mastery_df[mastery_df.title == 'Deutsch'].shape))
_________
>Orthografie: (6, 4)
>Mathematik: (34981, 4)
>Deutsch: (44648, 4)
```

Seeing that Orthografie only has 6 rows we decided to completely drop it and only consider Mathematics
and German as our topics.

One thing that we should
clarify here is that **we consider the weeks to be the active weeks(weeks that the user has done at least one kind of event)
of a user**. Meaning that in the case that a user solves some questions the first week then does not login to the platform for 5-6 weeks,
and comes back, the 2nd week will be considered the time where the user has returned. **this is because while analyzing the data
we have seen a lot of cases where a user solves a lot of questions for the first few weeks and then goes missing for several weeks**

#### b. Adding features

At this point we had the weekly mastery level for many of our users, to be able to
predict the mastery levels of the users we needed to expand our current dataframe with some features.
We decided to use the following metrics each of them split between the German and Mathematics topics:

1. num questions: The number of questions solved by that user in that week.
2. percentage correct: The percentage of correctly solved questions by that user in that week.
   Partially correct results account as 0.5 correct.
3. num review: The number of review tasks
4. num view: The number of tasks viewed

The splitting by topic was done by grabbing the document_id of the event from the documents table, this document id was then mapped to the
topics_translated table through the topic_id which gave us the information whether this question solved/reviewed was about Mathematics or German.

_One interesting thing we have noticed is that, we have counted the number of TOTAL questions solved per user per week without splitting it by topic.
And then the number of German and Mathematics questions solved, we then added the German and Mathematics solved per user per week and checked if it was equal
to the TOTAL number of questions solved per user per week. In most of the rows 62k/65k the number came out to be equal, but in some of the rows(3k) there were
small amount of differences, i.e. the sum of german + mathmatics questions were off by 1 or 2 than the TOTAL amount of questions. We think this is because of
dropping the Orthografie topic out of the picture along with some SUBMIT_ANSWER event rows having NaN document id(86 of them to be exact)_

---

### 3. Pruning the data

#### a. Users with negligable data

Now that we had a good timeseries data, we looked into some insights and tried to prune it to be better suited for our upcoming ML tasks.
One thing we have noticed is that there are a lot of users who have interacted with the platform for a really short amount of time, for instance 1 or
2 weeks. Because our task was to predict week N for each user by using the data from weeks 1...N-1. The users with only a few weeks would not be serving
us much in our tasks. Thus we had to decided to to only consider users who have interacted with the platform for at least X weeks. We have set this value
at the start to 5. This gave us `3047` users.

```python
max_weeks = mastery_df.groupby('user_id').weeks_since_first_transaction.max().reset_index()
users_with_no_week = max_weeks[max_weeks.weeks_since_first_transaction < 5]
mastery_df_subset = mastery_df[~mastery_df.user_id.isin(users_with_no_week.user_id)]
mastery_df_subset.user_id.nunique()
```

#### b. Users with no mastery

Currently our data had users who have interacted with the platform for at least 5 weeks.
But there was still an edge case we were yet to consider. There were some small amount of users who have obtained no mastery level
during their time spent on the platform.
As we were concerned with predicting the mastery level, we have decided that these users would not be valuable to us, and thus have dropped them from the dataframe.
The count of these users were 26.

[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/CNxME27U)

## Model Building

### Regression

For predicting the TARGET_WEEK selected we performed regression models by using the train data which from first week to TARGET_WEEK - 1

```python
# Define the models and their hyperparameters
models = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(),
    'Lasso': Lasso()
}

alphas = np.logspace(-3, 3, 20)  # Regularization strengths for Ridge and Lasso
```

Subsequently, a grid search is conducted for Ridge and Lasso to identify the best alpha value for each model. For Linear Regression, as there are no hyperparameters to optimize, the model is directly fitted to the training set.

```python
# Perform grid search for each model
for model_name, model in models.items():
    if model_name == 'LinearRegression':
        model.fit(X_train, y_train)
        mse = mean_squared_error(y_test, model.predict(X_test))
        print(f"{model_name} - Mean Squared Error: {mse}")
    else:
        param_grid = {'alpha': alphas}
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring=make_scorer(mean_squared_error))
        grid_search.fit(X_train, y_train)
        best_alpha = grid_search.best_params_['alpha']
        best_model = grid_search.best_estimator_
        mse = mean_squared_error(y_test, best_model.predict(X_test))
        print(f"{model_name} (alpha={best_alpha}) - Mean Squared Error: {mse}")
```

Now we are evaluating the performance of the three best models (Linear Regression, Ridge Regression, and Lasso Regression) on the target_data, which contains rows with weeks_since_first_transaction equal to TARGET_WEEK.

### LSTM and GRU

For LSTM and GRU models we created a **lstm_functions.py** script to define two deep learning models, an LSTM (Long Short-Term Memory) model and a GRU (Gated Recurrent Unit) model, along with a custom dataset class for loading and pre-processing the data. The models are intended to predict the mastery variable for a target week, given the input sequence of data from the preceding weeks.

#### LSTM Model
```python
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, device):
        super(LSTMModel, self).__init__()
        # Set the hidden size, number of layers, and device for the model
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        # Create an LSTM layer with the specified input size, hidden size, and number of layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True).to(self.device)
        # Add a fully connected layer with the same input and output size as the hidden size
        self.fc1 = nn.Linear(hidden_size, hidden_size).to(self.device)  # Add an additional linear layer
        # Add a final fully connected layer to produce the output of the desired size
        self.fc2 = nn.Linear(hidden_size, output_size).to(self.device)

    def forward(self, x):
        # Initialize hidden state (h0) and cell state (c0) with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        # Pass the input through the LSTM layer
        out, _ = self.lstm(x, (h0, c0))
        # Pass the output of the LSTM through the first fully connected layer
        out = self.fc1(out[:, -1, :])
        out = torch.relu(out)  # Add a ReLU activation function
        # Pass the output through the second fully connected layer to produce the final output
        out = self.fc2(out)
        return out
```

#### GRU Model
```python
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, device):
        super(GRUModel, self).__init__()
        # Set the hidden size, number of layers, and device for the model
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        # Create an GRU layer with the specified input size, hidden size, and number of layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True).to(self.device)
         # Add a fully connected layer with the same input and output size as the hidden size
        self.fc1 = nn.Linear(hidden_size, hidden_size).to(self.device)  # Add an additional linear layer
        # Add a final fully connected layer to produce the output of the desired size
        self.fc2 = nn.Linear(hidden_size, output_size).to(self.device)

    def forward(self, x):
        # Initialize hidden state (h0) with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        # Pass the input through the GRU layer
        out, _ = self.gru(x, h0)
        # Pass the output of the GRU through the first fully connected layer
        out = self.fc1(out[:, -1, :])
        out = torch.relu(out)  # Add a ReLU activation function
        # Pass the output through the second fully connected layer to produce the final output
        out = self.fc2(out)
        return out
```

We performed these models on both German and Math Datasets

## Model Evaluation

### Metrics
Mean Squared Error (MSE), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE) are popular evaluation metrics for regression tasks because they provide valuable insights into the performance of a model in predicting continuous outcomes. Each of these metrics has its unique advantages and contributes to the overall understanding of model performance.

- **Mean Squared Error (MSE)**: MSE measures the average squared difference between the predicted values and the actual values. By squaring the errors, MSE puts more emphasis on larger errors and penalizes them heavily. 
- **Mean Absolute Error (MAE)**: MAE calculates the average absolute difference between the predicted values and the actual values. This metric provides a more interpretable measure of the average error magnitude, as it is on the same scale as the target variable.
- **Root Mean Squared Error (RMSE)**: This metric combines the sensitivity to large errors (as in MSE) with the interpretability of the error scale (as in MAE). RMSE represents the standard deviation of the residuals (prediction errors) and provides an easily understandable measure of the average error magnitude.

We computed evaluation metric scores for Regression and demonstrated the best models for each Dataset

Then, we computed evaluation metric scores for LSTM and GRU and compared with a baseline score we have selected.

## Team Reflection

For this milestone, the members performed the following tasks:

**Aybars Yazici:** Data exploration and preprocessing, extracting mastery levels and the majority of the features.

**Ilker Gul:** Implementation of Regression, LSTM, and GRU models. Training, testing, and evaluating the models on the given dataset. Visualization of the results.

**Can Kirimca:** Evaluation of the models and visualization of the results. Extraction of a few features and separating the user records into Math and German.

Although different members worked on different components of the project, we frequently held meetings to discuss the intermediate results and consulted each member on each step. 

Since LSTM and GRU have large numbers of parameters, our main concern was that the amount of data might be insufficient to train those models. However, these models performed better than we expected and managed the capture the sequential patterns in our data, and as previously stated, they performed significantly better than regression models. One potential task for the future might be to perform data augmentation to train these models on larger data as we expect them to perform better on larger datasets.

Furthermore, one of the possible reasons why the regressions performed poorly is that we need to find features that describe the patterns in the data. One of our strategies for the next milestone will be to explore the dataset further and extract more meaningful features to improve the predictive performance of each model used.
