# This file contains the methods the preprocess the data for lernnavi
# Please check the m4_lernnavi_uclumasa.ipynb for a more in-depth explanation on how the preprocessing works
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = './data' # You many change the directory
print("---Starting data loading---")
print("\t-->Loading users...")
users = pd.read_csv('{}/users.csv.gz'.format(DATA_DIR))
print("\t\tUsers loaded")
print("\t-->Loading events...")
events = pd.read_csv('{}/events.csv.gz'.format(DATA_DIR))
print("\t\tEvents loaded")
print("\t-->Loading transactions...")
transactions = pd.read_csv('{}/transactions.csv.gz'.format(DATA_DIR))
print("\t\tTransactions loaded")
print("\t-->Loading documents...")
documents = pd.read_csv('{}/documents.csv.gz'.format(DATA_DIR))
print("\t\tDocuments loaded")
print("\t-->Loading topics translated...")
topics_translated = pd.read_csv('{}/topics_translated.csv'.format(DATA_DIR)).rename(columns={"german_name": "challenge_name"})
print("\t\tTopics translated loaded")
print("---Data loading finished!---")

def load_mastery_array():
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
        total_diligence = []
        for topic in topics:
            children = topic['children']
            for child in children:
                total_mastery.append(topic['userData']['mastery'])
                total_diligence.append(topic['userData']['diligence'])
        user_id = row['user_id']
        start_time = row.event_date
        # add the row to the list
        rows.append([user_id, title, sum(total_mastery), start_time, sum(total_diligence)])
    return rows

def get_mastery_dfs(rows):
    print("---Starting dataframe creation---")
    # Create new dataframe from the list
    mastery_df = pd.DataFrame(rows, columns=['user_id', 'title', 'mastery', 'start_time' , 'diligence'])
    # Find the earliest start time for each user in events table
    earliest_start_time = events.groupby('user_id').event_date.min().reset_index()
    # rename column to min_start_time
    earliest_start_time = earliest_start_time.rename(columns={'event_date': 'min_start_time'})
    # merge with earliest_transaction
    mastery_df = mastery_df.merge(earliest_start_time, on='user_id', how='left')
    # convert start_time to datetime
    mastery_df.start_time = pd.to_datetime(mastery_df.start_time)
    # convert earliest_transaction to datetime
    mastery_df.min_start_time = pd.to_datetime(mastery_df.min_start_time)
    # calculate the number of weeks since first transaction
    mastery_df['weeks_since_first_transaction'] = (mastery_df.start_time - mastery_df.min_start_time).dt.days//7
    # drop the columns start_time and min_start_time
    mastery_df = mastery_df.drop(columns=['start_time','min_start_time'])
    # A user can check their mastery multiple times a week,
    # Find the max mastery for each user in each week, and find the max diligence for each user in each week
    # Keep those rows and drop the rest
    mastery_df = mastery_df.groupby(['user_id','title','weeks_since_first_transaction']).agg({'mastery': 'max', 'diligence': 'max'}).reset_index()
    # Multiply mastery col by 10 as explained in the data metadata file
    mastery_df.mastery = mastery_df.mastery*10
    mastery_df = mastery_df[mastery_df.title != 'Orthografie']
    mastery_df_german = mastery_df[mastery_df['title'] == "Deutsch"]
    mastery_df_math = mastery_df[mastery_df['title'] == "Mathematik"]

    # Now we will add extra features to this table
    """
        CREATE THE TABLE
    """
    new_transactions = transactions[['transaction_token','evaluation', 'document_id']]
    new_events = events[['user_id', 'transaction_token', 'event_date', 'action']]
    new_transactions = new_transactions.merge(new_events, on='transaction_token', how='right')
    new_transactions = new_transactions.merge(earliest_start_time, on='user_id', how='left')
    new_transactions.event_date = pd.to_datetime(new_transactions.event_date)
    new_transactions.min_start_time = pd.to_datetime(new_transactions.min_start_time)
    new_transactions['weeks_since_first_transaction'] = (new_transactions['event_date'] - new_transactions['min_start_time']).dt.days // 7

    """
        FIND WEEKLY EVENT COUNT
    """
    # Removed event counts as it's hard to distinguish events between different topics.
    # Find the number of transactions for each user in each week
    #num_events_weekly = new_transactions.groupby(['user_id','weeks_since_first_transaction']).action.count().reset_index()
    # Rename the column to num_events
    #num_events_weekly = num_events_weekly.rename(columns={'action': 'num_events'})

    """
        FIND WEEKLY QUESTIONS SOLVED
    """
    # Only consider question answering events (action = 'SUBMIT_ANSWER')
    num_questions_weekly = new_transactions[new_transactions.action == 'SUBMIT_ANSWER']
    num_questions_weekly = num_questions_weekly.dropna(subset = ["document_id"])
    doc_to_topic = documents.merge(topics_translated, how='left', left_on='topic_id', right_on='id')[['document_id','math']]
    doc_to_topic = doc_to_topic.drop_duplicates("document_id")
    num_questions_weekly = num_questions_weekly.merge(doc_to_topic, how = 'left', on='document_id')

    #Drop rows with math = NaN
    num_questions_weekly = num_questions_weekly[num_questions_weekly['math'].notna()]

    #Create german dataframe
    num_questions_weekly_german = num_questions_weekly[num_questions_weekly['math'] == 0]
    #Create math dataframe
    num_questions_weekly_math = num_questions_weekly[num_questions_weekly['math'] == 1]
    # Count the number of questions for each user in each week
    num_questions_weekly_german = num_questions_weekly_german.groupby(['user_id','weeks_since_first_transaction']).action.count().reset_index()
    num_questions_weekly_math = num_questions_weekly_math.groupby(['user_id','weeks_since_first_transaction']).action.count().reset_index()

    # Rename column to num_questions
    num_questions_weekly_math = num_questions_weekly_math.rename(columns={'action': 'num_questions'})
    num_questions_weekly_german = num_questions_weekly_german.rename(columns={'action': 'num_questions'})

    """
    FIND WEEKLY CORRECT QUESTIONS SOLVED
    """
    # Only consider question answering events that are correct (evaluation = 'CORRECT')
    num_correct_weekly = new_transactions[(new_transactions.evaluation == 'CORRECT') & (new_transactions.action == 'SUBMIT_ANSWER')]
    #Merge with documents and topics to separate german and math
    num_correct_weekly = num_correct_weekly.merge(doc_to_topic, how = 'left', on='document_id')
    #Drop rows with math = NaN
    num_correct_weekly = num_correct_weekly[num_correct_weekly['math'].notna()]

    #Create german dataframe
    num_correct_weekly_german = num_correct_weekly.query('math == 0')
    #Create math dataframe
    num_correct_weekly_math = num_correct_weekly.query('math == 1')

    # Count the number of questions for each user in each week
    num_correct_weekly_german = num_correct_weekly_german.groupby(['user_id','weeks_since_first_transaction']).action.count().reset_index()
    num_correct_weekly_math = num_correct_weekly_math.groupby(['user_id','weeks_since_first_transaction']).action.count().reset_index()

    # Rename column to num_questions
    num_correct_weekly_math = num_correct_weekly_math.rename(columns={'action': 'num_correct'})
    num_correct_weekly_german = num_correct_weekly_german.rename(columns={'action': 'num_correct'})

    """
        FIND WEEKLY PARTIALLY CORRECT QUESTIONS SOLVED
    """
    # Only consider question answering events that are correct (evaluation = 'PARTIAL')
    num_partial_weekly = new_transactions[(new_transactions.evaluation == 'PARTIAL') & (new_transactions.action == 'SUBMIT_ANSWER')]
    num_partial_weekly = num_partial_weekly.merge(doc_to_topic, how = 'left', on='document_id')
    #Drop rows with math = NaN
    num_partial_weekly = num_partial_weekly[num_partial_weekly['math'].notna()]

    #Create german dataframe
    num_partial_weekly_german = num_partial_weekly.query('math == 0')
    #Create math dataframe
    num_partial_weekly_math = num_partial_weekly.query('math == 1')

    # Count the number of questions for each user in each week
    num_partial_weekly_german = num_partial_weekly_german.groupby(['user_id','weeks_since_first_transaction']).action.count().reset_index()
    num_partial_weekly_math = num_partial_weekly_math.groupby(['user_id','weeks_since_first_transaction']).action.count().reset_index()

    # Rename column to num_questions
    num_partial_weekly_math = num_partial_weekly_math.rename(columns={'action': 'num_partial'})
    num_partial_weekly_german = num_partial_weekly_german.rename(columns={'action': 'num_partial'})


    # Merge the three tables together for german
    num_questions_weekly_german = num_questions_weekly_german.merge(num_correct_weekly_german, on=['user_id','weeks_since_first_transaction'], how='left')
    num_questions_weekly_german = num_questions_weekly_german.merge(num_partial_weekly_german, on=['user_id','weeks_since_first_transaction'], how='left')
    num_questions_weekly_german.fillna(0, inplace=True)

    # Merge the three tables together for math
    num_questions_weekly_math = num_questions_weekly_math.merge(num_correct_weekly_math, on=['user_id','weeks_since_first_transaction'], how='left')
    num_questions_weekly_math = num_questions_weekly_math.merge(num_partial_weekly_math, on=['user_id','weeks_since_first_transaction'], how='left')
    num_questions_weekly_math.fillna(0, inplace=True)

    # Create new column percentage_correct = (num_correct +0.5*num_partial)/ num_questions
    num_questions_weekly_german['percentage_correct'] = 100 * (num_questions_weekly_german.num_correct + 0.5*num_questions_weekly_german.num_partial)/num_questions_weekly_german.num_questions
    # Drop the columns num_correct and num_partial
    num_questions_weekly_german = num_questions_weekly_german.drop(columns=['num_correct','num_partial'])

    # Create new column percentage_correct = (num_correct +0.5*num_partial)/ num_questions
    num_questions_weekly_math['percentage_correct'] = 100 * (num_questions_weekly_math.num_correct + 0.5*num_questions_weekly_math.num_partial)/num_questions_weekly_math.num_questions
    # Drop the columns num_correct and num_partial
    num_questions_weekly_math = num_questions_weekly_math.drop(columns=['num_correct','num_partial'])

    """
    FIND THE REVIEW TASK COUNT
    """
    # Only consider question answering events (action = 'SUBMIT_ANSWER')
    num_review_weekly = new_transactions[new_transactions.action == 'VIEW_QUESTION']
    num_review_weekly = num_review_weekly.dropna(subset = ["document_id"])
    num_review_weekly = num_review_weekly.merge(doc_to_topic, how = 'left', on='document_id')
    #Drop rows with math = NaN
    num_review_weekly = num_review_weekly[num_review_weekly['math'].notna()]

    #Create german dataframe
    num_review_weekly_german = num_review_weekly.query('math == 0')
    #Create math dataframe
    num_review_weekly_math = num_review_weekly.query('math == 1')
    # Count the number of questions for each user in each week
    num_review_weekly_german = num_review_weekly_german.groupby(['user_id','weeks_since_first_transaction']).action.count().reset_index()
    num_review_weekly_math = num_review_weekly_math.groupby(['user_id','weeks_since_first_transaction']).action.count().reset_index()

    # Rename column to num_review
    num_review_weekly_math = num_review_weekly_math.rename(columns={'action': 'num_review'})
    num_review_weekly_german = num_review_weekly_german.rename(columns={'action': 'num_review'})

    """
    FIND THE VIEW COUNT
    """
    # Only consider question answering events (action = 'SUBMIT_ANSWER')
    num_view_weekly = new_transactions[new_transactions.action == 'REVIEW_TASK']
    num_view_weekly = num_view_weekly.dropna(subset = ["document_id"])
    num_view_weekly = num_view_weekly.merge(doc_to_topic, how = 'left', on='document_id')
    #Drop rows with math = NaN
    num_view_weekly = num_view_weekly[num_view_weekly['math'].notna()]

    #Create german dataframe
    num_view_weekly_german = num_view_weekly.query('math == 0')
    #Create math dataframe
    num_view_weekly_math = num_view_weekly.query('math == 1')

    # Count the number of questions for each user in each week
    num_view_weekly_german = num_view_weekly_german.groupby(['user_id','weeks_since_first_transaction']).action.count().reset_index()
    num_view_weekly_math = num_view_weekly_math.groupby(['user_id','weeks_since_first_transaction']).action.count().reset_index()

    # Rename column to num_review
    num_view_weekly_math = num_view_weekly_math.rename(columns={'action': 'num_view'})
    num_view_weekly_german = num_view_weekly_german.rename(columns={'action': 'num_view'})

    """
    FIND THE WINDOW VISIBLE RATIO FOR EACH USER
    """

    # in the events table for each user find the count of action='WINDOW_VISIBLE_FALSE'
    # and action='WINDOW_VISIBLE_TRUE'
    # and then calculate the ratio of WINDOW_VISIBLE_TRUE / (WINDOW_VISIBLE_TRUE + WINDOW_VISIBLE_FALSE)

    # drop all rows except those with action = 'WINDOW_VISIBLE_FALSE'
    temp = new_transactions[new_transactions.action == 'WINDOW_VISIBLE_FALSE']

    # count the number of rows for each user
    num_window_visible_false = temp.groupby(['user_id','weeks_since_first_transaction']).action.count()
    # fill the missing user ids with 0
    num_window_visible_false = num_window_visible_false.to_frame().fillna(0).reset_index()
    # drop all rows except those with action = 'WINDOW_VISIBLE_TRUE'
    temp = new_transactions[new_transactions.action == 'WINDOW_VISIBLE_TRUE']

    # count the number of rows for each user
    num_window_visible_true = temp.groupby(['user_id','weeks_since_first_transaction']).action.count()

    # fill the missing user ids with 0
    num_window_visible_true = num_window_visible_true.to_frame().fillna(0).reset_index()

    # Create a new dataframe with 2 columns: user_id and the ratio
    df_window_visible = pd.DataFrame({'user_id': num_window_visible_true.user_id, 'weeks_since_first_transaction': num_window_visible_true.weeks_since_first_transaction, 'ratio_window_visible': num_window_visible_true.action / (num_window_visible_true.action + num_window_visible_false.action)})

    """
    MERGE ALL THE TABLES
    """
    # Merge the tables together for german
    #mastery_df = mastery_df.merge(num_events_weekly, on=['user_id','weeks_since_first_transaction'], how='left')
    mastery_df_german = mastery_df_german.merge(num_questions_weekly_german, on=['user_id','weeks_since_first_transaction'], how='left')
    mastery_df_german = mastery_df_german.merge(num_review_weekly_german, on=['user_id','weeks_since_first_transaction'], how='left')
    mastery_df_german = mastery_df_german.merge(num_view_weekly_german, on=['user_id','weeks_since_first_transaction'], how='left')
    mastery_df_german = mastery_df_german.merge(df_window_visible, on=['user_id','weeks_since_first_transaction'], how='left')
    # Fill the NaN values in num_question and num_events with 0
    mastery_df_german[['num_questions', 'num_review']] = mastery_df_german[['num_questions', 'num_review']].fillna(0)

    # Merge the tables together for math
    #mastery_df = mastery_df.merge(num_events_weekly, on=['user_id','weeks_since_first_transaction'], how='left')
    mastery_df_math = mastery_df_math.merge(num_questions_weekly_math, on=['user_id','weeks_since_first_transaction'], how='left')
    mastery_df_math = mastery_df_math.merge(num_review_weekly_math, on=['user_id','weeks_since_first_transaction'], how='left')
    mastery_df_math = mastery_df_math.merge(num_view_weekly_math, on=['user_id','weeks_since_first_transaction'], how='left')
    mastery_df_math = mastery_df_math.merge(df_window_visible, on=['user_id','weeks_since_first_transaction'], how='left')
    # Fill the NaN values in num_question and num_events with 0
    mastery_df_math[['num_questions', 'num_review']] = mastery_df_math[['num_questions', 'num_review']].fillna(0)

    # We only want to consider the active weeks of the users
    temp = mastery_df_german.groupby('user_id').weeks_since_first_transaction.apply(list)
    mastery_df_german = mastery_df_german.join(temp, on='user_id', how='left', rsuffix='_list')
    # update the weeks_since_first_transaction column to the index of the list
    mastery_df_german['weeks_since_first_transaction'] = mastery_df_german.apply(lambda x: x['weeks_since_first_transaction_list'].index(x['weeks_since_first_transaction']), axis=1)
    # drop the list column
    mastery_df_german = mastery_df_german.drop(columns=['weeks_since_first_transaction_list'])

    # We only want to consider the active weeks of the users
    temp = mastery_df_math.groupby('user_id').weeks_since_first_transaction.apply(list)
    mastery_df_math = mastery_df_math.join(temp, on='user_id', how='left', rsuffix='_list')
    # update the weeks_since_first_transaction column to the index of the list
    mastery_df_math['weeks_since_first_transaction'] = mastery_df_math.apply(lambda x: x['weeks_since_first_transaction_list'].index(x['weeks_since_first_transaction']), axis=1)
    # drop the list column
    mastery_df_math = mastery_df_math.drop(columns=['weeks_since_first_transaction_list'])
    print("---Dataframe creation finished! ---")
    return mastery_df_german, mastery_df_math