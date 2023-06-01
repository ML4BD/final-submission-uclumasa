import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tslearn.metrics import cdist_dtw
from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import spectral_embedding
from scipy.sparse.csgraph import laplacian
from scipy import linalg

def mkdir_p(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''

    from errno import EEXIST
    from os import makedirs,path

    try:
        makedirs(mypath)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise


def prepare_data(students: pd.DataFrame, min_week: int, scale: bool = False) -> np.ndarray:
    """
    reshapes DataFrame from long to wide and returns an np.array
    :param df: pd.DataFrame with data in long format
    :param min_week: minimum number of weeks, the student must have data for
    :param scale: bool, if True, scale the data to have mean 0 and std 1
    :return: np.array with reshaped data
    """
    student_max_week = students.groupby('user_id')['weeks_since_first_transaction'].max()
    # Drop students who have max_weeks_since_first_transaction < min_week
    df = students[students['user_id'].isin(student_max_week[student_max_week >= min_week].index)]
    df = df[df['weeks_since_first_transaction'] < min_week]
    df_array = (df.sort_values(['user_id', 'weeks_since_first_transaction'], ascending=True)
                .groupby('user_id')
                .agg({'num_questions': lambda x: list(x)}))
    # drop users who have all 0s in the num_questions column
    df_array = df_array[df_array['num_questions'].apply(lambda x: sum(x)) > 0]
    if scale:
        df_array['num_questions'] = df_array['num_questions'].apply(lambda x: (x - np.mean(x)) / np.std(x))
    df_array.reset_index(inplace=True)
    data = np.asarray(df_array.num_questions.values.tolist())
    # also return the user_ids corresponding to each index in the array
    user_ids = df_array.user_id.values
    return user_ids, data

def visualize_data(student: np.ndarray): 
    """
    Visualizes student data as a histogram
    On the x axis put the index
    On the y axis put the number of questions(i.e. the value at that index)
    :param student: np array of student data
    """
    plt.figure(figsize=(15, 5))
    plt.bar(range(len(student)), student)
    plt.xlabel('Weeks')
    plt.ylabel('Number of questions')
    plt.show()

def spectral_clustering(W, n_clusters, random_state=111):
    """
    Spectral clustering
    :param W: np array of adjacency matrix
    :param n_clusters: number of clusters
    :return: tuple (kmeans, proj_X, eigenvals_sorted)
        WHERE
        kmeans scikit learn clustering object
        proj_X is np array of transformed data points
        eigenvals_sorted is np array with ordered eigenvalues 
        
    """
    # Compute eigengap heuristic
    L = laplacian(W, normed=True)
    eigenvals, _ = linalg.eig(L)
    eigenvals = np.real(eigenvals)
    eigenvals_sorted = eigenvals[np.argsort(eigenvals)]

    # Create embedding
    random_state = np.random.RandomState(random_state)
    proj_X = spectral_embedding(W, n_components=n_clusters,
                              random_state=random_state,
                              drop_first=False)

    # Cluster the points using k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state = random_state, n_init=10)
    kmeans.fit(proj_X)

    return kmeans, proj_X, eigenvals_sorted


def get_adjacency(S, connectivity='full'):
    """
    Computes the adjacency matrix
    :param S: np array of similarity matrix
    :param connectivity: type of connectivity 
    :return: adjacency matrix
    """
    
    if(connectivity=='full'):
        adjacency = S
    elif(connectivity=='epsilon'):
        epsilon = 0.5
        adjacency = np.where(S > epsilon, 1, 0)
    else:
        raise RuntimeError('Method not supported')
        
    return adjacency

def plot_metrics(n_clusters_list, metric_dictionary, save=False, outdir= 'Images', filename='metrics.png'):
    """
    Plots metric dictionary (auxilary function)
    [Optional]
    
    :param n_clusters_list: List of number of clusters to explore 
    :param metric_dictionary: 
    """
    # turn interaacive mode off
    plt.ioff()
    fig = plt.figure(figsize=(24, 20), dpi=80)
    i = 1

    for metric in metric_dictionary.keys():
        plt.subplot(3, 2, i)

        if metric == 'Eigengap':
            clusters = len(n_clusters_list)
            eigenvals_sorted = metric_dictionary[metric]
            plt.scatter(range(1, len(eigenvals_sorted[:clusters * 2]) + 1), eigenvals_sorted[:clusters * 2])
            plt.xlabel('Eigenvalues')
            plt.xticks(range(1, len(eigenvals_sorted[:clusters * 2]) + 1))
        else:
            plt.plot(n_clusters_list, metric_dictionary[metric], '-o')
            plt.xlabel('Number of clusters')
            plt.xticks(n_clusters_list)
        # set the y ticks to be from 0 to 1
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.ylabel(metric)
        if save:
            # Create new directory
            mkdir_p(outdir)
            plt.savefig(f'{outdir}/{filename}')
        i += 1

def get_heuristics_spectral(W, n_clusters_list, plot=True, save=False, outdir='Images', filename='metrics.png'):
    """
    Calculates heuristics for optimal number of clusters with Spectral Clustering
    
    :param W: np array of adjacency matrix
    :param n_clusters_list: List of number of clusters to explore
    :plot: bool, plot the metrics if true
    """
    silhouette_list = []
    eigengap_list = []
    
    df_labels = pd.DataFrame()

    for k in n_clusters_list:

        kmeans, proj_X, eigenvals_sorted = spectral_clustering(W, k)
        y_pred = kmeans.labels_
        df_labels[str(k)] = y_pred

        if k == 1:
            silhouette = np.nan
        else:
            silhouette = silhouette_score(proj_X, y_pred)
        silhouette_list.append(silhouette)


    metric_dictionary = {
                         'Silhouette': silhouette_list,
                         'Eigengap': eigenvals_sorted,
                        }
    
    if(plot):
        plot_metrics(n_clusters_list=n_clusters_list, metric_dictionary=metric_dictionary, save=save, outdir=outdir, filename=filename)
        return df_labels
    else:
        return df_labels, metric_dictionary
    
def get_distance_matrix(X, metric='euclidean', window=2):
    """
    calculates distance matrix given a metric
    :param X: np.array with students' time-series
    :param metric: str distance metric to compute
    :param window: int for DTW
    :return: np.array with distance matrix
    """
    norms = np.linalg.norm(X, axis=1)
    data_normalized = X / norms[:, np.newaxis]

    if metric == 'dtw':
        distance_matrix = cdist_dtw(data_normalized,
                                    global_constraint='sakoe_chiba',
                                    sakoe_chiba_radius=window)
    else:
        distance_vector = distance.pdist(data_normalized, metric)
        distance_matrix = distance.squareform(distance_vector)
    return distance_matrix

def get_affinity_matrix(D, gamma=1):
    """
    calculates affinity matrix from distance matrix
    :param D: np.array distance matrix
    :param gamma: float coefficient for Gaussian Kernel
    :return:
    """
    S = np.exp(-gamma * D ** 2)
    return S

def visualize_clusters(data, labels, weeks):
    """
    visualize the different time-series of students belonging to each cluster. 
    :param data: np.array with students' time-series
    :param labels: np.array predicted labels from clustering model
    :return: 

    Note: both data and label are arrays and data[0]'s label is labels[0]
    """

    # get unique labels
    plt.ion()
    unique_labels = np.unique(labels)
    print(f"Number of clusters: {len(unique_labels)}, Label Length: {len(labels)}")
    fig, axs = plt.subplots(1, len(unique_labels), figsize=(16, 4), facecolor='w', edgecolor='k')
    axs = axs.ravel()
    student_count_in_each_label = {label: 0 for label in unique_labels}
    # plot each cluster
    for label in unique_labels:
        # get indices of data points with label
        index = np.where(labels == label)
        # get number of students in cluster
        n_students = len(index[0])
        student_count_in_each_label[label] = n_students
        for i in index[0]:
            axs[label].bar(range(weeks), data[i], alpha=0.3)
        axs[label].set_title('Group {0}'.format(label))
        axs[label].set_ylabel('Num_questions')
        axs[label].set_xlabel('weeks')
        # limit y axis to 0 - 500
        #axs[label].set_ylim([0, 1])
    for label in unique_labels:
        print('Group {0} has {1} students'.format(label, student_count_in_each_label[label]))
    plt.show()