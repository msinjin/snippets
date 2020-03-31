from matplotlib import pyplot as plt
import re

def grid_search_2_df(grid_results):
    """
    Return a table of mean CV scores from a GridSearchCV object.
    """
    params_df = json_normalize(grid_results.cv_results_['params'])
    mean_test_score = pd.Series(
        grid_results.cv_results_['mean_test_score'], 
        name = 'mean_test_score')
    sd_test_score = pd.Series(
        grid_results.cv_results_['std_test_score'],
        name = 'sd_test_score')
    df = pd.concat([params_df.reset_index(drop = True), 
                    mean_test_score, 
                    sd_test_score], axis = 1)
    return df.sort_values('mean_test_score', ascending = False)

def plot_grid_search(grid_results):
    """
    Plot grid search CV scores of a GridSearchCV object.
    !! ASSUMING YOUR GRID ONLY HAS 2 HYPER-PARAMETERS !!
    """
    
    cv_results = grid_results.cv_results_
    cv_results_names = list(cv_results.keys())
    param_names = [s for s in cv_results_names if 'param_' in s]
    assert len(param_names) == 2, \
    "Only grid searches with exactly 2 hyper-paramters are allowed"
    grid_param_1 = sorted(list(set(list(cv_results[param_names[0]].data))))
    grid_param_2 = sorted(list(set(list(cv_results[param_names[1]].data))))
    
    scores_mean = cv_results['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(
        len(grid_param_2),
        len(grid_param_1))
    
    # TODO: implement sd plotting
    # scores_sd = cv_results['std_test_score']
    # scores_sd = np.array(scores_sd).reshape(
    #     len(grid_param_2),
    #     len(grid_param_1))
    
    # Plot Grid search scores
    # Param 1 is the X-axis, 
    # Param 2 is represented as a different curve (color line)
    _, ax = plt.subplots(1,1)
    for idx, val in enumerate(grid_param_2):
        ax.plot(grid_param_1, scores_mean[idx,:], '-o', 
        label= re.sub('^.*__', '', param_names[1]) + ': ' + str(val))
    
    ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
    ax.set_xlabel(re.sub('^.*__', '', param_names[0]), fontsize=16)
    ax.set_ylabel('CV Average Score', fontsize=16)
    ax.legend(loc="best", fontsize=15)
