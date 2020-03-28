import pandas as pd
import sklearn.metrics

def score(pred, labels_train):
    #Input : pred is a numpy array. labels_train is a dataframe

    pred_df = pd.DataFrame(columns=['grapheme_root', 'vowel_diacritic','consonant_diacritic'])

    for i in [0,1,2]:
        pred_df.iloc[:,i] = np.argmax(pred[i], axis=1)
    scores = []

    for component in ['grapheme_root','vowel_diacritic','consonant_diacritic']:
        y_true_subset = labels_train[component].values
        y_pred_subset = pred_df[component].values
        scores.append(sklearn.metrics.recall_score(
            y_true=y_true_subset, y_pred=y_pred_subset, average='macro'))
        print('Score {0}: {1:.3f}'.format(component, scores[-1]))

    final_score = np.average(scores, weights=[2,1,1])
    print('Final score: {0:.3f}'.format(final_score))

    return final_score