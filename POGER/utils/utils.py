from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix

class Averager:
    def __init__(self):
        self.sum = 0
        self.count = 0

    def add(self, value):
        self.sum += value
        self.count += 1

    def get(self):
        return self.sum / self.count

def metrics(y_true, y_score):
    # print(y_true)
    # print(y_score)
    results = dict()
    y_pred = y_score.argmax(dim=1)
    results['accuracy'] = accuracy_score(y_true, y_pred)
    results['detail_f1'] = f1_score(y_true, y_pred, average=None).tolist()
    results['f1'] = f1_score(y_true, y_pred, average='macro')

    results['precision'] = precision_score(y_true, y_pred, average=None).tolist()
    results['recall'] = recall_score(y_true, y_pred, average=None).tolist()
    if y_score.shape[1] == 2:
        results['auc_ovo'] = roc_auc_score(y_true, y_score[:, 1])
        results['auc_ovr'] = roc_auc_score(y_true, y_score[:, 1])
    else:
        results['auc_ovo'] = roc_auc_score(y_true, y_score, average='macro', multi_class='ovo')
        results['auc_ovr'] = roc_auc_score(y_true, y_score, average='macro', multi_class='ovr')
    # print(confusion_matrix(y_true, y_pred))
    return results
