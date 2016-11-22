import xlrd, os, sys, codecs, random, math
from sklearn import linear_model, svm, tree, ensemble, cross_validation, metrics, neural_network
from sklearn.neighbors import NearestNeighbors
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import precision_recall_curve
from pystruct.models import ChainCRF
from pystruct.learners import FrankWolfeSSVM
import sklearn.feature_selection
import matplotlib.pyplot as plt
import numpy as np
import xlrd, xlwt

filepath = 'F:/results/final_labeled_8_2'
excelfilepath = "D:/Research/Thesis2/experiments_documentation/ML_results_8_2/SVM/undersampling_precision_recall.xls"

def readFile_excel(filepath):
    filelist = os.listdir(filepath)
    records = list()

    for filename in filelist:
        onefilepath = os.path.join(filepath, filename)

        print(onefilepath)
        data = xlrd.open_workbook(onefilepath)
        table = data.sheets()[0]
        nrows = table.nrows

        for i in range(nrows):
            #skip the titles
            if (i == 0):
                continue
            
            record = table.row_values(i)

            word = record[0]
            if (len(word) == 0):
                continue

            records.append(record)

    return records

def readFile_csv(filepath):
    filelist = os.listdir(filepath)
    records = list()

    for filename in filelist:
        onefilepath = os.path.join(filepath, filename)

        print(onefilepath)
        f = codecs.open(onefilepath, 'r', 'utf8')
        #the first three lines of each file are useless:
        titles1 = f.readline()
        titles2 = f.readline()
        titles3 = f.readline()
        print "titles1: ", titles1
        print "titles2: ", titles2
        print "titles3: ", titles3
        
        for line in f:
            line_data = line.rstrip().split(',')

            word = line_data[0]
            if (len(word) == 0):
                continue
            
            #construct new record vector, values are floats
            record = list()
            record.append(word)
            for v in line_data[1:]:
                if (v == u''):
                    continue
                record.append(float(v))
            
            records.append(record)
    
    return records

def generate_samples(records):
    f_vectors = list()
    f_labels = list()
    samples = list()
    
    for i in range(len(records)):
        record = records[i]

        fvector = record[1:-1]
        flabel = record[-1]
        if (flabel != 1.0 and flabel != 0.0):
            #error labeling
            print "error labeleing: ", flabel
            continue

        f_vectors.append(fvector)
        f_labels.append(flabel)

    samples.append(f_vectors)
    samples.append(f_labels)

    return samples

def normalize(samples):
    #normalize every feature
    fvectors = samples[0]
    for i in range(len(fvectors[0])):
        #find the max and min values
        fmax = -sys.maxint - 1
        fmin = sys.maxint

        for record in fvectors:
            if record[i] > fmax:
                fmax = record[i]
            if record[i] < fmin:
                fmin = record[i]
        print i, fmax, fmin

        #normalization in place
        for f_index in range(len(samples[0])):
            try:
                norm_value = (samples[0][f_index][i] - fmin) / (fmax - fmin)
            except ZeroDivisionError:
                norm_value = fmax
            
            samples[0][f_index][i] = norm_value

    return samples

def MLfit_parameter_choose(data_train, records, choose, data_test, folds):
    fvector = data_train[0]
    labels = data_train[1]

    clf_best = None
    clf_gridsearch = None
    #grid search using 5-fold cross-validation
    if (choose == 1):
        #parameter search range
        C_range = np.logspace(-2, 5, 8)
        gamma_range = np.logspace(-3, 4, 8)
        param_grid = dict(gamma=gamma_range, C=C_range)
        clf_gridsearch = GridSearchCV(svm.SVC(), param_grid=param_grid,  scoring = 'accuracy', cv=folds)
        clf_gridsearch.fit(fvector, labels)

        print "the best parameter combinations: ", clf_gridsearch.best_params_, "score: ", clf_gridsearch.best_score_

        clf_best = clf_gridsearch.best_estimator_
    
    elif(choose == 3):
        tree_range = np.arange(1, 110, 10)
        max_feature_range = np.arange(1, 14, 1)
        depth_range = np.arange(1, 150, 10)

        param_grid = dict(n_estimators = tree_range, max_features = max_feature_range, max_depth = depth_range)
        clf_gridsearch = GridSearchCV(ensemble.RandomForestClassifier(), param_grid=param_grid, scoring = 'accuracy', cv=folds)
        clf_gridsearch.fit(fvector, labels)

        print "the best parameter combinations: ", clf_gridsearch.best_params_, "score: ", clf_gridsearch.best_score_
        print "feature importace: ", clf_gridsearch.best_estimator_.feature_importances_

        clf_best = clf_gridsearch.best_estimator_

    elif(choose == 2):
        depth_range = np.arange(1, 110, 10)
        max_feature_range = np.arange(1, 14, 1)

        param_grid = dict(max_depth = depth_range, max_features = max_feature_range)
        clf_gridsearch = GridSearchCV(tree.DecisionTreeClassifier(), param_grid=param_grid, scoring = 'accuracy', cv=folds)
        clf_gridsearch.fit(fvector, labels)

        print "the best parameter combinations: ", clf_gridsearch.best_params_, "score: ", clf_gridsearch.best_score_
        print "feature importace: ", clf_gridsearch.best_estimator_.feature_importances_

        clf_best = clf_gridsearch.best_estimator_

    elif (choose == 4):
        C_range = np.logspace(-2, 5, 8)
        solver_range = ['newton-cg', 'lbfgs', 'liblinear']

        param_grid = dict(C = C_range, solver = solver_range)
        clf_gridsearch = GridSearchCV(linear_model.LogisticRegression(penalty = 'l2'), param_grid=param_grid, scoring = 'accuracy', cv=folds)
        clf_gridsearch.fit(fvector, labels)

        print "the best parameter combinations: ", clf_gridsearch.best_params_, "score: ", clf_gridsearch.best_score_

        clf_best = clf_gridsearch.best_estimator_

    elif (choose == 5):
        #does not support neural network
        alpha_range = np.logspace(-5, 4, 10)
        param_grid = dict(alpha = alpha_range)
        clf = GridSearchCV(neural_network.MLPClassifier(), param_grid=param_grid, scoring = 'accuracy', cv=folds)
        clf.fit(fvector, labels)

        print "the best parameter combinations: ", clf.best_params_, "score: ", clf.best_score_

        clf_best = clf

    for tup in clf_gridsearch.grid_scores_:
        print tup

    '''
    #write to excel file
    writefile = xlwt.Workbook()
    table = writefile.add_sheet('sheet1')
    writefile.save(excelfilepath)'''
    
    #test the best classifier with test data
    #results = clf.predict(data_test[0])
    #precision, recall, accuracy, fallout = statistics(results, data_test[1])
    #print "precision: ", precision, "recall: ", recall, "accuracy: ", accuracy

    #return the trained classifier 
    return clf_best

def MLfitCRF(data_train, data_test, records, folds):
    fvector = np.array([data_train[0]])
    labels = np.array([data_train[1]])

    #create CRF model
    CRFmodel = ChainCRF()
    #create ML classifier
    ssvm = FrankWolfeSSVM(model = CRFmodel, C = 0.1)
    #training
    ssvm.fit(fvector, labels)

    #model testing
    fvector_test = np.array(data_test[0])
    labels_test = np.array(data_test[1])
    score = ssvm.score(fvector_train, labels_test)

    print score

    return

def MLfit_precision_recall_curve(data, choose, clf):
    fvector = data[0]
    labels = data[1]
    area = 0
    
    if (choose == 1):
    #clf.fit(fvector, labels)
        decision_values = clf.decision_function(fvector)

        precisions, recalls, thresholds = metrics.precision_recall_curve(labels, decision_values)
        area = metrics.average_precision_score(labels, decision_values)
    
    elif (choose == 3):
        proba_values = clf.predict_proba(fvector)[:,1]

        precisions, recalls, thresholds = metrics.precision_recall_curve(labels, proba_values)
        area = metrics.average_precision_score(labels, proba_values)
        
    elif (choose == 2):
        proba_values = clf.predict_proba(fvector)[:,1]

        precisions, recalls, thresholds = metrics.precision_recall_curve(labels, proba_values)
        area = metrics.average_precision_score(labels, proba_values)

    elif (choose == 4):
        proba_values = clf.predict_proba(fvector)[:,1]

        precisions, recalls, thresholds = metrics.precision_recall_curve(labels, proba_values)
        area = metrics.average_precision_score(labels, proba_values)

    #having problems writing numpy floats
    #write to excel file
    writefile = xlwt.Workbook()
    table = writefile.add_sheet('sheet1')
    for i in range(len(thresholds)):
        print thresholds[i], precisions[i], recalls[i]
        #table.write(i + 1, np.float64(thresholds[i]).item(), np.float64(precisions[i]).item(), np.float64(recalls[i]).item())

    print "area under the curve: ", area
    #writefile.save(excelfilepath)
    
    plt.plot(recalls, precisions)
    plt.xlabel("recall", fontsize = 18)
    plt.ylabel("precision", fontsize = 18)
    plt.title("precision-recall curve: Decision Tree with undersampling, AUC = {0:0.2f}".format(area), fontsize = 18)
    plt.show()

def MLfit_pure_test(data, clf, records_test):
    fvector = data[0]
    labels = data[1]

    #five times of running, average
    precisions = list()
    recalls = list()
    accuracys = list()
    fallouts = list()
    for i in range(5):
        results = clf.predict(fvector)
        precision, recall, accuracy, fallout = statistics(results, labels)

        precisions.append(precision)
        recalls.append(recall)
        accuracys.append(accuracy)
        fallouts.append(fallout)
    
    '''
    print len(results), len(labels), len(records_test)
    for i in range(len(records_test)):
        if (labels[i] != results[i]):
            print records_test[i], labels[i], results[i]'''

    return [sum(precisions) / float(5), sum(recalls) / float(5), sum(accuracys) / float(5), sum(fallouts) / float(5)]
    
def feature_selection_univariate(data):
    fvector = data[0]
    labels = data[1]

    Kbest = sklearn.feature_selection.SelectKBest(sklearn.feature_selection.chi2, k = 5)
    best_result = Kbest.fit(fvector, labels)

    print best_result.scores_

def feature_selection_RFE(data, choose, folds):
    fvector = data[0]
    labels = data[1]
    
    rfecv = None
    if (choose == 2):
        svc = svm.SVC(kernel = 'linear')
        rfecv = sklearn.feature_selection.RFECV(estimator = svc, cv = folds,
                                                scoring = "accuracy")
        rfecv.fit(fvector, labels)

    print("Optimal number of features : %d" % rfecv.n_features_)

    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()

    return
    
def statistics(predicted, labels_test):
    print 'length: ', len(predicted), len(labels_test)
    total_amount = len(predicted)
    #semantic scholarly entity as positive
    TP_count = 0
    FP_count = 0
    TN_count = 0
    FN_count = 0
    
    for i in range(total_amount):
        if (predicted[i] == 1.0 and labels_test[i] == 1.0):
            TP_count += 1
        elif (predicted[i] == 0.0 and labels_test[i] == 1.0):
            FN_count += 1
        elif (predicted[i] == 0.0 and labels_test[i] == 0.0):
            TN_count += 1
        else:
            FP_count += 1

    print TP_count, FP_count, TN_count, FN_count
    TP = float(TP_count) / total_amount
    FP = float(FP_count) / total_amount
    TN = float(TN_count) / total_amount
    FN = float(FN_count) / total_amount
    ACC = float(TP_count + TN_count) / total_amount

    print TP, FP, TN, FN
    print "False Positives: ", FP, "False Negatives: ", FN
    
    try:
        precision = float(TP_count) / float(TP_count + FP_count)
    except ZeroDivisionError:
        precision = 0

    try:
        recall = float(TP_count) / float(TP_count + FN_count)
    except ZeroDivisionError:
        recall = 0

    try:
        fallout = float(FP) / float(FP + TN)
    except ZeroDivisionError:
        recall = 0
    
    #print 'precision: ', precision, 'recall: ', recall
    print '      '

    return [precision, recall, ACC, fallout]

    
def data_split(total_amounts, folds, i):      
    test_amounts = total_amounts / folds

    start = i * test_amounts
    #two cases for locate end
    if (i == folds - 1):
        end = total_amounts
    else:
        end = start + test_amounts
    
    return [start, end]

def under_sampling(records):
    positive = list()
    negative = list()

    for record in records:
        label = record[-1]
        if (label == 1.0):
            positive.append(record)
        elif (label == 0.0):
            negative.append(record)
        else:
            print 'error'
    
    print '+: ', len(positive), '-: ', len(negative)
    #generate random numbers for undersampling
    random_list = list()
    for i in range(len(positive)):
        random_list.append(random.randint(0, len(negative) - 1))

    #generate new records
    records_new = list()
    for i in range(len(positive)):
        records_new.append(positive[i])
        records_new.append(negative[random_list[i]])

    return records_new

def over_sampling(records):
    positive = list()
    negative = list()

    for record in records:
        label = record[-1]
        if (label == 1.0):
            positive.append(record)
        elif (label == 0.0):
            negative.append(record)
        else:
            print 'error: ', record

    print '+: ', len(positive), '-: ', len(negative)

    random_list = list()
    for i in range(len(negative)):
        random_list.append(random.randint(0, len(positive) - 1))

    records_new = list()
    for i in range(len(negative)):
        records_new.append(positive[random_list[i]])
        records_new.append(negative[i])

    return records_new
    
def SMOTE(records, N, k):
    #list for synthetic samples
    synthetic = list()

    positive = list()
    negative = list()

    for record in records:
        label = record[-1]
        if (label == 1.0):
            positive.append(record[1:-1])
        elif (label == 0.0):
            negative.append(record[1:-1])
        else:
            print 'error'

    #create the object for calculating KNN, we find the k + 1 NN to exclude itself
    neigh = NearestNeighbors(n_neighbors = k + 1)
    neigh.fit(positive)

    for point in positive:
        #for each point, find its K nearest neighbors, we do not want the first one
        neighbors = neigh.kneighbors([point], return_distance = True)

        #eliminate the first one, this is the point itself, numpy ndarray
        neighbors_value = neighbors[0][0][1:]
        neighbors_index = neighbors[1][0][1:]
        #print neighbors_value, neighbors_index
            
        #generate synthetic point
        #print "original: ", point
        #print "neighbors: ", records[neighbors_index[0]]
        populate(point, neighbors_index, records, N, k, synthetic)

    records.extend(synthetic)
    return records

def populate(origin_point, neighbors_index, records, N, k, synthetic):
    for i in range(N):
        random_choose = random.randint(0, k - 1)
        neighbor = records[neighbors_index[random_choose]][1:-1]

        new_vector = list()
        new_vector.append('synthetic')
        for attr_index in range(len(neighbor)):
            dif = neighbor[attr_index] - origin_point[attr_index]
            gap = random.uniform(0.0, 1.0)
            #there are integers and floats
            new_value = 0
            if (origin_point[attr_index] - math.floor(origin_point[attr_index]) == 0.0):
                new_value = round(origin_point[attr_index] + gap * dif)
            else:
                new_value = origin_point[attr_index] + gap * dif
            
            new_vector.append(new_value)

        new_vector.append(1.0)
        #print origin_point, neighbor, new_vector
        #put new data into synthetic list
        synthetic.append(new_vector)

    return


def easy_ensemble(fvector_train, labels_train, fvector_test, labels_test):
    print 'easy ensemble'
    #split
    fvector_positive = list()
    labels_positive = list()
    fvector_negative = list()
    labels_negative = list()
    for i in range(len(fvector_train)):
        if (labels_train[i] == 1):
            fvector_positive.append(fvector_train[i])
            labels_positive.append(labels_train[i])
        else:
            fvector_negative.append(fvector_train[i])
            labels_negative.append(labels_train[i])

    pos_len = len(fvector_positive)
    neg_len = len(fvector_negative)
    T = int(math.ceil(float(neg_len) / float(pos_len)))

    #AdaboostClassifier
    voting = [0 for i in range(len(fvector_test))]
    #clf = ensemble.AdaBoostClassifier(n_estimators = 10)
    clf = svm.SVC(C = 0.5, kernel = 'rbf', gamma = 'auto')
    for i in range(T):
        fvector_train_new = list()
        labels_train_new = list()
        if (len(fvector_negative) <= pos_len):
            #the last iteration, combine the rest of negative samples with
            #positive samples together 
            fvector_train_new.extend(fvector_negative)
            labels_train_new.extend(labels_negative)
            fvector_train_new.extend(fvector_positive)
            labels_train_new.extend(labels_positive)
        
        else:
            #randomly choose pos_len number of negative samples
            for s in range(pos_len):
                neg_len = len(fvector_negative)
                pos = random.randint(0, neg_len - 1)
                fvector_train_new.append(fvector_negative[pos])
                labels_train_new.append(labels_negative[pos])
                del fvector_negative[pos]
                del labels_negative[pos]

            #combine positive with negative samples
            fvector_train_new.extend(fvector_positive)
            labels_train_new.extend(labels_positive)

        print 'new data set: ', len(fvector_train_new), len(labels_train_new)
        clf.fit(fvector_train_new, labels_train_new)
        curResults = clf.predict(fvector_test)
        for i in range(len(fvector_test)):
            voting[i] += curResults[i]

    criteria = T / 2
    results = [0 for i in range(len(voting))]
    for i in range(len(voting)):
        if (voting[i] > criteria):
            results[i] = 1.0
    
    return results

def split_train_test(records):
    random_list = random.sample(range(0, len(records) - 1), 500)

    test_set = list()
    train_set = list()
    for i in range(len(records)):
        if (i in random_list):
            test_set.append(records[i])
        else:
            train_set.append(records[i])

    print "training set: ", len(train_set), "testing set: ", len(test_set)
    return [train_set, test_set]
    
if __name__ == '__main__':
    records_origin = readFile_csv(filepath)
    #split data, 400 just for test
    records, records_test = split_train_test(records_origin)

    #three sampling methods:
    #records_new = records
    records_new = SMOTE(records, 4, 3)
    #records_new = under_sampling(records)
    print len(records)
    
    
    count_0 = 0
    count_1 = 0
    #just for test
    for record in records:
        if (record[-1] == 0.0):
            count_0 += 1
        else:
            count_1 += 1
    print "positive vs negative: ", count_1, count_0

    #create feature vector
    samples = generate_samples(records_new)
    samples_test = generate_samples(records_test)
    
    print len(samples[0]), len(records_new), len(samples_test[0]), len(records_test)

    #parameter choose function
    clf = MLfit_parameter_choose(samples, records, 4, samples_test, 3)
    
    ##plotting function
    MLfit_precision_recall_curve(samples_test, 4, clf)

    #measure on the testing data set
    precision, recall, accuracy, fallout = MLfit_pure_test(samples_test, clf, records_test)

    print "final test results"
    print "precision: ", precision, "recall: ", recall, "accuracy: ", accuracy, "fallout: ", fallout

    #feature_selection_univariate(samples)
    #feature_selection_RFE(samples, 2, 5)
    #MLfitCRF(samples, samples_test, records, 5)
