import warnings
import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
warnings.filterwarnings("ignore", category=FutureWarning)

def cross(mod, X, y, cv_number):
    cv_results = cross_validate(mod, X, y, cv=cv_number, scoring='accuracy', return_estimator= True)
    return cv_results

def grid(X, y):
    parameter_space = {'n_neighbors': [1, 3, 5, 11, 19],
                       'weights': ['uniform', 'distance'],
                       'algorithm': ['ball_tree', 'kd_tree', 'brute', 'auto'],
                       'metric': ['euclidean', 'manhattan']}
    model = KNeighborsClassifier()
    GSCV = GridSearchCV(model, parameter_space, n_jobs=-1, cv=3, scoring= 'accuracy')
    GSCV.fit(X, y)
    return GSCV

path = r"C:\Users\miair\PycharmProjects\untitled10\train_number.csv"
path_test = r"C:\Users\miair\PycharmProjects\untitled10\test_number.csv"
path_predictions = r"C:\Users\miair\PycharmProjects\untitled10\sample_submission_number.csv"
data = pd.read_csv(path)
data_test = pd.read_csv(path_test)
data_predictions = pd.read_csv(path_predictions)
X = data.drop('label', axis=1)
y = data['label']
X_test = data_test
print('Кросс-валидация(1), подобор параметров(2) или тестирование наилучшей модели(3): ')
Choose_number = input()
model = KNeighborsClassifier(algorithm= 'ball_tree', metric= 'euclidean', n_neighbors= 3, weights= 'distance')
if (Choose_number == '3'):
    # Обучение и тестирование наилучшей модели
    model.fit(X, y)
    y_predict_train = model.predict(X)
    y_predict_test = model.predict(X_test)
    print('Accuracy тренировочного набора: ', accuracy_score(y, y_predict_train))
    y_predict_test = pd.DataFrame({'Label': y_predict_test})
    data_predictions['Label'] = y_predict_test['Label']
    data_predictions.to_csv('prediction_number.csv', sep=';', index=False)  # запись в csv предсказаний оценок
elif (Choose_number == '1'):
    # Кросс-валидация
    cv_results = cross(model, X, y, 10)
    print('Значение кросс-валидации тренировочного набора:', cv_results['train_score'])
    print('Значение кросс-валидации тестового набора:', cv_results['test_score'])
elif (Choose_number == '2'):
    # Подбор параметров
    clf = grid(X, y)
    # Лучший результат
    print('Best parameters found:\n', clf.best_params_)
    # Все результаты
    print(" ")
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))


