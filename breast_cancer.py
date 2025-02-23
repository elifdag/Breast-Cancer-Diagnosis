import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split #veriyi eğitim ve test setlerine böler
from sklearn.preprocessing import StandardScaler #veriyi standartlaştırır(ölçeklendirme yapar)
from sklearn.svm import SVC #SVC: SVM sınıflandırıcısını sağlar
from sklearn.model_selection import GridSearchCV #hiperparametre optimizasyonu yapar
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tkinter import END

def generate_output(canvas,logText,progressbar,true_text,false_text):
    data = pd.read_csv('wdbc.data')

    data['diagnosis'] = (data['diagnosis'] == 'B').astype(int)

    X = data.drop(['diagnosis', 'id'], axis=1)
    y = data['diagnosis']

    feature_names = X.columns

    df = pd.concat([pd.DataFrame(y, columns=['diagnosis']), X], axis=1)

    logText.insert(END,"df.head:".center(70,"-")+"\n")
    logText.insert(END,df.head())
    logText.insert(END,"\n")
    logText.insert(END,"df.diagnosis value counts:".center(70,"-")+"\n")
    logText.insert(END,df['diagnosis'].value_counts())
    logText.insert(END,"\n")

    plt.figure(figsize=(6,4))
    sns.countplot(x='diagnosis', data=df)
    plt.title("Hedef değişkenin dağılımı (0 = Malignant, 1 = Benign)")
    plt.show()

    plt.figure(figsize=(15,10))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', cbar=True)
    plt.title("Korelasyon matrisi")
    plt.show()


    corr_threshold = 0.8
    high_corr = correlation_matrix[(correlation_matrix.abs() > corr_threshold) & (correlation_matrix.abs() <1.0)]
    logText.insert(END,"\nYüksek korelasyon gösteren özellik çiftleri:\n")
    logText.insert(END,high_corr.dropna(how='all').dropna(axis=1, how='all'))

    X_train , X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = SVC(kernel='linear', random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    logText.insert(END,"Karışıklık matrisi:\n")
    logText.insert(END,confusion_matrix(y_test, y_pred)) 
    logText.insert(END,"\n")

    logText.insert(END,"Sınıflandırma raporu:\n")
    logText.insert(END,classification_report(y_test, y_pred))
    logText.insert(END,"\n")
    logText.insert(END,f"Doğruluk skoru: {accuracy_score(y_test, y_pred):.2f}")
    param_grid = {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf']}
    grid = GridSearchCV(SVC(random_state=42), param_grid, cv=5)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred_best = best_model.predict(X_test)

    logText.insert(END,"\nEn iyi modelin performansı:\n")
    logText.insert(END,classification_report(y_test, y_pred_best))
    logText.insert(END,"\n")
    logText.insert(END,f"Doğruluk Skoru (Optimizasyon Sonrası): {accuracy_score(y_test, y_pred_best):.2f}")
    logText.insert(END,"\n")

    logText.see(END)

    truePredict = (y_test == y_pred_best).sum()
    test_size = len(X_test)

    progressbar['value'] = (truePredict/test_size) * 100

    truePercent = (truePredict/test_size) * 100
    falsePercent = ((test_size-truePredict)/test_size) * 100
    canvas.itemconfig(true_text, text=f"Doğru Tahmin: {truePredict}/{test_size} | %{truePercent:.2f}")
    canvas.itemconfig(false_text, text=f"Yanlış Tahmin: {test_size-truePredict}/{test_size} | %{falsePercent:.2f}")