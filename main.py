import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("./data/Churn_Predictions.csv")
df1 = pd.read_csv("./data/Churn_Predictions.csv")


# datasetime genel bi gözlem yapmak için 
df.head()
df.info()

# uzun kolon ismini kısaltık
df.rename(columns ={"EstimatedSalary":"Salary","Geography":"Country"} , inplace = True)

# RowNumber kolonunu sildim 
# bu kolon index ile aynı bilgileri taşıdığı için 
del df["RowNumber"]

#CustomerId kolonunu sildim
# tüm değerler unique olduğu için algoritmam için bir anlam ifade etmeyecektir
del df["CustomerId"] 

# Surname kolonunu sildim
# tüm değerler unique olduğu için algoritmam için bir anlam ifade etmeyecektir
del df["Surname"]


# veri setimizde null değerler var mı kontrol ediyoruz
#  veri setimizde hiç nul varmı dedik varsa satır bazında bana gösteriyor
df[df.isnull().any(axis=1)]
# null değerimiz yok


# verimizin kolonları arasındaki genel istatistikleri görmek için
#? bu verileri cvs formatında bulunduğum dizine yazdık
#? bu şekilde incelemek daha kolay gelecek
df.describe().T.to_csv("describe.csv")

# 
# ?: kolon bazında incelemeye başlıyoruz


# ?: genel dağılımına baktık değerler nasıl dağılmış
df["CreditScore"].plot(kind="hist", bins=50)

# ?: kolonumuzdaki aykırı değerleri gözlemlemek için bir boxplot çizdirdik
# ?: Country bu kolona göre groupby yapıp ülkelere göre değerlendikdik
sns.boxplot(x="CreditScore", y="Country", data=df)
# ?: kolonumuzda aykırı değerler var, bunu gözlemleyebildik

# : şimdi bu  ülkelere göre ayırı değerleri silelim
# outlier tespitini yapacağımız kolonu seçiyoruz
for column in ["CreditScore"]:
    # *: ülkelre göre aykırı  değer despiti için Country kolonundaki unique değerleri seçiyoruz
    for group in df["Country"].unique():
        # *: uniq olan ilk değeri alıyoruz
        selected_group = df[df["Country"] == group]
        # *: groupby yapılmış yeni veri setimizdeki CreditScore kolonunu seçiyoruz
        selected_colomn = selected_group[column]
        # *: groupby yapılmış yeni veri setimizdeki CreditScore ortalmasını ve standart sapmasını alıyoruz
        std = selected_colomn.std()
        avg = selected_colomn.mean() 

        # *: 3sigma kuralını uyguluyoruz
        three_sigma_plus = avg + (3*std)
        three_sigma_minus = avg - (3*std)
        
        # *: groupby yapılmış yeni veri setimizdeki 3*sdt aşşağıda yada yukarda olanalrın index numaralarını listeye atadık
        outliers = selected_colomn[((selected_group[column] > three_sigma_plus) | (selected_group[column] < three_sigma_minus))].index
        # *: indexlerini bulduğumuz outlier değerleri siliyoruz 
        df.drop(index=outliers,axis=0,inplace=True)
        print(column , group, outliers)   





# CreditScore dağılımına bakıp aralıklara bölüyoruz
df["CreditScore"].plot(kind="hist", bins=50)
def credit_score_range(value):
    if value <300:
        return 1
    elif value >=300 and value <500:
        return 2
    elif value >=500 and value <600:
        return 3
    elif value >=600 and value <750:
        return 4
    elif value >=750 and value <800:
        return 5
    else:
        return 6
# aralıklara böldüğümüz değerli kolonumuza yazdırdık
df["CreditScore"] = df["CreditScore"].apply(lambda x: credit_score_range(x))


# =============================================================================
#! Gender
#  bu kononumuzdaki female ve male değişkenlerini 0 ve 1 olarak değiştirdik
F_M_dict = {"Female":1,"Male":0}
df["Gender"] = df["Gender"].map(F_M_dict)


#========================================================================================================================================#
#! Age 

df["Age"].describe()
df["Age"].corr(df["Exited"])
# : şimdi bu  ülkelere göre ayırı değerleri silelim
# outlier tespitini yapacağımız kolonu seçiyoruz
for column in ["Age"]:
    # *: ülkelre göre aykırı  değer despiti için Country kolonundaki unique değerleri seçiyoruz
    for group in df["Country"].unique():
        # *: uniq olan ilk değeri alıyoruz
        selected_group = df[df["Country"] == group]
        # *: groupby yapılmış yeni veri setimizdeki Age kolonunu seçiyoruz
        selected_colomn = selected_group[column]
        # *: groupby yapılmış yeni veri setimizdeki Age ortalmasını ve standart sapmasını alıyoruz
        std = selected_colomn.std()
        avg = selected_colomn.mean() 

        # *: 3sigma kuralını uyguluyoruz
        three_sigma_plus = avg + (3*std)
        three_sigma_minus = avg - (3*std)
        
        # *: groupby yapılmış yeni veri setimizdeki 3*sdt aşşağıda yada yukarda olanalrın index numaralarını listeye atadık
        outliers = selected_colomn[((selected_group[column] > three_sigma_plus) | (selected_group[column] < three_sigma_minus))].index
        # *: indexlerini bulduğumuz outlier değerleri siliyoruz 
        df.drop(index=outliers,axis=0,inplace=True)
        print(column , group, outliers) 


# for i in range(int(df["Age"].min()), int(df["Age"].max())):
#     df['TEMP'] = df["Age"] > i
#     v = df['TEMP'].corr(df["Exited"])
#     print(i, v)
    # if a > abs(corr):
        # a = corr
        # best_range = i
        # print(i, corr)

# burada çıkan sonuç 29 ile 54 yaş aralığındakilerin Exited kolonumuz ile olna corr değerlei yüksek
# çıkan aralıkta mı değil mi diye yeni bir kolon oluşturacağız

#  Age kolonumuzu aralıklara bölüyoruz
def age_range(value):
    if value >=29 and value <=54:
        return 1
    else:
        return 0
df["Age_range"] = df["Age"].apply(lambda x: age_range(x))
df["Age_range"].value_counts()



# ülke       erkek  kadın 
# Fransa	  62    62
# ispanya	  65	65 
# Almanya	  65    65 
# ülkelerin emeklilik yaşlarına göre emekli mi değil mi diye yeni bir kolon oluşturduk
def pension(value,country):
    if country == "France":
        if value >=62:
            return 1
        else:
            return 0
    if country == "Spain":
        if value >=65:
            return 1
        else:
            return 0
    if country == "Germany":
        if value >=65:
            return 1
        else:
            return 0
df["Pension"] = df.apply(lambda x: pension(x["Age"],x["Country"]),axis=1)
#========================================================================================================================================#




#========================================================================================================================================#
#! Tenure column
# genel olarak tenure kolonunun değerlerini kontrol ediyoruz
df["Tenure"].describe()
df["Tenure"].corr(df["Exited"])
df["Tenure"].plot(kind="hist", bins=50)
sns.displot(data=df, y="Tenure", hue="Country", multiple="stack")


#========================================================================================================================================#
#! Balance column
df["Balance"].describe()

sns.boxplot(data=df, x="Country", y="Balance")
#:germany kategorik değişkenimizin balance değerlerinde aykırı değerleri gözlemledik


# aykırı değerleri siliyoruz
for column in ["Balance"]:
    # *: ülkelre göre aykırı  değer despiti için Country kolonundaki unique değerleri seçiyoruz
    for group in df["Country"].unique():
        # *: uniq olan ilk değeri alıyoruz
        selected_group = df[df["Country"] == group]
        # *: groupby yapılmış yeni veri setimizdeki Age kolonunu seçiyoruz
        selected_colomn = selected_group[column]
        # *: groupby yapılmış yeni veri setimizdeki Age ortalmasını ve standart sapmasını alıyoruz
        std = selected_colomn.std()
        avg = selected_colomn.mean() 

        # *: 3sigma kuralını uyguluyoruz
        three_sigma_plus = avg + (3*std)
        three_sigma_minus = avg - (3*std)
        
        # *: groupby yapılmış yeni veri setimizdeki 3*sdt aşşağıda yada yukarda olanalrın index numaralarını listeye atadık
        outliers = selected_colomn[((selected_group[column] > three_sigma_plus) | (selected_group[column] < three_sigma_minus))].index
        # *: indexlerini bulduğumuz outlier değerleri siliyoruz 
        df.drop(index=outliers,axis=0,inplace=True)
        print(column , group, outliers) 



# Balance kolonundaki değerleri 0 ise 1 değilse 0 yaptım 
df["Balance_0"] = df["Balance"].apply(lambda x: 1 if x == 0 else 0)
df["Exited"].corr(df["Balance_0"])
df["Balance"].corr(df["Exited"])
# Balance_0 da daha yüksek bir korelasyon yakaladık

# Balance kolonundaki değerler çok yüksek olduğu için logaritması aldık
df["Balance"] = df["Balance"].apply(lambda x: x if x == 0 else np.log2(x))


#========================================================================================================================================#
#! Salary column

# boxplot ile aykırı değerleri gözlemlemeye çalışıyoruz
sns.boxplot(data=df, x="Country", y="Salary")

# aykırı değerleri siliyoruz
for column in ["Salary"]:
    # *: ülkelre göre aykırı  değer despiti için Country kolonundaki unique değerleri seçiyoruz
    for group in df["Country"].unique():
        # *: uniq olan ilk değeri alıyoruz
        selected_group = df[df["Country"] == group]
        # *: groupby yapılmış yeni veri setimizdeki Age kolonunu seçiyoruz
        selected_colomn = selected_group[column]
        # *: groupby yapılmış yeni veri setimizdeki Age ortalmasını ve standart sapmasını alıyoruz
        std = selected_colomn.std()
        avg = selected_colomn.mean() 

        # *: 3sigma kuralını uyguluyoruz
        three_sigma_plus = avg + (3*std)
        three_sigma_minus = avg - (3*std)
        
        # *: groupby yapılmış yeni veri setimizdeki 3*sdt aşşağıda yada yukarda olanalrın index numaralarını listeye atadık
        outliers = selected_colomn[((selected_group[column] > three_sigma_plus) | (selected_group[column] < three_sigma_minus))].index
        # *: indexlerini bulduğumuz outlier değerleri siliyoruz 
        df.drop(index=outliers,axis=0,inplace=True)
        print(column , group, outliers) 
# kolonumuzda aykırı değer yok

# Salary kolonundaki değerler yüksek olduğu için logaritması aldık
df["Salary"] = df["Salary"].apply(lambda x: np.log2(x))

# =============================================================================
#! Country 

# get_dammies ile country kolonunu işliyoruz
df = pd.get_dummies(df, columns=['Country'])



#========================================================================================================================================#
#! yeni kolonlar yaratmak

# yıllık maaşı 12 bölüp aylık maaş dıye yeni bir kolon oluşturuyoruz
df["month_salary"] = df["Salary"]/12



# age kolounun aralıllarının Exited kolonu ile olan en iyi korelasyonu buluyoruz
for x in range(df["Age"].min(),df["Age"].max()):
    df["test"] = df["Age"] >x
    corr = df["Exited"].corr(df["test"])
    print(x, corr)
# 42 olarak belirledim hem Age kolonumuzun ortalamasına da yakın
# :  
df["age_best_corr"]  = df["Age"].apply(lambda x: 42 if x >=30 and x<=54 else x )
df["Age"].mean()
# df[(df["age_best_corr"] ==0) & (df["Exited"]==0)]
df["age_best_corr"].value_counts()

columns =list(df.columns)
# farklı kolonlararsında işlemler yaparak Exited kolonumuzda yüksek korelasyon ve yeni kolonlar elde ekmet istiyorum
for c1 in columns:
  for c2 in columns:
    if c1>c2 and c1!="Exited" and c2!="Exited":
      df["test"] = df[c1]+df[c2]
      corr  =df["test"].corr(df["Exited"])
      if abs(corr)>0.15:
        print(c1, '+', c2, corr)
del df["test"]
df["Salary*Age"] =df["Salary"]*df["Age"]
df["month_salary+Country_Germany"] =df["month_salary"]+df["Country_Germany"]
df["IsActiveMember+Balance_0"] =df["IsActiveMember"]+df["Balance_0"]
df["month_salary"].corr(df["month_salary+Country_Germany"])
df.to_csv("last_case_dataset.csv",index=False)


# =============================================================================


# ! modelleme

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import time 

y = df["Exited"]
x = df.drop(["Exited"],axis=1)
y_encoded = LabelEncoder().fit_transform(y)
x_scaled = StandardScaler().fit_transform(x)


#  Feature importances uygulamadanki sonuçlar
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_encoded, test_size=0.3, random_state=101)
start_time = time.process_time()
model = RandomForestClassifier(n_estimators=500).fit(x_train, y_train)
print(time.process_time() - start_time )
preds = model.predict(x_test)
print(confusion_matrix(y_test,preds))
print(classification_report(y_test,preds))
model.score(x_test,y_test)




#  Feature importances uygulandıktan sonraki sonuçlar
feature_import = pd.Series(model.feature_importances_, index=x.columns)
feature_import.nlargest(10).plot(kind="barh")
best_feature = feature_import.nlargest(7).index
x_reduced = x[best_feature]
xr_scaled = StandardScaler().fit_transform(x_reduced)
xr_train, xr_test, yr_train, yr_test = train_test_split(xr_scaled, y_encoded, test_size=0.3, random_state=101)

start_time = time.process_time()
model = RandomForestClassifier(n_estimators=500).fit(xr_train, yr_train)
print(time.process_time() - start_time )
preds = model.predict(xr_test)
print(confusion_matrix(yr_test,preds))
print(classification_report(yr_test,preds))

model.score(xr_test,yr_test)
















































































































































































