import pandas as pd
import seaborn as sns        # görsellestirme kütüphanesi
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats           # bu proejde skewness değeri bulacağımız için istatistiksel bir kütüphane lazım
from scipy.stats import norm,skew

from sklearn.preprocessing import RobustScaler,StandardScaler     # datayı scale ederken her ikisini de denicez bu sefer
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet      # 4 tane regression modellerini kullanıcaz
from sklearn.model_selection import train_test_split,GridSearchCV      # veriyi bölmek için traintestspliti aldık, grid search cross valudationu da yapıcaz
from sklearn.metrics import mean_squared_error             # metriğimiz bu olcak
from sklearn.base import clone   # bu yöntemi averaging model yaparken yani modelleri birleştirirken kullanıcaz

# XGBoost
import xgboost as xgb
# warning
import warnings
warnings.filterwarnings('ignore')

column_name = ["MPG","Cylinders","Displacement","HorsePower","Weight","Acceleration","ModelYear","Origin"]   # feature variable lar - sadece araç isimlerini yazmadık ona da gerek yok
data = pd.read_csv(r'C:\Users\asaro\Downloads\auto+mpg\auto-mpg.data', names = column_name, na_values='?', comment='\t', sep=" ", skipinitialspace=True)   # boş veri varsa soru işareti koyucaz # sep'e bosluk dedik veri bosluklarla ayrıldıgı için # skipinitialspace'i true diyerek skiplemesini sağlıcaz
# MPG dediğimiz yakıt tüketimini kastediyor. bizim target variableımız oldugu için target adıyla değiştirelim daha anlaşılır olur.
data = data.rename(columns = {"MPG":"target"})

"""
Origin için; 1-amerika 2-avrupa 3-japonya originli araçlar
"""

print(data.head())
print("Data Shape: ",data.shape)
data.info()    # horsepower içinde 392 değer var yani 6 tane veri boş (missing value)  # categorical verimiz de yok. float ve int hepsi AMA origin sütunu 1-2-3 olarak gidiyor int ten cok categorical gibi.o yüzden onehotencoding yapabiliriz.
describe=data.describe()  # TARGET'in mean'i medyandan büyük yani sola yatık-sağa doğru kuyruğu uzar.bu yüzden pozitif skewnessdir. normal dağılım diyemeyiz.

# %% missing value

print(data.isna().sum())    # horsepowerda 6 tanes missing value varmıs - horsepowerın dagılımına baktık,mean>medyan(%50) çıktı yani sola yatık, kuyrugu sağa uzanıyor , pozitif skewness demek

data["HorsePower"] = data["HorsePower"].fillna(data["HorsePower"].mean())
print(data.isna().sum()) 
sns.distplot(data.HorsePower)

# %% EDA - data analysis

# numeric featurelerimiz var, o zaman olmazsa olmazımız korelasyona bakıcaz

corr_matrix = data.corr()
sns.clustermap(corr_matrix,annot=True,fmt=".2f")
plt.title("Correlation between features")
plt.show()
# Target bizim depended variablemiz,diğer variablelara bağımlı. Diğer kalanlar ise independed variablelarımız.
# target ile horsepower,weight,cylinders,displacement arasında negatif yüksek korelasyon var tabloda görülen.

# hem tabloyu küçültmek hem de belirli değer üzerindekileri görmek için threshold uyguluyoruz

threshold = 0.75    # 0.75 ve üzeri korelasyon olanları görelim
filtre = np.abs(corr_matrix["target"]) > threshold  # negatif-pozitif farketmez. o yüzden absolute value kullanıyoruz.(abs)
corr_features = corr_matrix.columns[filtre].tolist()

sns.clustermap(data[corr_features].corr(),annot=True,fmt=".2f")
plt.title("Correlation between features")
plt.show()

# birbirleriyle yüksek korelasyona sahip featureler varsa bunlar birbiriyle eşdüzlemdir. yani bunların arasında "MULTICOLLINEARITY" vardır. Bu dezavantajdır.

sns.pairplot(data,diag_kind="kde", markers="+")     # kde'yi histogram olarak düşün öyle çizecek, 
plt.show()             # bu tabloya bakınca target ile origin,cylinders arasındakilerde 3 değer var. Bunlar categorical veri olabilir yani !!!
# bu tablodan outlier,dağılım,ilişki yorumu yapılabilir.
"""
cylinders and origin can be categorical      # bunları feature engineering'de categoricale çeviricez demek bu
"""

# cylinders ve origini daha derinden inceleyelim (feature engineeringde ele alıcaz)
plt.figure()
sns.countplot(data["Cylinders"])
print(data["Cylinders"].value_counts())

plt.figure()
sns.countplot(data["Origin"])
print(data["Origin"].value_counts())

# boxplot           - outlierlere bakabilmek için
for c in data.columns:            
    plt.figure()
    sns.boxplot(x=c,data=data,orient="v")    # x=feature'ımız      # 8 tane boxplotumuz oluştu
"""
tabloda outlierleri gördük. Outlier: horsepower and acceleration 
"""

# %% OUTLIER

thr = 2        # önce threshold belirliyoruz    # en alt değer 1.5, çok veri kaybetmek istemiyorsak duruma göre değiştiririz

horsepower_desc = describe["HorsePower"]   # IQR hesabı için yüzde 75 ve 25 e bakıcaz - bunlar 4. ve 6. indexte (Q1-Q3)
q3_hp = horsepower_desc[6]
q1_hp = horsepower_desc[4]
IQR_hp = q3_hp - q1_hp
top_limit_hp = q3_hp + thr*IQR_hp                   # üst limiti bulucaz outlierin
bottom_limit_hp = q1_hp - thr*IQR_hp                   # alt limiti bulucaz outlierin
filter_hp_bottom = bottom_limit_hp < data["HorsePower"]
filter_hp_top = data["HorsePower"] < top_limit_hp             # alt ve üst için filtre yaptık. Bu sınırlar içindeyse sıkıntı yok outlier değil demek
filter_hp = filter_hp_bottom & filter_hp_top       # alt ve üst limit filtrelerini birleştiriyoruz.

data = data[filter_hp]                  # en son datamıza bu filtreyi uyguluyoruz
## bunu uygulayınca data boyutu 398 den 397 ye düştü. 1 tane outlier varmıs yani horsepower için

acceleration_desc = describe["Acceleration"]   # IQR hesabı için yüzde 75 ve 25 e bakıcaz - bunlar 4. ve 6. indexte (Q1-Q3)
q3_acc = acceleration_desc[6]
q1_acc = acceleration_desc[4]
IQR_acc = q3_acc - q1_acc   # q3-q1
top_limit_acc = q3_acc + thr*IQR_acc                   # üst limiti bulucaz outlierin
bottom_limit_acc = q1_acc - thr*IQR_acc                   # alt limiti bulucaz outlierin
filter_acc_bottom = bottom_limit_acc < data["Acceleration"]
filter_acc_top = data["Acceleration"] < top_limit_acc             # alt ve üst için filtre yaptık. Bu sınırlar içindeyse sıkıntı yok outlier değil demek
filter_acc = filter_acc_bottom & filter_acc_top       # alt ve üst limit filtrelerini birleştiriyoruz.

data = data[filter_acc]            # data boyuyu bu kez de 395 e düştü 397'den. 2 outlier da accelerationda varmış.

"""
1 tane hp'de, 2 tane acc'de outlier çıktı - outlierleri çıkardık
"""

# %% FEATURE ENGINEERING 
# 2 şey incelicez ; dependent-independent featurelarda bulunan skewness(çarpıklık)  , onehotencoding

"""
kuyruk kısımları genelde outlier verileri barındırır
"""

# target dependent variable
sns.distplot(data.target,fit=norm)  # targetin dagılımına bakıcaz  # bir de displot var, onda çizgi çekmiyor ve grafik biraz daha farklı # fit=norm ile normal dagılımı da çizerek karsılastırdık

(mu,sigma) = norm.fit(data["target"])                             # mu ve sigma degerlerini bulucaz
print("mu: {}, sigma: {} ".format(mu,sigma))

"""
"mu" (μ), normal dağılımın ortalama değerini temsil eder. Bu değer, veri kümesinin merkezini ifade eder. Normal dağılımın grafiğinde, "mu" değeri verilerin zirve noktasının yerini gösterir.

"sigma" (σ), normal dağılımın standart sapmasını ifade eder. Standart sapma, verilerin ne kadar yayıldığını gösterir. Daha büyük bir standart sapma, verilerin daha fazla dağıldığını gösterir. Normal dağılımın grafiğinde, "sigma" değeri eğrinin genişliğini kontrol eder.
"""

# qq plot                  --- dagılıma histogram üzerinden bakabiliyoruz üstteki gibi. qq plotla da buna bakabiliriz.
plt.figure()
stats.probplot(data["target"],plot=plt) 
plt.show()          # tabloda soldakiler bizim datanın quantile'ları, alttakide normal dagılıma ait quantilelar. Kırmızı çizgiye otursa normal dagılım derdik


data["target"] = np.log1p(data["target"])    # loglayıp skewnessi azaltmayı deniyoruz

plt.figure()
sns.distplot(data.target,fit=norm)      # normal dagılıma daha cok benzemeye basladı

(mu,sigma) = norm.fit(data["target"])                             # mu ve sigma degerlerine bakıyoruz tekrar
print("mu: {}, sigma: {} ".format(mu,sigma))

plt.figure()
stats.probplot(data["target"],plot=plt) 
plt.show()          # tekrar bakınca kırmızı çizgiye daha oturdugunu görüyoruz

# independent feature'larımızın skewness değerlerine bakıcaz
skewed_features = data.apply(lambda x: skew(x.dropna())).sort_values(ascending=False)   # burada na value yok ama yine de olsaydı skewnessi etkileyeceği için dropna yaptık gösterdik  # ascendingi false yaparak büyükten küçüğe sırala dedik 
skewness = pd.DataFrame(skewed_features,columns=["skewed"])     # skewed_features'ımızı dataframe içine attık
       # target'ı düzelttiğimiz için neredeyse skew'i 0. diğerlerinde de skew>1 ise poz.skew, skew<-1 ise neg.skew  

"""
hoursepower 1'in bi tık üstünde düzeltmeye gerek yok. Olsaydı Box Cox Transformation kullanıcaktık. Sonraki projelerde skewneesi düzeltmek gerektiğinde bunu kullanırız.
"""

# %% OHE     - one hot encoding
"""
categorical verilere bu işlem uygulanarak bu sütun kalkar ve yerine onları temsil eden taha fazla sütun gelir.
ÖRN: origin categorical demiştik. 1-2-3 degerleri var mesela. encode işlemi sonrası bu sütun kalkar ve origin1-o2-o3 sütunları gelir.
Bu sütunlarda 1'i 0-0-1 ; 2'yi 0-1-0 ; 3'ü 1-0-0 temsil eder gibi.

Bunu yapma sebebi categorical veriler bizim modelimizi bozacaktır. (origin 1-2-3 değeri alınca yanlışsa mesela error 1 vericek. ama 2. ve 3. ile aradaki fark 1-2 olacagı için daha fazla error gibi gözükebilir.)
"""

data["Cylinders"] = data["Cylinders"].astype(str)
data["Origin"] = data["Origin"].astype(str)         # bu veriler categorical olmadıgı için aşağıdaki satır otomatik çalısmayacaktı. ilk bunları categoricale çevirdik

data = pd.get_dummies(data)    # normalde bunu yapınca datadaki categorical feature'ları otomatik ohe'ye sokar ama bizim datada böyle veri yoktu. biz yorumlayarak categorical olarak ele alabiliriz dedik. 
                                
# %% SPLIT - STAND

# split
x=data.drop(["target"],axis=1)       # verimizi train edeceğimiz independent feature'larımız
y=data.target                   # dependent feature

X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=0.9,random_state=42)      # problemimiz zaten zor olmadıgı için testsize'ı 0.9 verdik ki biraz zorlansın, az veri ile train etsin.

# standardization

scaler = RobustScaler()      # RobustScaler
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)     # burda sadece transform ediyoruz. Çünkü sclaer'ımız Xtraine göre zaten ayarlanmıştı
## mean ve std'leri ayarlanmış oldu. mean'i 0 std'si 1 oldu.

# %% REGRESSION MODELS

#LR  - linear regression
"""
dependent-independent variablelar arasında bir line fit etmeye yarar. 
lr amacı ; errorların hepsini minimalize etmektir. en küçük error değerine sahip line'ı fit etmeye calısır.
least-squared erroru minimize etmeye calısıyoruz; least_squared error = (fit ettiğimiz line ile gerçek pointler arasındaki errorların karelerinin toplamı)
"""

lr=LinearRegression()
lr.fit(X_train,Y_train)
print("LR Coef: ",lr.coef_)
y_predicted_dummy = lr.predict(X_test)  # test verisetimizi kullanarak predictiona bakıyoruz 
mse = mean_squared_error(Y_test,y_predicted_dummy)
print("Linear Regression MSE: ",mse)       # mse 0.02 elde ettik

###### mean-squared error = least-squared error

#RR       - ridge regression(L2)     --- regularization yöntemleri overfitting önleyen, varyansı azaltan yöntemlerdir. 
"""
overfiti engellememizi sağlayan tekniktir.  - regularization tekniğidir.
(least-squared error + lambda.(slope)**2)    -  rr'nin amacı bunu minimize etmek. lr sadece least-square erroru minimize eder
lr ile karsılastırınca hem overfittingi engelleyip hem test datasetteki erroru daha azaltmıs olabiliriz.

"""

ridge = Ridge(random_state=42,max_iter=10000)    # 10k fazla ama bunu yapabileceği maksimum train(fitting) işlemi diyebiliriz 

alphas = np.logspace(-4,0.5,30)        # Amacımız alpha değerini tune ederek GridSearchCV ile best parametreleri almasını sağlamak # -4'den 0.5'e kadar gelsin ve 30 tane oluştursun

tuned_parameters = [{'alpha':alphas}]         # GridSearchCV içine dict olarak alır.
n_folds = 5

clf = GridSearchCV(ridge, tuned_parameters, cv=n_folds, scoring="neg_mean_squared_error", refit=True)     # ridge yöntemini kullanıcaz # tuned_parameters ile alphayı tune edicez # neg_mean_square_error normal meanin tam tersi # refit ile eğer bu clf'yi ileride kullanırsak alacağı parametreler, eğer false yaparsak clf'yi bidaha kullanamayız
clf.fit(X_train,Y_train)
scores = clf.cv_results_["mean_test_score"]
score_std = clf.cv_results_["std_test_score"]

print("Ridge Coef: ",clf.best_estimator_.coef_)      # best parametreleri almak istiyoruz ve katsayılarına bakalım
ridge = clf.best_estimator_
print("Ridge Best Estimator: ",ridge)

y_predicted_dummy = clf.predict(X_test)       # clf otomatik olarak en iyi parametreleri kullanarak testi gerçekleştiricek
mse = mean_squared_error(Y_test,y_predicted_dummy)
print("Ridge MSE: ",mse)
print("--------------------------------------------")

# son olarak alphanın skora göre nasıl değiştigini gözlemleyelim alttaki kod ile
plt.figure()
plt.semilogx(alphas,scores)    # alphayı atarken logspace ile atamıştık o yüzden çizerken de bununla çizmek zorundayız
plt.xlabel("alpha") 
plt.ylabel("score")
plt.title("Ridge")        # ridge coef degerlerini lasso ile değerlendiriyoruz normalde birazdan bakıcaz ona da. şu an bu rakamlardan bişey anlamayız
                       # mse 0.018 çıktı, az önce 0.02ydi(lr'de). Az da olsa bi başarı sağladık

# LR(L1)          - lasso regression

"""
(least-squared error + lambda.|slope|)    - bu denklemi minimize etmeyi amaçlar. rr'den tek farkı slope mutlak olarak işlemde.
rr'ye göre avantajı burada FeatureSelection kullanılabilir. Bunun sebebi lasso'da gereksiz coef'lere 0 değeri atanabilir. rr'de bu olmuyor 0.00001 vs oluyo
üstteki yazıya göre lassoda bir independent variablenin karsılıgı 0 oluyorsa dependent variableye etkisi yok demektir.
!!!! eğer high correlated variablelar varsa ki vardı, lasso en önemlilerini alıp kalanları 0'a atıyor.
overfitting'i ridge gibi bu da önler. zaten regularization yöntemlerinin asıl amacı bu
"""

lasso = Lasso(random_state=42,max_iter=10000)       # lasso regressionu hazırladık
alphas = np.logspace(-4,-0.5,30)

tuned_parameters = [{'alpha':alphas}]
n_folds = 5

clf = GridSearchCV(lasso, tuned_parameters, cv=n_folds, scoring='neg_mean_squared_error',refit=True)
clf.fit(X_train,Y_train)
scores = clf.cv_results_['mean_test_score']
scores_std = clf.cv_results_['std_test_score']

print("Lasso coef: ",clf.best_estimator_.coef_)
lasso = clf.best_estimator_
print("Lasso Best Estimator: ",lasso)

y_predicted_dummy = clf.predict(X_test)
mse = mean_squared_error(Y_test, y_predicted_dummy)
print("Lasso MSE: ",mse)
print("-----------------------------------------------------------")

plt.figure()
plt.semilogx(alphas,scores)
plt.xlabel("alpha")
plt.ylabel("score")
plt.title("Lasso")       # esn son 5 satırdada alphanın skora göre değişimini görmek için plot çizdiriyoruz
                     # bu sonuçta da mse 0.017 cıktı. az da olsa giderek düşüyor.
                       # lassoda 0. şeklinde yazıyor rr'deki gibi devamı yok. gereksizleri direkt atıyor yani

# ElasticNet         - bu da regularization yöntemi

"""
(least-squared error + lambda1.(slope)**2 + lambda2.|slope|)     değerini minimize etmeyi amaçlar. -- lasso ve ridge in karısımı gibi
high correlated feature'larda çok işe yarıyor. feature'ların çıkartılmasında çok kullanılıyor
"""

parametersGrid = {"alpha":alphas,
                  "l1_ratio": np.arange(0.0,1.0,0.05)}       # l1_ratio = l1 ve l2 arasındaki ratio demek

eNet = ElasticNet(random_state=42, max_iter=10000)
clf = GridSearchCV(eNet, parametersGrid, cv=n_folds, scoring="neg_mean_squared_error",refit=True)
clf.fit(X_train,Y_train)

print("ElasticNet Coef: ",clf.best_estimator_.coef_)
eNet = clf.best_estimator_
print("ElasticNet Best Estimator: ",eNet)

y_predicted_dummy = clf.predict(X_test)
mse = mean_squared_error(Y_test, y_predicted_dummy)
print("ElasticNet MSE: ",mse)              # mse 0.0174 çıktı, şu ana kadarki en iyi sonuç


"""
    StandarScaler ile:
        linear reg mse = 0.0206
        Ridge reg mse = 0.0181
        Lasso reg mse = 0.0175
        Elasticnet reg mse = 0.0174
"""

### RobustScaler ile outlier ları da veriden uzaklastırınca daha iyi sonuc cıkıyor. üstteki scaler=StandartScaler'i Robust ile değiştirip bak.
### RobustScaler'da en düşük mse Lassoda cıkıyor yani en iyi değer onda cıkıyor.


# %% XGBoost

"""
büyük karmasık verisetleri için bir algoritmadır.   
"""

parameters_Grid = {'nthread':[4],      # when use hyperthread, xgboost my become slower
               'objective':['reg:linear'],
               'learning_rate': [.03,0.05,.07],
               'max_depth':[5,6,7],
               'min_child_weight':[4],
               'silent':[1],
               'subsample':[0.7],
               'n_estimators':[500,1000]}


#model_xgb = xgb.XGBRegressor(objective='reg:linear',max_depth=5,min_child_weight=4,subsample=0.7,n_estimators=1000,learning_rate=0.07)  # görevimiz lin reg yapmak objective ile # max 5'e insin diyoruz  # 1000 tane ağaç olsun #lr'miz 0.07 olsun
model_xgb = xgb.XGBRegressor()# normalde üstteki gibiydi mse kötü cıkınca tune ettik mecbur üstte. degerleri kendimiz üstte verdik yani o yüzden içini boş bıraktık

clf = GridSearchCV(model_xgb, parameters_Grid, cv=n_folds, scoring="neg_mean_squared_error",refit=True)
clf.fit(X_train,Y_train)
model_xgb = clf.best_estimator_      # xgb en iyi degerleri alsın diye bu satırı ekliyoruz
"""
grid search'i önce olusturup eğiticez ondan sonra best parametreleri vericez.
"""
y_predicted_dummy = clf.predict(X_test)
mse = mean_squared_error(Y_test, y_predicted_dummy)
print("XGBRegressor MSE: ",mse)      # xgb mse 0.019 cıktı kötü bi değer. Bu yüzden parametreleri tune edicez.
                                        # tune edince 0.018 oldu biraz daha iyi


# %% Averaging Models

## xgboost ve lassonun ortalamasını alıcaz. en iyi lasso çıkmıstı geçen robotscaler ile.

class AveragingModels():
    def __init__(self,models):       # dısarıdan modellerimizi aldık - lasso ve xgb
        self.models = models
        
    # we define clones of the original models to fit the data in 
    def fit(self,X,y):
        self.models_ = [clone(x) for x in self.models]   # bu modelleri kullanarak modeli fit edicez. 

        # train cloned base models
        for model in self.models_:
            model.fit(X,y)

        return self

    # now we do the predictions for cloned models and average them
    def predict(self,X):
        predictions = np.column_stack([model.predict(X) for model in self.models_])
        return np.mean(predictions, axis=1)

averaged_models = AveragingModels(models=(model_xgb,lasso))
averaged_models.fit(X_train, Y_train)         # train işlemini yaptık

y_predicted_dummy = averaged_models.predict(X_test)
mse = mean_squared_error(Y_test, y_predicted_dummy)
print("Averaged Models MSE: ",mse)               # test işlemini yaptık   # mse en düşük burda cıktı. en iyi sonuc = 0.015


































