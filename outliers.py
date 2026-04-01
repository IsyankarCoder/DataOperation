import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.cbook import boxplot_stats
import random
from random import sample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler




#Nitelik                         Türkçesi                                                         Kısaltma         Veri Tipi
#-----------------------------------------------------------------------------------------------------------------------------
#miles per gallon (mpg)          mil cinsinden şehir içindeki yakıt tüketimi (galon başına mil)   mpg              Sürekli
#cylinders                       silindirler                                                      cylinders        Ayrık 
#displacement                    motor deplasmani                                                 displacement     Sürekli
#horsepower                      beygir gücü                                                      horsepower       Sürekli
#weight                          ağirlik                                                          weight           Sürekli
#acceleration                    hizlanma                                                         acceleration     Sürekli
#model year                      model yili                                                       model_year       Ayrik
#origin                          menşei                                                           origin           Ayrik
#car name                        araba adi                                                        car_name         Metin 

nitelikadlari = ["mpg","cylinders","displacement","horsepower","weight","acceleration","model_year","origin","car_name"]

#"\s+" değeri, verilerin bir veya daha fazla boşluk karakteriyle ayrıldığını belirtmek için kullanılmıştır.
# Veri dosyasında sütun adları bulunmadığından header = None olarak alınmıştır.
# Sütun adlarının verildiği names parametresine ilk satırda nitelik isimlerinin saklandığı nitelikAdlari isimli değişken verilmiştir.
# decimal parametresiyle de verideki nümerik niteliklerin ondalıklı sayı ayıracı olarak nokta (.)
rows = []
with open("auto-mpg.data", "r", encoding="utf-8") as f:
	for line in f:
		line = line.strip()
		if not line:
			continue
		# split into left (first 8 fields) and car name (rest, often quoted)
		if "\t" in line:
			left, name = line.split("\t", 1)
		else:
			parts = line.rsplit('"', 1)
			if len(parts) == 2:
				left = parts[0]
				name = '"' + parts[1]
			else:
				left = line
				name = ''
		tokens = left.split()
		if len(tokens) < 8:
			continue
		first8 = tokens[:8]
		car_name = name.strip().strip('"')
		rows.append(first8 + [car_name])

verisetim = pd.DataFrame(rows, columns=nitelikadlari)
for col in ["mpg","cylinders","displacement","horsepower","weight","acceleration","model_year","origin"]:
	verisetim[col] = pd.to_numeric(verisetim[col].replace('?', np.nan), errors='coerce')
# `car_name` sütununda aynı ada sahip birden fazla satır olabilir.
# Bazı pandas işlemleri duplicate index etiketleriyle reindex yapmaya çalışırken hata verir
# ("cannot reindex on an axis with duplicate labels"). Bu yüzden önce `horsepower`
# kolonu sayısala çevrilip eksik değerler atılıyor, sonra index benzersizleştiriliyor.
verisetim['horsepower'] = pd.to_numeric(verisetim['horsepower'], errors='coerce')
verisetim = verisetim.dropna(subset=['horsepower'])

try:
	verisetim.set_index("car_name", inplace=True)
except ValueError as e:
	# Hata bilgisini yazdır
	print("ValueError while setting index:", e)
	# Daha fazla bağlam isterseniz traceback yazdırabilirsiniz:
	# import traceback; traceback.print_exc()
	# Eğer duplicate label hatası gelirse, aynı isimlerin arkasına bir sayac ekleyerek benzersiz index oluştur
	verisetim['car_name_unique'] = verisetim['car_name'] + '_' + verisetim.groupby('car_name').cumcount().astype(str)
	verisetim.set_index('car_name_unique', inplace=True)

q1 = verisetim["horsepower"].quantile(0.25)
q3=verisetim["horsepower"].quantile(0.75)

IQR =q3-q1
alt = q1-1.5*IQR
ust=  q3+1.5*IQR


ust_aykirideger = np.where(verisetim["horsepower"]>=ust)[0]
alt_aykirideger = np.where(verisetim["horsepower"]<=alt)[0]

print(ust_aykirideger)
print("--------------------------------")
print(alt_aykirideger)

sns.boxplot(y="horsepower",data=verisetim,palette="magma")
sns.set_style("whitegrid")
plt.figure(figsize=(6,8))
sns.boxplot(y=verisetim["horsepower"], palette="magma")
plt.title("Horsepower Boxplot")
plt.ylabel("Horsepower")
plt.tight_layout()
plt.show()
 
 
#eğer veri değerlerini verisetinden kaldırmak istersek 
#verisetim.drop(index = ust_aykirideger,inplace=True) 
#verisetim.drop(index = alt_aykirideger,inplace=True)
