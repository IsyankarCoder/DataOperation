import numpy as np
import pandas as pd
import seaborn as sns
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
verisetim.set_index("car_name", inplace=True)

bolumkategori = ["Düşük","Orta","Yüksek"]
bolmeler = [8,23.4,29.9,46.6] 
verisetim["durum"] = pd.cut(verisetim["mpg"],bins=bolmeler,labels=bolumkategori) 
  
pd.set_option("display.max_columns",20)
print(verisetim.describe(include="all"))

duplicate_first = verisetim.duplicated(keep="first")
duplicate_last =verisetim.duplicated(keep="last")

print(duplicate_first)
print("--------------------------")
print(duplicate_last)

print("--------------------------")
indislerim= duplicate_first | duplicate_last
print(verisetim[indislerim])

#verisetim.drop_duplicates()