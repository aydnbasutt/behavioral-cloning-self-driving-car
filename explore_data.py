import pandas as pd
import matplotlib.pyplot as plt

# CSV'yi oku. Simülatör başlık satırı (header) eklemiyor,
# o yüzden sütun isimlerini biz veriyoruz.
columns = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
data = pd.read_csv('data/driving_log.csv', names=columns)

# Genel bilgi
print("Toplam satır sayısı:", len(data))
print("\nİlk 5 satır:")
print(data.head())

print("\nDireksiyon açısı istatistikleri:")
print(data['steering'].describe())

# Direksiyon açılarının dağılımını görselleştir
plt.hist(data['steering'], bins=25)
plt.xlabel('Direksiyon Açısı')
plt.ylabel('Adet')
plt.title('Direksiyon Açısı Dağılımı')
plt.show()