# Sinyaller_Sistemler_Proje
Dönem sonu projesi-gürültü giderme ve ses temizleme sistemi.

## Gerekli Kütüphaneler

Gerekli kütüphaneleri yüklemek için :
pip install soundfile 
pip install numpy 
pip install sounddevice

## gürültü_engelleme.py Nasıl çalışır ?

Programı herhangi bir ide üzerinden çalıştırın (ya da terminalde py gürültü.py). Sonrasında gürültüsü giderilecek olan dosya adını giriniz (örn : kalabalık.wav) ardından temizlenmiş sesin kaydedileceği dosyanın ismini giriniz (örn : kalabalık_temiz.wav) böylece temizlenmiş ses dosyaya kaydedilecektir.

## gercek_zamanlı_gürültü_engelleme.py Nasıl çalışır ? 

Programın en iyi çalıştığı durum program çalıştırıldığında kalibrasyon süresince ortamda gürültünün bulunduğu durumdur. Program çalıştırıldığında kalibrasyon süresi kadar bekleyiniz (değiştirilmediyse 3sn) sonrasında program siz keyboard interrupt yapana kadar (CTRL + C) kaydettiği sesi temizlenmiş halde çıkış cihazınıza verecektir. Program sonlandığında gerçek zamanlı filtrelenecek sesin kaydedileceği dosya ismini giriniz (örn : gercek_zaman.wav) ardından sonucu görmek için kaydedilen sese bakabilirsiniz

*******UYARI*********

Bu programlar kişisel mikrofonum üzerinden test edilmiştir farklı mikrofonlar üzerinde farklı sonuçlar alınabilmektedir bu nedenle kaydettiğim tüm seslerin temiz halini de paylaşıyorum. Kendi mikrofonunuz üzerinde düzgün çalışmıyorsa gürültü engellemede kullanılan parametreler üzerinde oynama yapabilirsiniz.
