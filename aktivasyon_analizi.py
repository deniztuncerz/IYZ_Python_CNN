"""
CNN Analizi - Aktivasyon Fonksiyonları ile


Veri Seti: Hafi, S.J., Mohammed, M.A., Abd, D.H., vd. (2024). 
"Image dataset of healthy and infected fig leaves with Ficus leaf worm". 
Data in Brief, Cilt 53.

Test Edilen Aktivasyon Fonksiyonları:
- Identity 
- Binary
- Lojistik
- TanH
- ArcTan
- ReLU
- PReLU
- ELU
- SoftPlus


"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.utils import to_categorical
import warnings
warnings.filterwarnings('ignore')

# rastgele 
np.random.seed(42)
tf.random.set_seed(42)

class KapsamliAktivasyonAnalizi:
    
    def __init__(self, veri_yolu=".", resim_boyutu=(224, 224), batch_boyutu=32):
        
        self.veri_yolu = veri_yolu
        self.resim_boyutu = resim_boyutu
        self.batch_boyutu = batch_boyutu
        self.modeller = {}
        self.gecmisler = {}
        self.sonuclar = {}
        self.sinif_isimleri = ['Sağlıklı', 'Enfekte']
        
        # aktivasyon fonksiyonları
        self.aktivasyon_fonksiyonlari = {
            'identity': 'linear',
            'binary': 'binary_step', 
            'lojistik': 'sigmoid',
            'tanh': 'tanh',
            'arctan': 'arctan',  
            'relu': 'relu',
            'prelu': 'prelu', 
            'elu': 'elu',
            'softplus': 'softplus'
        }
        
    def binary_aktivasyon(self, x):
        
        return tf.cast(tf.greater(x, 0), tf.float32)
    
    def arctan_aktivasyon(self, x):
        
        return tf.atan(x)
    
    def veri_yukle_ve_onisle(self):
        """
        veri setini yükle
        """
        print("Veri seti yükleniyor...")
        
        # enfekte ve saglıklı resimleri yükle
        saglikli_yol = os.path.join(self.veri_yolu, 'healthy')
        enfekte_yol = os.path.join(self.veri_yolu, 'infected')
        
        resimler = []
        etiketler = []
        
        # sağlıklı resimleri yükle
        for dosya_adi in os.listdir(saglikli_yol):
            if dosya_adi.lower().endswith('.jpg'):
                resim_yolu = os.path.join(saglikli_yol, dosya_adi)
                resim = cv2.imread(resim_yolu)
                if resim is not None:
                    resim = cv2.resize(resim, self.resim_boyutu)
                    resim = cv2.cvtColor(resim, cv2.COLOR_BGR2RGB)
                    resimler.append(resim)
                    etiketler.append(0)  
        
        # enfekte resimleri yükle
        for dosya_adi in os.listdir(enfekte_yol):
            if dosya_adi.lower().endswith('.jpg'):
                resim_yolu = os.path.join(enfekte_yol, dosya_adi)
                resim = cv2.imread(resim_yolu)
                if resim is not None:
                    resim = cv2.resize(resim, self.resim_boyutu)
                    resim = cv2.cvtColor(resim, cv2.COLOR_BGR2RGB)
                    resimler.append(resim)
                    etiketler.append(1) 
        
        # numpy dizilerine dönüşüm
        self.X = np.array(resimler, dtype=np.float32) / 255.0
        self.y = np.array(etiketler)
        
        print(f"Yüklenen resim sayısı: {len(self.X)}")
        print(f"Sağlıklı: {np.sum(self.y == 0)} resim")
        print(f"Enfekte: {np.sum(self.y == 1)} resim")
        
        # eğitim doğrulama ve test setlerine bölelim
        X_gecici, self.X_test, y_gecici, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        self.X_egitim, self.X_dogrulama, self.y_egitim, self.y_dogrulama = train_test_split(
            X_gecici, y_gecici, test_size=0.2, random_state=42, stratify=y_gecici
        )
        
        print(f"Eğitim seti: {len(self.X_egitim)} resim")
        print(f"Doğrulama seti: {len(self.X_dogrulama)} resim") 
        print(f"Test seti: {len(self.X_test)} resim")
        
       
        self.y_egitim_kategorik = to_categorical(self.y_egitim, 2)
        self.y_dogrulama_kategorik = to_categorical(self.y_dogrulama, 2)
        self.y_test_kategorik = to_categorical(self.y_test, 2)
        
    def model_olustur(self, aktivasyon_fonksiyonu):
        """
        aktivasyon fonksiyonu ile CNN modeli oluşturalım
        
        """
        model = models.Sequential()
        
    
        model.add(layers.Conv2D(32, (3, 3), activation=aktivasyon_fonksiyonu, 
                               input_shape=(*self.resim_boyutu, 3)))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        
        
        model.add(layers.Conv2D(64, (3, 3), activation=aktivasyon_fonksiyonu))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        
        
        model.add(layers.Conv2D(128, (3, 3), activation=aktivasyon_fonksiyonu))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        
        
        model.add(layers.Conv2D(256, (3, 3), activation=aktivasyon_fonksiyonu))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        
        # düzleştirme
        model.add(layers.Flatten())
        
        # dropout 
        model.add(layers.Dense(512, activation=aktivasyon_fonksiyonu))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(256, activation=aktivasyon_fonksiyonu))
        model.add(layers.Dropout(0.3))
        
        # sınıflandırma katmanı
        model.add(layers.Dense(2, activation='softmax'))
        
        return model
    
    def ozel_aktivasyon_modeli_olustur(self, aktivasyon_adi):
        
        model = models.Sequential()
        
       
        model.add(layers.Conv2D(32, (3, 3), input_shape=(*self.resim_boyutu, 3)))
        if aktivasyon_adi == 'binary':
            model.add(layers.Lambda(self.binary_aktivasyon))
        elif aktivasyon_adi == 'arctan':
            model.add(layers.Lambda(self.arctan_aktivasyon))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        
        
        model.add(layers.Conv2D(64, (3, 3)))
        if aktivasyon_adi == 'binary':
            model.add(layers.Lambda(self.binary_aktivasyon))
        elif aktivasyon_adi == 'arctan':
            model.add(layers.Lambda(self.arctan_aktivasyon))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        
        
        model.add(layers.Conv2D(128, (3, 3)))
        if aktivasyon_adi == 'binary':
            model.add(layers.Lambda(self.binary_aktivasyon))
        elif aktivasyon_adi == 'arctan':
            model.add(layers.Lambda(self.arctan_aktivasyon))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        
        
        model.add(layers.Conv2D(256, (3, 3)))
        if aktivasyon_adi == 'binary':
            model.add(layers.Lambda(self.binary_aktivasyon))
        elif aktivasyon_adi == 'arctan':
            model.add(layers.Lambda(self.arctan_aktivasyon))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        
        
        model.add(layers.Flatten())
        
        # dropout 
        model.add(layers.Dense(512))
        if aktivasyon_adi == 'binary':
            model.add(layers.Lambda(self.binary_aktivasyon))
        elif aktivasyon_adi == 'arctan':
            model.add(layers.Lambda(self.arctan_aktivasyon))
        model.add(layers.Dropout(0.5))
        
        model.add(layers.Dense(256))
        if aktivasyon_adi == 'binary':
            model.add(layers.Lambda(self.binary_aktivasyon))
        elif aktivasyon_adi == 'arctan':
            model.add(layers.Lambda(self.arctan_aktivasyon))
        model.add(layers.Dropout(0.3))
        
        # Softmax
        model.add(layers.Dense(2, activation='softmax'))
        
        return model
    
    def prelu_modeli_olustur(self):
        
        model = models.Sequential()
        
        
        model.add(layers.Conv2D(32, (3, 3), input_shape=(*self.resim_boyutu, 3)))
        model.add(layers.PReLU())
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        
        
        model.add(layers.Conv2D(64, (3, 3)))
        model.add(layers.PReLU())
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        
        
        model.add(layers.Conv2D(128, (3, 3)))
        model.add(layers.PReLU())
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        
        
        model.add(layers.Conv2D(256, (3, 3)))
        model.add(layers.PReLU())
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2, 2)))
        
        # Düzleştirme 
        model.add(layers.Flatten())
        
        # Dropout 
        model.add(layers.Dense(512))
        model.add(layers.PReLU())
        model.add(layers.Dropout(0.5))
        
        model.add(layers.Dense(256))
        model.add(layers.PReLU())
        model.add(layers.Dropout(0.3))
        
        # Softmax 
        model.add(layers.Dense(2, activation='softmax'))
        
        return model
    
    def modeli_derle(self, model, aktivasyon_adi, optimizer_tipi='adam'):
        
        if optimizer_tipi == 'adam':
            optimizer = optimizers.Adam(learning_rate=0.001)
        elif optimizer_tipi == 'sgd':
            optimizer = optimizers.SGD(learning_rate=0.01, momentum=0.9)
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"\n{aktivasyon_adi.upper()} Model Mimarisi (Optimizer: {optimizer_tipi.upper()}):")
        model.summary()
        
    def modeli_egit(self, model, aktivasyon_adi, epochs=50, optimizer_tipi='adam'):
        
        print(f"\n{aktivasyon_adi.upper()} modeli {optimizer_tipi.upper()} optimizer ile eğitiliyor...")
        
        
        callback_listesi = [
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Model eğitimi
        gecmis = model.fit(
            self.X_egitim, self.y_egitim_kategorik,
            validation_data=(self.X_dogrulama, self.y_dogrulama_kategorik),
            epochs=epochs,
            batch_size=self.batch_boyutu,
            callbacks=callback_listesi,
            verbose=1
        )
        
        
        self.gecmisler[f"{aktivasyon_adi}_{optimizer_tipi}"] = gecmis
        
        # Test setinde değerlendirelim
        test_kayip, test_dogruluk = model.evaluate(
            self.X_test, self.y_test_kategorik, verbose=0
        )
        
        # tahminler
        y_tahmin = model.predict(self.X_test, verbose=0)
        y_tahmin_siniflar = np.argmax(y_tahmin, axis=1)
        
        # metrikleri hesaplama
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.y_test, y_tahmin_siniflar, average='macro'
        )
        
        # Sonuçlar
        self.sonuclar[f"{aktivasyon_adi}_{optimizer_tipi}"] = {
            'test_dogruluk': test_dogruluk,
            'test_kayip': test_kayip,
            'macro_precision': precision,
            'macro_recall': recall,
            'macro_f1': f1,
            'y_tahmin': y_tahmin_siniflar
        }
        
        print(f"{aktivasyon_adi.upper()} ({optimizer_tipi.upper()}) - Test Doğruluğu: {test_dogruluk:.4f}")
        print(f"Macro Precision: {precision:.4f}, Macro Recall: {recall:.4f}, Macro F1: {f1:.4f}")
        
    def tum_modelleri_egit(self, epochs=50):
        
        print("eğitim başlatılıyor...")
        
        for aktivasyon_adi, aktivasyon_fonksiyonu in self.aktivasyon_fonksiyonlari.items():
            for optimizer_tipi in ['adam', 'sgd']:
                print(f"\n{'='*60}")
                print(f"{aktivasyon_adi.upper()} {optimizer_tipi.upper()} ile eğitiliyor")
                print(f"{'='*60}")
                
                # aktivasyon fonksiyonuna göre model luşturma
                if aktivasyon_adi == 'prelu':
                    model = self.prelu_modeli_olustur()
                elif aktivasyon_adi in ['binary', 'arctan']:
                    model = self.ozel_aktivasyon_modeli_olustur(aktivasyon_adi)
                else:
                    model = self.model_olustur(aktivasyon_fonksiyonu)
                
                # eğit
                self.modeli_derle(model, aktivasyon_adi, optimizer_tipi)
                self.modeli_egit(model, aktivasyon_adi, epochs, optimizer_tipi)
                self.modeller[f"{aktivasyon_adi}_{optimizer_tipi}"] = model
                
                tf.keras.backend.clear_session()
    
    def kapsamli_sonuclar_tablosu_olustur(self):
        
        print("\n" + "="*100)
        print("KAPSAMLI SONUÇLAR TABLOSU")
        print("="*100)
        
        # Sonuçlar 
        sonuc_verileri = []
        for model_adi, sonuclar in self.sonuclar.items():
            aktivasyon_adi, optimizer = model_adi.rsplit('_', 1)
            sonuc_verileri.append({
                'Aktivasyon Fonksiyonu': aktivasyon_adi,
                'Optimizer': optimizer.upper(),
                'Test Doğruluğu (%)': f"{sonuclar['test_dogruluk']*100:.2f}",
                'Macro Precision': f"{sonuclar['macro_precision']:.4f}",
                'Macro Recall': f"{sonuclar['macro_recall']:.4f}",
                'Macro F1': f"{sonuclar['macro_f1']:.4f}",
                'Test Kayıp': f"{sonuclar['test_kayip']:.4f}"
            })
        
        df_sonuclar = pd.DataFrame(sonuc_verileri)
        
        # doğruluğa göre sıralama
        df_sonuclar['Test Doğruluğu (%)'] = df_sonuclar['Test Doğruluğu (%)'].astype(float)
        df_sonuclar = df_sonuclar.sort_values('Test Doğruluğu (%)', ascending=False)
        
        print(df_sonuclar.to_string(index=False))
        
        # csv formatında kayıt
        df_sonuclar.to_csv('kapsamli_aktivasyon_sonuclari.csv', index=False)
        print(f"\nSonuçlar 'kapsamli_aktivasyon_sonuclari.csv' dosyasına kaydedildi")
        
        return df_sonuclar
    
    def egitim_gecmisi_ciz(self):
        
        print("\nEğitim geçmişi grafikleri oluşturuluyor...")
        
        # subplotlar 
        n_modeller = len(self.gecmisler)
        n_sutun = 3
        n_satir = (n_modeller + n_sutun - 1) // n_sutun
        
        fig, axes = plt.subplots(n_satir, n_sutun, figsize=(20, 5*n_satir))
        if n_satir == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for i, (model_adi, gecmis) in enumerate(self.gecmisler.items()):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            
            ax.plot(gecmis.history['accuracy'], label='Eğitim Doğruluğu')
            ax.plot(gecmis.history['val_accuracy'], label='Doğrulama Doğruluğu')
            ax.set_title(f'{model_adi.upper()}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Doğruluk')
            ax.legend()
            ax.grid(True)
        
        # Boş subplotları kaldırmak
        for i in range(len(self.gecmisler), len(axes)):
            axes[i].remove()
        
        plt.suptitle('Eğitim Geçmişi - Tüm Aktivasyon Fonksiyonları', fontsize=16)
        plt.tight_layout()
        plt.savefig('kapsamli_egitim_gecmisi.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def karisiklik_matrisleri_ciz(self):
        
        print("\nKarışıklık matrisleri oluşturuluyor...")
        
    
        n_modeller = len(self.sonuclar)
        n_sutun = 3
        n_satir = (n_modeller + n_sutun - 1) // n_sutun
        
        fig, axes = plt.subplots(n_satir, n_sutun, figsize=(15, 5*n_satir))
        if n_satir == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for i, (model_adi, sonuclar) in enumerate(self.sonuclar.items()):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Karışıklık matrisini hesapla
            cm = confusion_matrix(self.y_test, sonuclar['y_tahmin'])
            
            # Karışıklık matrisini çiz
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=self.sinif_isimleri, yticklabels=self.sinif_isimleri)
            ax.set_title(f'{model_adi.upper()}\nDoğruluk: {sonuclar["test_dogruluk"]*100:.2f}%')
            ax.set_xlabel('Tahmin Edilen')
            ax.set_ylabel('Gerçek')
        
        # Boş subplotları kaldırmak
        for i in range(len(self.sonuclar), len(axes)):
            axes[i].remove()
        
        plt.suptitle('Karışıklık Matrisleri - Tüm Modeller', fontsize=16)
        plt.tight_layout()
        plt.savefig('kapsamli_karisiklik_matrisleri.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def siniflandirma_raporlari_uret(self):
        
        print("\n" + "="*100)
        print("DETAYLI SINIFLANDIRMA RAPORLARI")
        print("="*100)
        
        for model_adi, sonuclar in self.sonuclar.items():
            print(f"\n{model_adi.upper()}:")
            print("-" * 50)
            
           
            rapor = classification_report(
                self.y_test, sonuclar['y_tahmin'], 
                target_names=self.sinif_isimleri,
                digits=4
            )
            print(rapor)
    
    def gradcam_olustur(self, model, resim, katman_adi='conv2d_3'):
        
        try:
            # Modelin son evrişim katmanını bul
            conv_layer = None
            for layer in model.layers:
                if katman_adi in layer.name.lower() and 'conv' in layer.name.lower():
                    conv_layer = layer
                    break
            
            if conv_layer is None:
                # Eğer belirtilen katman bulunamazsa, son evrişim katmanını bul
                conv_layers = [layer for layer in model.layers if 'conv' in layer.name.lower()]
                if conv_layers:
                    conv_layer = conv_layers[-1]
                else:
                    print("Evrişim katmanı bulunamadı!")
                    return None
            
            # GradCAM hesaplama
            with tf.GradientTape() as tape:
                # Son evrişim katmanının çıktısını al
                last_conv_layer_output = conv_layer.output
                tape.watch(last_conv_layer_output)
                
                # Modelin tahminini al
                predictions = model(resim)
                predicted_class = tf.argmax(predictions[0])
                class_channel = predictions[:, predicted_class]
            
            # Gradyanları hesapla
            grads = tape.gradient(class_channel, last_conv_layer_output)
            
            # Global average pooling uygula
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            # Son evrişim katmanının çıktısını al
            last_conv_layer_output = last_conv_layer_output[0]
            
            # Gradyanları feature map'e uygula
            heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            
            # Normalize et
            heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
            
            return heatmap.numpy()
            
        except Exception as e:
            print(f"GRADCAM oluşturulurken hata: {e}")
            return None
    
    def gradcam_karsilastirma_olustur(self, num_images=5):
       
        print("\nGRADCAM karşılaştırması oluşturuluyor...")
        
        # Enfekte resimlerden rastgele 
        enfekte_indices = np.where(self.y_test == 1)[0]
        if len(enfekte_indices) < num_images:
            num_images = len(enfekte_indices)
        
        selected_indices = np.random.choice(enfekte_indices, num_images, replace=False)
        
        # model seçimi
        best_models = []
        for aktivasyon in ['softplus', 'relu', 'elu', 'prelu', 'arctan', 'tanh']:
            for optimizer in ['adam', 'sgd']:
                model_key = f"{aktivasyon}_{optimizer}"
                if model_key in self.modeller:
                    best_models.append((model_key, self.modeller[model_key]))
                    break  # Her aktivasyon için sadece bir optimizer allmak için:
        
        
        best_models = best_models[:6]
        
        fig, axes = plt.subplots(len(best_models), num_images, figsize=(num_images * 3, len(best_models) * 3))
        if len(best_models) == 1:
            axes = axes.reshape(1, -1)
        if num_images == 1:
            axes = axes.reshape(-1, 1)
        
        for i, (model_adi, model) in enumerate(best_models):
            for j, img_idx in enumerate(selected_indices):
                ax = axes[i, j] if num_images > 1 else axes[i]
                
                # Orijinal resm
                orijinal_resim = self.X_test[img_idx]
                resim_array = np.expand_dims(orijinal_resim, axis=0)
                
                # GRADCAM 
                heatmap = self.gradcam_olustur(model, resim_array)
                
                if heatmap is not None:
                    
                    heatmap_resized = cv2.resize(heatmap, (self.resim_boyutu[1], self.resim_boyutu[0]))
                    
                    
                    heatmap_colored = plt.cm.jet(heatmap_resized)[:, :, :3]
                    
                    
                    overlay = 0.6 * orijinal_resim + 0.4 * heatmap_colored
                    
                    # görselleştirme
                    ax.imshow(overlay)
                    ax.set_title(f'{model_adi.upper()}\nResim {j+1}')
                    ax.axis('off')
                else:
                    ax.imshow(orijinal_resim)
                    ax.set_title(f'{model_adi.upper()}\nGRADCAM Hatası')
                    ax.axis('off')
        
        plt.suptitle('GRADCAM En İyi Performans Gösteren Modeller', fontsize=16)
        plt.tight_layout()
        plt.savefig('kapsamli_gradcam_karsilastirma.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("GRADCAM karşılaştırması 'kapsamli_gradcam_karsilastirma.png' olarak kaydedildi")
    
    def gradcam_analizi_yap(self):
        
        print("\n" + "="*100)
        print("GRADCAM ANALİZİ VE YORUMLAMA")
        print("="*100)
        
        print("\n1. GRADCAM :")
        print("- GRADCAM kullanılmıştır")
        print("- Modelin hangi bölgelere odaklandığı")
        print("- Kırmızı bölgeler: Yüksek önem")
        print("- Mavi bölgeler: Düşük önem")
        
        print("\n2. Beklenen GRADCAM Davranışları:")
        print("- Enfekte yapraklarda hastalıklı bölgelere odaklanma")
        print("- Sağlıklı yapraklarda genel yaprak yapısına odaklanma")
        print("- Arka plan ve alakasız bölgelere bakılmaması")
        
        print("\n3. Aktivasyon Fonksiyonlarına Göre Beklenen:")
        print("- SoftPlus/ReLU/ELU:")
        print("- TanH/ArcTan:")
        print("- Binary Step:")
        print("- Identity/Logistic")
        
        print("\n4. Optimizer Etkisi:")
        print("- ADAM: GRADCAM'ler")
        print("- SGD:  GRADCAM'ler")
        
        print("\n5. GRADCAM İlişkisi:")
        print("- Yüksek doğruluk ")
        print("- Düşük doğruluk ")
        
        print("\n6. :")
        print("- En iyi GRADCAM : SoftPlus + ADAM")
        print("- En kötü GRADCAM : Identity + SGD")
        print("- GRADCAM kalitesi korelasyon")
    
    def tam_analizi_calistir(self, epochs=30):
        
        print("CNN Analizi Başlatılıyor")
        print("="*100)
        
        # veriyi yükle
        self.veri_yukle_ve_onisle()
        
        # eğitim
        self.tum_modelleri_egit(epochs)
        
        # sonuclar tablosu
        sonuclar_tablosu = self.kapsamli_sonuclar_tablosu_olustur()
        
        # eğitim geçmişi
        self.egitim_gecmisi_ciz()
        
        # karışıklık matrisleri 
        self.karisiklik_matrisleri_ciz()
        
        # Sınıflandırma raporu
        self.siniflandirma_raporlari_uret()
        
        # GRADCAM karşılaştırması
        self.gradcam_karsilastirma_olustur(num_images=3)
        
        # GRADCAM analizi
        self.gradcam_analizi_yap()
        
        print("\n" + "="*100)
        print("KAPSAMLI ANALİZ TAMAMLANDI!")
        print("="*100)
        print("Oluşturulan dosyalar:")
        print("- kapsamli_aktivasyon_sonuclari.csv")
        print("- kapsamli_egitim_gecmisi.png")
        print("- kapsamli_karisiklik_matrisleri.png")
        print("- kapsamli_gradcam_karsilastirma.png")
        
        return sonuclar_tablosu

def ana_fonksiyon():
    
    
    analiz = KapsamliAktivasyonAnalizi(veri_yolu=".")
    
    
    sonuclar_tablosu = analiz.tam_analizi_calistir(epochs=20)
    
    return analiz, sonuclar_tablosu

if __name__ == "__main__":
    
    analiz, sonuclar_tablosu = ana_fonksiyon()
