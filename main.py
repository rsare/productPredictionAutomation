import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
from itertools import combinations

from src.features.build_features import (
    build_customer_features,
    build_product_features,
    create_final_dataset,
)
from src.models.train_model import train_lgbm_model
from src.models.predict_model import (
    generate_same_category_recommendations,
    generate_cross_category_recommendations,
)


# --- Yardımcı Fonksiyonlar ---

def _norm_str_series(s: pd.Series) -> pd.Series:
    """
    ID ve kategori alanlarını ortak formata çek: string, trim, '.0' kuyruğunu at.
    """
    return (
        s.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def create_dynamic_cross_category_mapping(df: pd.DataFrame, min_pairs: int = 5) -> dict:
    """
    Veri setinden müşterilerin birlikte en çok satın aldığı kategorileri analiz ederek
    dinamik bir çapraz kategori haritası (mapping) oluşturur.
    Her 'son_alınan_kategori' için en çok birlikte alınan 'hedef_kategori'yi belirler.

    Args:
        df (pd.DataFrame): Sipariş verilerini içeren DataFrame.
        min_pairs (int): Bir eşleşmenin haritaya dahil edilmesi için minimum satın alma çifti sayısı.

    Returns:
        dict: Dinamik olarak oluşturulmuş cross-category mapping sözlüğü.
    """
    print("Dinamik çapraz kategori eşleştirmesi oluşturuluyor...")

    # Müşteri ve ana kategori verilerini hazırlama
    df_norm = df[['CustomerId', 'MainCategoryId']].copy()
    for col in ['CustomerId', 'MainCategoryId']:
        df_norm[col] = _norm_str_series(df_norm[col])

    # Müşteri bazlı benzersiz kategori listeleri
    customer_categories = df_norm.groupby('CustomerId')['MainCategoryId'].unique()

    # Birlikte alınan kategori çiftlerini sayma
    pair_counts = {}
    for categories in customer_categories:
        # 1'den fazla kategori alan müşterileri ele al
        if len(categories) > 1:
            # Tüm kategori çiftlerini oluştur
            for c1, c2 in combinations(sorted(categories), 2):
                pair = tuple(sorted((c1, c2)))
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

    if not pair_counts:
        print("Birlikte alınan kategori çifti bulunamadı. Boş bir harita döndürülüyor.")
        return {}

    # Pandas DataFrame'e çevirerek daha kolay analiz yapma
    pairs_df = pd.DataFrame(pair_counts.items(), columns=['category_pair', 'count'])
    pairs_df[['cat1', 'cat2']] = pd.DataFrame(pairs_df['category_pair'].tolist(), index=pairs_df.index)

    # Her kategori için en çok birlikte alınan hedef kategoriyi bulma
    # Burada varsayım: X kategorisi alan müşteri en çok Y kategorisi de almıştır.
    mapping = {}
    all_categories = pd.concat([pairs_df['cat1'], pairs_df['cat2']]).unique()

    for cat in all_categories:
        # Kaynak kategori olarak 'cat' olan tüm çiftleri filtrele
        as_source = pairs_df[pairs_df['cat1'] == cat]
        # Hedef kategoriyi 'cat2' olarak seç ve en yüksek sayıyı bul
        if not as_source.empty:
            target_cat = as_source.loc[as_source['count'].idxmax()]['cat2']
            if as_source['count'].max() >= min_pairs:
                mapping[cat] = target_cat

        # Kaynak kategori olarak 'cat2' olan tüm çiftleri filtrele
        as_target = pairs_df[pairs_df['cat2'] == cat]
        # Hedef kategoriyi 'cat1' olarak seç ve en yüksek sayıyı bul
        if not as_target.empty:
            source_cat = as_target.loc[as_target['count'].idxmax()]['cat1']
            if as_target['count'].max() >= min_pairs and source_cat not in mapping:
                mapping[source_cat] = cat

    print(f"Oluşturulan eşleştirme haritası ({len(mapping)} adet): {mapping}")
    return mapping


def visualize_recommendations(recommendations_df: pd.DataFrame, title: str, output_prefix: str):
    """
    Öneri DataFrame'ini görselleştiren ve kaydeten fonksiyon.
    """
    if recommendations_df.empty:
        print(f"'{title}' için görselleştirilecek veri yok.")
        return

    # 1. En Çok Önerilen Ürünlerin (En Çok Olasılıkla İlk 20)
    plt.figure(figsize=(12, 7))
    top_products = recommendations_df['ProductId'].value_counts().nlargest(20).index
    sns.countplot(data=recommendations_df[recommendations_df['ProductId'].isin(top_products)],
                  y="ProductId",
                  order=top_products,
                  palette="viridis")
    plt.title(f'{title} - En Çok Önerilen İlk 20 Ürün', fontsize=16)
    plt.xlabel('Öneri Sayısı', fontsize=12)
    plt.ylabel('Ürün ID', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_top_products.png')
    print(f"'{output_prefix}_top_products.png' kaydedildi.")
    # plt.show()

    # 2. Müşteri Başına Düşen Ortalama Öneri Sayısı Dağılımı
    plt.figure(figsize=(10, 6))
    recommendations_df['CustomerId'] = recommendations_df['CustomerId'].astype(str)
    avg_recs_per_customer = recommendations_df.groupby('CustomerId').size().reset_index(name='num_recommendations')
    sns.histplot(data=avg_recs_per_customer, x='num_recommendations', bins=30, kde=True, color='skyblue')
    plt.title(f'{title} - Müşteri Başına Öneri Sayısı Dağılımı', fontsize=16)
    plt.xlabel('Öneri Sayısı', fontsize=12)
    plt.ylabel('Müşteri Sayısı', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_recs_per_customer.png')
    print(f"'{output_prefix}_recs_per_customer.png' kaydedildi.")
    # plt.show()

    # 3. Öneri Güvenilirliklerinin (Probability) Dağılımı
    plt.figure(figsize=(12, 6))
    sns.histplot(recommendations_df['probability'], kde=True, color='salmon', label='Öneri Olasılıkları')
    plt.title(f'{title} - Öneri Güvenilirliklerinin Dağılımı', fontsize=16)
    plt.xlabel('Güvenilirlik (Probability)')
    plt.ylabel('Frekans')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_probability_distribution.png')
    print(f"'{output_prefix}_probability_distribution.png' kaydedildi.")
    # plt.show()

    print(f"'{title}' için görselleştirmeler tamamlandı.")


# --- Ana Çalıştırma Bloğu ---
if __name__ == "__main__":
    print("Veri setleri yükleniyor...")
    try:
        df = pd.read_csv("data/saresiparis50.csv", sep=";", encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        try:
            df = pd.read_csv("data/saresiparis50.csv", sep=";", encoding="ISO-8859-9", low_memory=False)
        except FileNotFoundError:
            print("HATA: 'data/saresiparis50.csv' dosyası bulunamadı. Lütfen dosya yolunu kontrol edin.")
            sys.exit(1)
        except Exception as e:
            print(f"HATA: CSV dosyası okunurken beklenmedik bir hata oluştu: {e}")
            sys.exit(1)
    except Exception as e:
        print(f"HATA: CSV dosyası okunurken beklenmedik bir hata oluştu: {e}")
        sys.exit(1)

    if "CreatedOnUtc" in df.columns:
        print("CreatedOnUtc alanı işleniyor...")
        df["CreatedOnUtc"] = pd.to_datetime(df["CreatedOnUtc"], errors="coerce", utc=True)
    else:
        print("Uyarı: 'CreatedOnUtc' sütunu bulunamadı. Tarih bazlı sıralama yapılamayabilir.")

    print("Özellikler oluşturuluyor...")
    try:
        customer_features = build_customer_features(df)
        product_features = build_product_features(df)
        df_final_training_data = create_final_dataset(df, customer_features, product_features, neg_per_pos=3, seed=42)

        if df_final_training_data.empty:
            print("HATA: Eğitim veri seti oluşturulamadı.")
            sys.exit(1)

        print("Model eğitiliyor...")
        lgb_model, X_train = train_lgbm_model(df_final_training_data)

        if lgb_model is None or X_train is None:
            print("HATA: Model eğitilemedi.")
            sys.exit(1)

    except Exception as e:
        print(f"HATA: Özellik oluşturma veya model eğitimi sırasında bir hata oluştu: {e}")
        sys.exit(1)

    # -----------------------------------------------------------
    # SENARYO 1: AYNI KATEGORİDEN ÜRÜN ÖNERİSİ
    # -----------------------------------------------------------
    print("\n--- Aynı Kategoriden Ürün Önerileri Oluşturuluyor ---")
    rec_same_category = generate_same_category_recommendations(
        lgb_model, X_train, df, customer_features, product_features, top_k=10
    )
    print("Aynı kategori öneri listesi oluşturuldu.")
    if not rec_same_category.empty:
        print("Örnek 'Aynı Kategori' Önerileri:")
        print(rec_same_category.head(15))
    else:
        print("Aynı kategori için öneri bulunamadı.")

    # -----------------------------------------------------------
    # SENARYO 2: FARKLI KATEGORİDEN ÜRÜN ÖNERİSİ
    # -----------------------------------------------------------
    print("\n--- Farklı Kategoriden Ürün Önerileri Oluşturuluyor ---")
    # Dinamik olarak cross_category_mapping oluşturuluyor
    cross_category_mapping = create_dynamic_cross_category_mapping(df)

    rec_cross_category = generate_cross_category_recommendations(
        lgb_model, X_train, df, customer_features, product_features, cross_category_mapping, top_k=10
    )
    print("Farklı kategori öneri listesi oluşturuldu.")
    if not rec_cross_category.empty:
        print("Örnek 'Farklı Kategori' Önerileri:")
        print(rec_cross_category.head(15))
    else:
        print("Farklı kategori için öneri bulunamadı.")

    # --- Görselleştirme Çağrıları ---
    print("\n--- Görselleştirmeler Oluşturuluyor ---")
    if not rec_same_category.empty:
        visualize_recommendations(rec_same_category, "Aynı Kategori Önerileri", "same_category_recommendations")
    else:
        print("Aynı kategori önerileri için görselleştirme atlandı (veri yok).")

    if not rec_cross_category.empty:
        visualize_recommendations(rec_cross_category, "Farklı Kategori Önerileri", "cross_category_recommendations")
    else:
        print("Farklı kategori önerileri için görselleştirme atlandı (veri yok).")

    # --- Terminalden Kullanıcı Girdisi Alma ve Öneri Sunma ---
    print("\n--- İnteraktif Öneri Sistemi ---")
    print("Bu modül, belirli bir müşteri ID'si için öneri almanızı sağlar.")

    while True:
        user_input = input("Öneri almak istediğiniz müşteri ID'sini girin (veya çıkmak için 'q' yazın): ").strip()
        if user_input.lower() == 'q':
            print("Uygulama kapatılıyor. İyi günler!")
            break
        if not user_input.isdigit():
            print("Geçersiz giriş. Lütfen yalnızca sayısal bir müşteri ID girin.")
            continue

        customer_id_to_recommend = str(user_input)
        print(f"\n--- Müşteri {customer_id_to_recommend} İçin Öneriler ---")

        # Önerilerin CustomerId sütunlarını stringe çevirerek karşılaştırma yapalım
        if 'CustomerId' in rec_same_category.columns:
            customer_recs_same = rec_same_category[
                rec_same_category['CustomerId'].astype(str) == customer_id_to_recommend]
            if not customer_recs_same.empty:
                print(f"\n{customer_id_to_recommend} için Aynı Kategori Önerileri:")
                print(customer_recs_same.head(5)[['ProductId', 'probability']])
            else:
                print(f"{customer_id_to_recommend} için aynı kategori önerisi bulunamadı.")
        else:
            print("Aynı kategori öneri DataFrame'inde 'CustomerId' sütunu yok.")

        if 'CustomerId' in rec_cross_category.columns:
            customer_recs_cross = rec_cross_category[
                rec_cross_category['CustomerId'].astype(str) == customer_id_to_recommend]
            if not customer_recs_cross.empty:
                print(f"\n{customer_id_to_recommend} için Farklı Kategori Önerileri:")
                print(customer_recs_cross.head(5)[['ProductId', 'probability']])
            else:
                print(f"{customer_id_to_recommend} için farklı kategori önerisi bulunamadı.")
        else:
            print("Farklı kategori öneri DataFrame'inde 'CustomerId' sütunu yok.")