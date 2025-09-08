import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from src.features.build_features import _norm_str_series
from src.models.predict_model import generate_cross_category_recommendations


def generate_cross_category_map_from_data(df: pd.DataFrame, min_support: float = 0.01, min_threshold: float = 0.7) -> dict:
    """
    Veri setinden geçmiş siparişlere göre çapraz kategori eşleşmeleri çıkarır.
    """
    print("Çapraz kategori eşleşmeleri çıkarılıyor...")

    # Veriyi işle: Sipariş bazında ana kategorileri grupla
    df_processed = df.copy()
    # Gerekli sütunları normalize et
    for col in ["CustomerId", "ProductId", "MainCategoryId", "OrderId"]: # OrderId veya benzeri bir sipariş ID'si olmalı
        if col in df_processed.columns:
            df_processed[col] = _norm_str_series(df_processed[col])

    # Sadece gerekli sütunlarla çalışalım
    if "OrderId" not in df_processed.columns or "MainCategoryId" not in df_processed.columns:
        print("Hata: 'OrderId' ve 'MainCategoryId' sütunları gereklidir.")
        return {}

    # Her siparişteki ana kategorileri belirle
    # Bir sepet (basket) bir sipariş ID'sine karşılık gelir.
    # İçeriği ise o siparişteki benzersiz ana kategorilerdir.
    transactions = df_processed.groupby("OrderId")["MainCategoryId"].apply(lambda x: list(set(x)))

    # Apriori için uygun formata getir: One-hot encoding
    # Her satır bir sipariş, her sütun bir kategori. Değer 1 ise o kategoriden var, 0 ise yok.
    all_categories = df_processed["MainCategoryId"].unique()
    basket_df = pd.DataFrame(index=transactions.index)

    for category in all_categories:
        basket_df[category] = transactions.apply(lambda x: category in x).astype(int)

    # Apriori algoritmasını çalıştır
    frequent_itemsets = apriori(basket_df, min_support=min_support, use_colnames=True)

    # Association Rules çıkar
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_threshold)

    # Anlamlı kuralları al ve mapping'i oluştur
    # Örnek: {antecedent_category: consequent_category}
    cross_mapping = {}
    for _, row in rules.iterrows():
        # Kural: antecedent -> consequent
        antecedents = list(row['antecedents'])
        consequents = list(row['consequents'])

        # Bizim amacımız tek bir kategoriden başka bir kategoriye geçişi bulmak
        if len(antecedents) == 1 and len(consequents) == 1:
            antecedent_cat = antecedents[0]
            consequent_cat = consequents[0]
            # Eğer bu eşleşme zaten yoksa veya daha düşük confiance ile varsa ekle
            if antecedent_cat not in cross_mapping or cross_mapping[antecedent_cat] != consequent_cat:
                 # Belki lift değerine göre de filtreleme yapabilirsiniz, daha güçlü ilişkileri bulmak için
                 if row.get('lift', 0) > 1.0: # Lift > 1.0, rastgelelikten daha iyi bir ilişki olduğunu gösterir
                    cross_mapping[antecedent_cat] = consequent_cat

    print(f"Bulunan çapraz kategori eşleşmesi sayısı: {len(cross_mapping)}")
    return cross_mapping


# main.py dosyasında bu fonksiyonu çağıracaksınız:

# src/features/analyze_features import et
from src.features.analyze_features import generate_cross_category_map_from_data

# ... diğer kodlar ...

if __name__ == "__main__":
    # ... veri yükleme, özellik oluşturma ...

    # Çapraz kategori haritasını otomatik oluştur
    print("Kategori haritası otomatik oluşturuluyor...")
    # Min_support ve min_threshold değerlerini veri setinize göre ayarlayın
    cross_category_mapping = generate_cross_category_map_from_data(
        df, min_support=0.01, min_threshold=0.7
    )

    # Eğer harita boşsa veya çok az eşleşme varsa, elle oluşturulan bir yedek harita kullanabilirsiniz
    if not cross_category_mapping or len(cross_category_mapping) < 5: # Örnek: en az 5 eşleşme bekleyelim
        print("Otomatik harita yetersiz, elle tanımlanmış örnek harita kullanılıyor.")
        cross_category_mapping = {
            '10001': '10002',
            '10003': '10005',
            # ...
        }

    # ... sonra bu haritayı generate_cross_category_recommendations fonksiyonuna gönderin
    rec_cross = generate_cross_category_recommendations(
        lgb_model, X_train, df, customer_features, product_features, cross_category_mapping, top_k=10
    )
    # ...