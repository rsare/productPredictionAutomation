import numpy as np
import pandas as pd


def _ensure_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # id alanlarını string yaparsak kategorik yönetimi kolaylaşır
    for col in [
        "OrderId",
        "CustomerId",
        "ProductId",
        "MainCategoryId",
        "SubCategoryId",
        "ManufacturerId",
        "BrandId",
        "Sku",
        "Barcode",
    ]:
        if col in df.columns:
            df[col] = df[col].astype(str)
    # CreatedOnUtc
    if "CreatedOnUtc" in df.columns:
        df["CreatedOnUtc"] = pd.to_datetime(df["CreatedOnUtc"], errors="coerce", utc=True)
    return df


def build_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_types(df)
    # toplam adet ve farklı kategori sayısı
    cust = (
        df.groupby("CustomerId")
        .agg(
            total_products_bought=("ProductId", "count"),
            unique_categories_bought=("MainCategoryId", pd.Series.nunique),
        )
        .reset_index()
    )
    # son satın alınan ana kategori
    if "CreatedOnUtc" in df.columns:
        last_rows = df.sort_values("CreatedOnUtc").groupby("CustomerId").tail(1)
    else:
        last_rows = df.groupby("CustomerId").tail(1)
    last_cat = last_rows[["CustomerId", "MainCategoryId"]].rename(
        columns={"MainCategoryId": "last_bought_category"}
    )
    cust = cust.merge(last_cat, on="CustomerId", how="left")
    return cust


def build_product_features(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_types(df)
    # ürün sabit özellikleri + bazı istatistikler
    prod = (
        df.groupby("ProductId")
        .agg(
            product_category=("MainCategoryId", "first"),
            product_subcategory=("SubCategoryId", "first"),
            total_sold=("SoldCount", "sum"),
            average_rating=("TotalRating", "mean"),
            prod_global_freq=("ProductId", "count"),
            Sku=("Sku", "first"),
            BrandId=("BrandId", "first"),
        )
        .reset_index()
    )
    return prod


def create_final_dataset(
    df: pd.DataFrame,
    customer_features: pd.DataFrame,
    product_features: pd.DataFrame,
    neg_per_pos: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Pozitif: (CustomerId, ProductId) satın alınanlar (Target=1)
    Negatif: aynı müşterinin satın aldığı ana kategorilerden, satın almadığı ürünler (Target=0)
    """
    rng = np.random.default_rng(seed)
    df = _ensure_types(df)

    # Pozitifler
    pos = df[["CustomerId", "ProductId", "MainCategoryId", "CreatedOnUtc"]].drop_duplicates()
    pos = pos.rename(columns={"MainCategoryId": "product_category"}).copy()
    pos["Target"] = 1

    # Müşteri -> aldığı ana kategoriler
    cust_cats = (
        df.groupby("CustomerId")["MainCategoryId"].apply(lambda x: list(pd.Series(x).astype(str).unique()))
    )

    # Ana kategori -> ürün listesi
    prod_by_cat = product_features.groupby("product_category")["ProductId"].apply(list).to_dict()

    # Negatif örnekler
    neg_rows = []
    for cust, cat_list in cust_cats.items():
        # müşterinin aldığı ürünler
        bought = set(df.loc[df["CustomerId"] == cust, "ProductId"].astype(str))
        # her aldığı kategori için o kategorideki diğer ürünlerden örnekle
        candidates = []
        for cat in cat_list:
            candidates.extend(prod_by_cat.get(cat, []))
        candidates = list(set(map(str, candidates)) - bought)
        if not candidates:
            continue
        n_pos = (df["CustomerId"] == cust).sum()
        n_neg = min(n_pos * neg_per_pos, len(candidates))
        sampled = rng.choice(candidates, size=n_neg, replace=False)
        for pid in sampled:
            neg_rows.append((str(cust), str(pid), 0))

    neg = pd.DataFrame(neg_rows, columns=["CustomerId", "ProductId", "Target"])
    # ürün kategorisini ekle
    neg = neg.merge(product_features[["ProductId", "product_category"]], on="ProductId", how="left")

    # Birleştir
    final = pd.concat(
        [
            pos[["CustomerId", "ProductId", "product_category", "Target"]],
            neg[["CustomerId", "ProductId", "product_category", "Target"]],
        ],
        ignore_index=True,
    )

    # Müşteri & Ürün özelliklerini ekle
    final = final.merge(customer_features, on="CustomerId", how="left")
    final = final.merge(product_features, on="ProductId", how="left")

    # Kategorik dtype'lar
    for c in ["CustomerId", "ProductId", "product_category", "product_subcategory", "last_bought_category"]:
        if c in final.columns:
            final[c] = final[c].astype("category")

    # Sayısal boşlukları doldur
    for c in ["total_sold", "average_rating", "prod_global_freq", "total_products_bought", "unique_categories_bought"]:
        if c in final.columns:
            final[c] = pd.to_numeric(final[c], errors="coerce").fillna(0)

    return final
