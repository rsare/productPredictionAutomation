
import numpy as np
import pandas as pd

# ===================== Yardımcılar =====================

def _norm_str_series(s: pd.Series) -> pd.Series:
    """ID ve kategori alanlarını ortak formata çek: string, trim, '.0' kuyruğunu at."""
    return (
        s.astype(str)
         .str.strip()
         .str.replace(r"\.0$", "", regex=True)
    )

def _last_bought_category_map(df: pd.DataFrame) -> pd.Series:
    """
    müşteri -> son satın alınan ana kategori (normalize edilmiş)
    """
    df = df.copy()
    # normalize ID/kategori
    for c in ["CustomerId", "ProductId", "MainCategoryId"]:
        if c in df.columns:
            df[c] = _norm_str_series(df[c])

    # zaman
    if "CreatedOnUtc" in df.columns:
        df["CreatedOnUtc"] = pd.to_datetime(df["CreatedOnUtc"], errors="coerce", utc=True)
        tmp = df.sort_values("CreatedOnUtc")
    else:
        tmp = df

    last_rows = tmp.groupby("CustomerId").tail(1)
    out = last_rows.set_index("CustomerId")["MainCategoryId"]
    return _norm_str_series(out)

def _align_to_train(X_train: pd.DataFrame, df_pred: pd.DataFrame) -> pd.DataFrame:
    """
    df_pred'i eğitimdeki kolon+dtype'lara (özellikle categorical categories) hizalar.
    """
    df_pred = df_pred.copy()

    # Eğitimde kategorik kolonlar ve kategori listeleri
    cat_cols = [c for c in X_train.columns if str(X_train[c].dtype) == "category"]
    train_cats = {c: X_train[c].cat.categories for c in cat_cols}

    # Eksik kolonları tamamla
    for c in X_train.columns:
        if c not in df_pred.columns:
            if c in cat_cols:
                df_pred[c] = pd.Categorical([np.nan] * len(df_pred), categories=train_cats[c])
            else:
                df_pred[c] = 0

    # Var olan kategorik kolonları eğitim kategorileriyle zorla
    for c in cat_cols:
        df_pred[c] = pd.Categorical(df_pred[c], categories=train_cats[c])

    # Sayısal kolonları güvenli çevir
    for c in df_pred.columns:
        if c not in cat_cols:
            df_pred[c] = pd.to_numeric(df_pred[c], errors="coerce").fillna(0)

    # Eğitimle aynı sıra
    return df_pred.reindex(columns=X_train.columns)

def _global_popular_products(product_features: pd.DataFrame, top_n: int = 100) -> list[str]:
    col = "prod_global_freq" if "prod_global_freq" in product_features.columns else None
    pf = product_features.copy()
    pf["ProductId"] = _norm_str_series(pf["ProductId"])
    if col:
        pf = pf.sort_values(col, ascending=False)
    return pf["ProductId"].dropna().astype(str).head(top_n).tolist()

# ===================== Öneri Fonksiyonları =====================

def generate_same_category_recommendations(
        lgb_model,
        X_train: pd.DataFrame,
        df: pd.DataFrame,
        customer_features: pd.DataFrame,
        product_features: pd.DataFrame,
        top_k: int = 10,
):
    """
    Her müşteri için SON satın aldığı ana kategorideki ürünlerden öneri.
    Kategori/ID eşleşmelerini normalize eder; aday yoksa popüler fallback kullanır.
    """
    # ---- Normalize alanlar ----
    df = df.copy()
    for c in ["CustomerId", "ProductId", "MainCategoryId", "SubCategoryId", "BrandId", "Sku"]:
        if c in df.columns:
            df[c] = _norm_str_series(df[c])

    customer_features = customer_features.copy()
    # 'last_bought_category' sadece customer_features'da varsa normalize edilsin.
    if "last_bought_category" in customer_features.columns:
        customer_features["last_bought_category"] = _norm_str_series(customer_features["last_bought_category"])

    product_features = product_features.copy()
    for c in ["ProductId", "product_category", "product_subcategory", "BrandId", "Sku"]:
        if c in product_features.columns:
            product_features[c] = _norm_str_series(product_features[c])

    # müşteri -> son kategori
    last_cat_map = _last_bought_category_map(df)

    # kategori -> ürün listesi (normalize edilmiş)
    prod_by_cat = (
        product_features.groupby("product_category")["ProductId"]
        .apply(lambda x: _norm_str_series(x).tolist())
        .to_dict()
    )

    # global popüler fallback havuzu
    global_top = _global_popular_products(product_features, top_n=200)

    all_customer_ids = _norm_str_series(df["CustomerId"]).unique()
    print(f"Toplam {len(all_customer_ids)} müşteri için aynı kategori önerileri hazırlanıyor...")

    all_recs = []
    batch_size = 1000
    batches = [all_customer_ids[i:i + batch_size] for i in range(0, len(all_customer_ids), batch_size)]

    # Müşteri-ürün geçmişi (normalize) – alınanları çıkaracağız
    bought = (
        df[["CustomerId", "ProductId"]]
        .drop_duplicates()
        .assign(
            CustomerId=lambda d: _norm_str_series(d["CustomerId"]),
            ProductId=lambda d: _norm_str_series(d["ProductId"]),
        )
    )

    for i, batch in enumerate(batches, 1):
        print(f"[{i}/{len(batches)}] batch işleniyor...")

        cand_frames = []
        for cust in batch:
            last_cat = last_cat_map.get(cust)
            # ana kategori yoksa / eşleşmezse -> fallback havuzu
            candidate_pids = prod_by_cat.get(last_cat, [])
            if not candidate_pids:
                candidate_pids = global_top

            if candidate_pids:
                cand_frames.append(pd.DataFrame({"CustomerId": cust, "ProductId": candidate_pids}))

        if not cand_frames:
            continue

        candidates = pd.concat(cand_frames, ignore_index=True)

        # daha önce alınanları çıkar
        candidates = candidates.merge(bought, on=["CustomerId", "ProductId"], how="left", indicator=True)
        candidates = candidates[candidates["_merge"] == "left_only"].drop(columns="_merge")
        if candidates.empty:
            continue

        # müşteri/ürün özellikleri
        candidates = candidates.merge(customer_features, on="CustomerId", how="left")
        candidates = candidates.merge(product_features, on="ProductId", how="left")

        # Son satın alınan kategoriyi (last_bought_category) ekle (customer_features'dan gelmiyorsa)
        if "last_bought_category" not in candidates.columns:
            candidates["last_bought_category"] = candidates["CustomerId"].map(last_cat_map)
        elif "last_bought_category" in customer_features.columns:
            # Eğer customer_features'da varsa ve NaN geliyorsa, map ile doldurmayı dene
            candidates["last_bought_category"] = candidates["last_bought_category"].fillna(
                candidates["CustomerId"].map(last_cat_map))

        # eğitimle hizala + tahmin
        X_pred = _align_to_train(X_train, candidates)
        scores = lgb_model.predict_proba(X_pred)[:, 1]
        candidates["probability"] = scores

        topk = (
            candidates.sort_values(["CustomerId", "probability"], ascending=[True, False])
            .groupby("CustomerId")
            .head(top_k)
            .reset_index(drop=True)
        )
        if not topk.empty:
            all_recs.append(topk)

    if not all_recs:
        return pd.DataFrame()

    final = pd.concat(all_recs, ignore_index=True)
    print("Aynı kategori için öneri listeleri oluşturuldu.")
    return final


def generate_cross_category_recommendations(
        lgb_model,
        X_train: pd.DataFrame,
        df: pd.DataFrame,
        customer_features: pd.DataFrame,
        product_features: pd.DataFrame,
        cross_category_mapping: dict,  # Bu sözlüğün doğru doldurulmuş olması KRİTİK
        top_k: int = 10,
):
    """
    Her müşteri için son aldığı ana kategoriden cross_category_mapping'de belirtilen hedef kategoriye öneri.
    Tip/format normalize edilir; aday yoksa popüler fallback.
    """
    # normalize
    df = df.copy()
    for c in ["CustomerId", "ProductId", "MainCategoryId"]:
        if c in df.columns:
            df[c] = _norm_str_series(df[c])

    customer_features = customer_features.copy()
    # 'last_bought_category' sadece customer_features'da varsa normalize edilsin.
    if "last_bought_category" in customer_features.columns:
        customer_features["last_bought_category"] = _norm_str_series(
            customer_features["last_bought_category"])  # _norm_str_series olarak düzeltildi

    product_features = product_features.copy()
    for c in ["ProductId", "product_category", "product_subcategory", "BrandId", "Sku"]:
        if c in product_features.columns:
            product_features[c] = _norm_str_series(product_features[c])  # _norm_str_series olarak düzeltildi

    # mapping'i stringe çevir (örn 10001 -> "10001")
    # Sadece string değerler için normalizasyon yapalım.
    mapping_norm = {}
    for k, v in cross_category_mapping.items():
        mapping_norm[str(_norm_str_series(pd.Series([k])).iloc[0])] = str(_norm_str_series(pd.Series([v])).iloc[0])

    last_cat_map = _last_bought_category_map(df)
    prod_by_cat = (
        product_features.groupby("product_category")["ProductId"]
        .apply(lambda x: _norm_str_series(x).tolist())
        .to_dict()
    )
    global_top = _global_popular_products(product_features, top_n=200)

    all_customer_ids = _norm_str_series (df["CustomerId"]).unique()  # _norm_str_series olarak düzeltildi
    print(f"Toplam {len(all_customer_ids)} müşteri için farklı kategori önerileri hazırlanıyor...")

    bought = (
        df[["CustomerId", "ProductId"]]
        .drop_duplicates()
        .assign(
            CustomerId=lambda d: _norm_str_series(d["CustomerId"]),  # _norm_str_series olarak düzeltildi
            ProductId=lambda d: _norm_str_series(d["ProductId"]),  # _norm_str_series olarak düzeltildi
        )
    )

    all_recs = []
    batch_size = 1000
    batches = [all_customer_ids[i:i + batch_size] for i in range(0, len(all_customer_ids), batch_size)]

    for i, batch in enumerate(batches, 1):
        print(f"[{i}/{len(batches)}] batch işleniyor...")

        rows = []
        for cust in batch:
            last_cat = last_cat_map.get(cust)
            target_cat = mapping_norm.get(last_cat)  # Burası artık doğru eşleşmeyi bulacak

            # Eşleşen hedef kategori var mı diye kontrol et
            if target_cat:
                pids = prod_by_cat.get(target_cat, [])  # Hedef kategori altındaki ürünleri al
            else:
                pids = []  # Hedef kategori yoksa, boş liste ile başla

            # Eğer hedef kategori altındaki ürünler boşsa VEYA eşleşen hedef kategori hiç yoksa fallback kullan
            if not pids:
                pids = global_top

            if pids:
                # last_bought_category_x sütunu için orijinal last_cat'i ekle
                rows.append(pd.DataFrame({
                    "CustomerId": cust,
                    "ProductId": pids,
                    "last_bought_category_x": last_cat  # 'x' soneki, olası karışıklıkları önler
                }))

        if not rows:
            continue

        candidates = pd.concat(rows, ignore_index=True)

        # Daha önce alınan ürünleri çıkar
        candidates = candidates.merge(bought, on=["CustomerId", "ProductId"], how="left", indicator=True)
        candidates = candidates[candidates["_merge"] == "left_only"].drop(columns="_merge")
        if candidates.empty:
            continue

        # Müşteri ve ürün özelliklerini ekle
        candidates = candidates.merge(customer_features, on="CustomerId", how="left")
        candidates = candidates.merge(product_features, on="ProductId", how="left")

        # last_bought_category sütunu (customer_features'dan gelmeyen veya NaN olanlar için) doldur
        if "last_bought_category" not in candidates.columns:
            candidates["last_bought_category"] = candidates["CustomerId"].map(last_cat_map)
        elif "last_bought_category" in customer_features.columns:
            # Eğer customer_features'da varsa ve NaN geliyorsa, map ile doldurmayı dene
            candidates["last_bought_category"] = candidates["last_bought_category"].fillna(
                candidates["CustomerId"].map(last_cat_map))

        # _align_to_train'in doğru çalışması için bazı kolon isimlerini ayarlayalım
        # Eğer _align_to_train X_train'deki kolonları bekliyorsa, aday kolonlarını ona göre ayarla
        # Örneğin, product_features'dan gelen 'product_category' X_train'de 'last_bought_category' olarak geçiyorsa vb.
        # Bu kısım, X_train'in nasıl oluşturulduğuna bağlıdır.
        # Aşağıdaki örnek, 'product_category'nin X_train'deki 'last_bought_category' ile aynı olmasını varsayar.
        if 'product_category' in candidates.columns and 'last_bought_category' in X_train.columns:
            if candidates['product_category'].dtype == 'object' and X_train[
                'last_bought_category'].dtype == 'object':  # veya kategorik ise
                candidates['last_bought_category'] = candidates[
                    'product_category']  # 'product_category'yi 'last_bought_category' olarak yeniden adlandır/kullan

        # Tahmin için veriyi eğitilmiş modele hizala
        X_pred = _align_to_train(X_train, candidates)
        scores = lgb_model.predict_proba(X_pred)[:, 1]
        candidates["probability"] = scores

        topk = (
            candidates.sort_values(["CustomerId", "probability"], ascending=[True, False])
            .groupby("CustomerId")
            .head(top_k)
            .reset_index(drop=True)
        )
        if not topk.empty:
            all_recs.append(topk)

    if not all_recs:
        print("Uygun müşteri/kategori bulunamadı.")
        return pd.DataFrame()

    final = pd.concat(all_recs, ignore_index=True)
    print("Farklı kategori için öneri listeleri oluşturuldu.")
    return final