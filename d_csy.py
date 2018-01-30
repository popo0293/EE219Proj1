from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=50, n_iter=10,random_state=42)
X_train_tfidf_df_5_LSI = svd.fit_transform(X_5_train_tfidf)
svd = TruncatedSVD(n_components=50, n_iter=10,random_state=42)
X_train_tfidf_df_2_LSI = svd.fit_transform(X_2_train_tfidf)

from sklearn.decomposition import NMF
nmf = NMF(n_components=50)
X_train_tfidf_df_5_NMF = nmf.fit_transform(X_5_train_tfidf)
nmf = NMF(n_components=50)
X_train_tfidf_df_2_NMF = nmf.fit_transform(X_2_train_tfidf)