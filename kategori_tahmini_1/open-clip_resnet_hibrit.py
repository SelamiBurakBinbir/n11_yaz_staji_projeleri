import os
import json
import math
import pickle
import warnings
from io import BytesIO
from datetime import datetime
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import requests
from PIL import Image

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import open_clip

from tqdm import tqdm
warnings.filterwarnings("ignore")


# =========================
# Yardımcılar
# =========================
def normalize_torch(x, dim=-1, eps=1e-12):
    return x / (x.norm(dim=dim, keepdim=True) + eps)

def to_half_on_device(x, device):
    return x.half().to(device) if device == "cuda" else x.to(device)

def cosine_sim_matrix(a, b):
    # cihaz hizalaması
    if a.device != b.device:
        b = b.to(a.device)
    # dtype hizalaması (her zaman float32'ye yükselt)
    return a.float() @ b.float().T

# =========================
# Ana Sınıf
# =========================
class HierarchicalMaxAccPredictor:
    """
    Amaç: Her kategori için doğruluğu maksimize etmek.
    Yöntem: kategori-bazlı α_c seçimi (CLIP vs Görsel), k-NN ve centroid birleştirmesi.
    """
    def __init__(self, csv_path=r"C:\Users\selam\Desktop\kod deneme\gd\veriler\veri_en.csv"):  # CHANGED
        # Veri / cihaz
        self.csv_path = csv_path
        self.df = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Modeller
        self.openclip_model = None
        self.openclip_tokenizer = None
        self.openclip_preprocess = None

        self.resnet50_model = None
        self.resnet50_transform = None

        # Hiyerarşi ve eşleştirmeler
        self.category_hierarchy = {}   # {level: [cats]}
        self.parent_child_map = {}     # {level: {parent:[children,...]}}

        # Text embedding cache (kategori başına)
        self.level_text_embeds = {}    # {level: torch (C,D) normalized, device}
        self.level_categories = {}     # {level: [cat1,...]}
        self.level_cat2idx = {}        # {level: {cat:i}}

        # Görsel embedding’ler (tüm örnekler)
        self.img_ids = []              # dataset indexleri listesi (feature üretilenler)
        self.img_clip = None           # torch (N,D) normalized, device
        self.img_resnet = None         # torch (N,2048) normalized, device

        # Label matrisleri (her level için)
        self.labels = {}               # {level: np.array shape=(N,)  (kategori indexi veya -1)}

        # Eğitim/Validasyon/Test bölmeleri
        self.train_idx = None
        self.val_idx = None
        self.test_idx = None

        # Referans centroid’leri ve kNN ayarı
        self.centroids = {}            # {level: torch (C,2048) normalized, device}
        self.best_k = {}               # {level: int}

        # Kategori-bazlı birleştirme katsayıları
        self.alpha_per_cat = {}        # {level: np.array shape=(C,), α_c in {0,1} (clip or visual)}

        # Cache yolları
        self.base_dir = r"C:\Users\selam\Desktop\kod deneme\resnet_openclip"
        self.cache_dir = os.path.join(self.base_dir, "cache_maxacc_tr")  # CHANGED (yeni cache klasörü)
        os.makedirs(self.cache_dir, exist_ok=True)

        self.text_cache_path = os.path.join(self.cache_dir, "text_embeds.pkl")
        self.feats_cache_path = os.path.join(self.cache_dir, "img_feats.npz")  # img_clip, img_resnet, img_ids
        self.labels_cache_path = os.path.join(self.cache_dir, "labels.pkl")

    def _load_image(self, image_path):  # CHANGED (yeni fonksiyon, yerelden oku)
        try:
            return Image.open(image_path).convert("RGB")
        except Exception:
            return None
    # -------------------------
    # Kurulum
    # -------------------------
    def setup(self):
        print("🚀 Kurulum başlıyor...")
        ok = self._setup_models() and self._load_data() and self._build_hierarchy()
        if not ok:
            return False
        ok = self._prepare_text_embeddings() and self._prepare_image_embeddings() and self._prepare_labels()
        if not ok:
            return False
        print("✅ Kurulum tamam.")
        return True

    def _setup_models(self):
        try:
            print("📡 OpenCLIP yükleniyor...")
            model_name = "xlm-roberta-base-ViT-B-32"
            pretrained = "laion5b_s13b_b90k"
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained, device=self.device
            )
            tokenizer = open_clip.get_tokenizer(model_name)
            self.openclip_model = model.eval()
            self.openclip_preprocess = preprocess
            self.openclip_tokenizer = tokenizer

            print("🧠 ResNet50 yükleniyor...")
            net = models.resnet50(pretrained=True).eval().to(self.device)
            self.resnet50_model = nn.Sequential(*list(net.children())[:-1]).eval()  # (1,2048,1,1)
            self.resnet50_model.to(self.device)

            self.resnet50_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            print("✅ Modeller hazır.")
            return True
        except Exception as e:
            print(f"❌ Model hatası: {e}")
            return False

    def _load_data(self):
        try:
            self.df = pd.read_csv(self.csv_path)
            print(f"✅ CSV yüklendi: {len(self.df)} satır")
            return True
        except Exception as e:
            print(f"❌ CSV hatası: {e}")
            return False

    def _build_hierarchy(self):
        print("🏗️ Hiyerarşi oluşturuluyor...")
        for L in [1,2,3,4]:
            col = f"category{L}Name_en"
            if col in self.df.columns:
                cats = self.df[col].dropna().unique().tolist()
                self.category_hierarchy[L] = cats
                self.level_categories[L] = cats
                self.level_cat2idx[L] = {c:i for i,c in enumerate(cats)}
                print(f"  L{L}: {len(cats)} kategori")

        # Parent-child eşleştirme
        for L in [2,3,4]:
            pcol = f"category{L-1}Name_en"
            ccol = f"category{L}Name_en"
            if pcol in self.df.columns and ccol in self.df.columns:
                mapping = defaultdict(list)
                for _, r in self.df[[pcol, ccol]].dropna().iterrows():
                    p, c = r[pcol], r[ccol]
                    if c not in mapping[p]:
                        mapping[p].append(c)
                self.parent_child_map[L] = dict(mapping)
                print(f"  Parent-child L{L-1}→L{L}: {len(mapping)} parent")
        return True

    # -------------------------
    # Embedding Hazırlığı
    # -------------------------
    def _prepare_text_embeddings(self):
        # Cache'ten yükle
        if os.path.exists(self.text_cache_path):
            try:
                with open(self.text_cache_path, "rb") as f:
                    packed = pickle.load(f)
                for L, pack in packed.items():
                    self.level_categories[L] = pack["cats"]
                    self.level_cat2idx[L] = {c:i for i,c in enumerate(pack["cats"])}
                    emb = torch.from_numpy(pack["emb"]).to(self.device)  # (C,D) fp16 normalized
                    self.level_text_embeds[L] = emb
                print("📦 Text embed cache yüklendi.")
                return True
            except Exception as e:
                print(f"⚠️ Text cache hatası: {e}")

        print("🧰 Text embedding üretiliyor...")
        cache_payload = {}
        with torch.no_grad():
            for L, cats in self.level_categories.items():
                if not cats: 
                    continue
                # 3 basit şablon
                templates = ["{}", "{} ürünü", "{} kategorisinde bir ürün"]
                prompts = []
                spans = []
                for c in cats:
                    s = len(prompts)
                    for t in templates:
                        prompts.append(t.format(c))
                    spans.append((s, s+len(templates)))

                # Encode (batch)
                all_vecs = []
                bs = 256
                for i in range(0, len(prompts), bs):
                    tok = self.openclip_tokenizer(prompts[i:i+bs]).to(self.device)
                    vec = self.openclip_model.encode_text(tok)
                    vec = normalize_torch(vec, dim=-1)
                    all_vecs.append(vec)
                all_vecs = torch.cat(all_vecs, dim=0)  # (len(prompts),D)

                # Kategori başına ortalama + normalize
                cat_mat = []
                for s,e in spans:
                    m = all_vecs[s:e].mean(dim=0, keepdim=True)
                    m = normalize_torch(m, dim=-1)
                    cat_mat.append(m)
                cat_mat = torch.cat(cat_mat, dim=0)  # (C,D)
                self.level_text_embeds[L] = to_half_on_device(cat_mat, self.device)

                cache_payload[L] = {
                    "cats": cats,
                    "emb": cat_mat.half().cpu().numpy()
                }

        with open(self.text_cache_path, "wb") as f:
            pickle.dump(cache_payload, f)
        print("💾 Text embed cache kaydedildi.")
        return True

    """
    def _load_image_http(self, image_path):
        try:
            url = f"http://n11scdn.akamaized.net/a1/org/{image_path}"
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            return Image.open(BytesIO(r.content)).convert("RGB")
        except Exception:
            return None
    """

    def _encode_image_clip(self, pil_img):
        x = self.openclip_preprocess(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            v = self.openclip_model.encode_image(x)
        v = normalize_torch(v, dim=-1)
        return v.squeeze(0)  # (D,)

    def _encode_image_resnet(self, pil_img):
        x = self.resnet50_transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            f = self.resnet50_model(x).view(1,-1)
        f = normalize_torch(f, dim=-1)
        return f.squeeze(0)  # (2048,)

    def _prepare_image_embeddings(self):
        # Cache'ten yükle
        if os.path.exists(self.feats_cache_path):
            try:
                data = np.load(self.feats_cache_path, allow_pickle=True)
                self.img_ids = data["img_ids"].tolist()
                self.img_clip = torch.from_numpy(data["img_clip"]).to(self.device)
                self.img_resnet = torch.from_numpy(data["img_resnet"]).to(self.device)
                print(f"📦 Görsel embed cache yüklendi: {len(self.img_ids)} örnek")
                return True
            except Exception as e:
                print(f"⚠️ Görsel cache hatası: {e}")

        print("🧰 Görsel embedding’ler üretiliyor (tüm dataset için)...")
        clip_vecs = []
        res_vecs = []
        valid_idx = []

        for idx in tqdm(range(len(self.df)), desc="Encode images"):
            row = self.df.iloc[idx]
            path = row.get("imagePath_s")
            if not isinstance(path, str) or len(path) == 0:
                continue
            img = self._load_image(path)
            if img is None:
                continue
            # encode
            clip_v = self._encode_image_clip(img).half().cpu().numpy()
            res_v  = self._encode_image_resnet(img).half().cpu().numpy()
            clip_vecs.append(clip_v)
            res_vecs.append(res_v)
            valid_idx.append(idx)

        if not clip_vecs:
            print("❌ Görsel embedding üretilemedi.")
            return False

        self.img_ids = valid_idx
        self.img_clip = to_half_on_device(torch.from_numpy(np.stack(clip_vecs)).to(self.device), self.device)
        self.img_resnet = to_half_on_device(torch.from_numpy(np.stack(res_vecs)).to(self.device), self.device)

        np.savez_compressed(
            self.feats_cache_path,
            img_ids=np.array(self.img_ids, dtype=np.int32),
            img_clip=self.img_clip.cpu().numpy(),
            img_resnet=self.img_resnet.cpu().numpy()
        )
        print(f"💾 Görsel embed cache kaydedildi: {len(self.img_ids)} örnek")
        return True

    def _prepare_labels(self):
        # Cache'ten yükle
        if os.path.exists(self.labels_cache_path):
            try:
                with open(self.labels_cache_path, "rb") as f:
                    self.labels = pickle.load(f)
                print("📦 Label cache yüklendi.")
                return True
            except Exception as e:
                print(f"⚠️ Label cache hatası: {e}")

        print("🧰 Label matrisleri hazırlanıyor...")
        self.labels = {}
        for L in [1,2,3,4]:
            cats = self.level_categories.get(L, [])
            if not cats:
                continue
            c2i = self.level_cat2idx[L]
            y = np.full(len(self.img_ids), -1, dtype=np.int32)
            for j, ds_idx in enumerate(self.img_ids):
                v = self.df.iloc[ds_idx].get(f"category{L}Name_en")
                if pd.notna(v) and v in c2i:
                    y[j] = c2i[v]
            self.labels[L] = y

        with open(self.labels_cache_path, "wb") as f:
            pickle.dump(self.labels, f)
        print("💾 Label cache kaydedildi.")
        return True

    # -------------------------
    # Train/Val/Test bölmesi
    # -------------------------
    def _split_data(self, val_ratio=0.2, test_ratio=0.0, seed=42):
        N = len(self.img_ids)
        rng = np.random.RandomState(seed)
        idx = np.arange(N)
        rng.shuffle(idx)
        n_val = int(N * val_ratio)
        n_test = int(N * test_ratio)
        self.val_idx = idx[:n_val]
        self.test_idx = idx[n_val:n_val+n_test] if n_test>0 else np.array([], dtype=np.int32)
        self.train_idx = idx[n_val+n_test:]
        print(f"📚 Split -> train:{len(self.train_idx)} val:{len(self.val_idx)} test:{len(self.test_idx)}")

    # -------------------------
    # Model Öğrenme (α_c ve k seçimi)
    # -------------------------
    def _build_centroids(self, L):
        y = self.labels[L]
        cats = self.level_categories[L]
        C = len(cats)
        emb = self.img_resnet[self.train_idx]     # (Nt,2048)
        lbl = y[self.train_idx]                   # (Nt,)
        cent = torch.zeros((C, self.img_resnet.shape[1]), dtype=torch.float32, device=self.device)
        for c in range(C):
            m = (lbl == c)
            if m.any():
                v = emb[m]
                v = v.mean(dim=0, keepdim=True)
                v = normalize_torch(v, dim=-1)
                cent[c] = v
            else:
                cent[c] = torch.zeros_like(cent[c])
        self.centroids[L] = cent.half()

    def _knn_vote_probs(self, L, query_idx, k):
        """
        L seviyesi için k-NN oylandırma.
        query_idx: np.array indeksleri (val/test)
        return: (Q, C) torch float32 [0..1], her kategori için komşu oranı
        """
        y = self.labels[L]
        cats = self.level_categories[L]
        C = len(cats)
        ref_feats = self.img_resnet[self.train_idx]         # (Nt,2048)
        ref_lbls  = y[self.train_idx]                       # (Nt,)
        qry_feats = self.img_resnet[query_idx]              # (Q,2048)

        sims = cosine_sim_matrix(qry_feats, ref_feats).float()   # (Q,Nt)
        topv, topi = torch.topk(sims, k=min(k, sims.shape[1]), dim=1, largest=True)

        # oy oranı
        probs = torch.zeros((qry_feats.shape[0], C), dtype=torch.float32, device=self.device)
        for qi in range(qry_feats.shape[0]):
            neigh_idx = topi[qi].detach().cpu().numpy().tolist()
            neigh_lbl = ref_lbls[neigh_idx]
            cnt = Counter(int(t) for t in neigh_lbl if t >= 0)
            if cnt:
                s = float(sum(cnt.values()))
                for c, v in cnt.items():
                    probs[qi, int(c)] = v / s
        return probs  # (Q,C)

    def _clip_scores(self, L, idx_array):
        """
        L seviyesi için CLIP image↔text skorları.
        return: (Q,C) torch float32 [−1..1], cosine (logits değil)
        """
        img = self.img_clip[idx_array]           # (Q,D)
        txt = self.level_text_embeds[L]          # (C,D)
        sims = cosine_sim_matrix(img, txt).float()
        return sims  # (Q,C), normalized cosine

    def _visual_scores(self, L, idx_array, k):
        """
        Görsel skor = 0.5*centroid_cos + 0.5*kNN_prob
        Her ikisi de [0..1] bandında; centroid cosine negatif olabilir → [0..1] clamp.
        """
        qry = self.img_resnet[idx_array]         # (Q,2048)
        cent = self.centroids[L]                 # (C,2048)
        cent_cos = cosine_sim_matrix(qry, cent).float()
        cent_cos = torch.clamp(cent_cos, 0.0, 1.0)

        knn_prob = self._knn_vote_probs(L, idx_array, k)    # (Q,C)

        return 0.5*cent_cos + 0.5*knn_prob

    def _select_k_and_alphas(self, L):
        """
        - k (1/3/5) → val doğruluğuna göre seç
        - her kategori için α_c ∈ {0,1} seç (hangi sinyal daha ayırt edici?)
        """
        # 1) k seçimi (global, L seviyesinde)
        best_acc = -1.0
        best_k = 1
        Ks = [1,3,5]
        y = self.labels[L]
        val_mask = self.val_idx
        val_has = y[val_mask] >= 0
        val_idx = val_mask[val_has]
        if len(val_idx) == 0:
            self.best_k[L] = 1
            self.alpha_per_cat[L] = np.zeros(len(self.level_categories[L]), dtype=np.float32)
            return

        clip_val = self._clip_scores(L, val_idx)                 # (Q,C)
        self._build_centroids(L)                                  # centroidler train ile
        # k seçimi
        for k in Ks:
            vis_val = self._visual_scores(L, val_idx, k)         # (Q,C)
            # Basit birleşik skor (0.5/0.5) ile accuracy
            comb = 0.5*clip_val + 0.5*vis_val
            pred = torch.argmax(comb, dim=1).cpu().numpy()
            acc = np.mean(pred == y[val_idx])
            if acc > best_acc:
                best_acc, best_k = acc, k
        self.best_k[L] = best_k

        # 2) Her kategori için α_c seçimi
        # Fikir: pozitif (o kategori) ile negatifler arasında "ortalama ayrım"
        # Δ_clip = mean(clip_pos - clip_neg), Δ_vis = mean(vis_pos - vis_neg)
        # Eğer Δ_clip > Δ_vis → α_c = 1 (CLIP); else α_c = 0 (Visual)
        vis_val = self._visual_scores(L, val_idx, best_k)
        C = clip_val.shape[1]
        alpha = np.zeros(C, dtype=np.float32)

        # Negatif ortalama, tüm örnekler üzerinden kategori c sütununda
        clip_np = clip_val.detach().cpu().numpy()
        vis_np  = vis_val.detach().cpu().numpy()
        yv = y[val_idx]

        for c in range(C):
            pos = (yv == c)
            neg = (yv != c)

            if pos.sum() == 0 or neg.sum() == 0:
                # veri yok → görsel taraf genelde daha güvenli
                alpha[c] = 0.0
                continue

            clip_pos_mean = clip_np[pos, c].mean()
            clip_neg_mean = clip_np[neg, c].mean()
            vis_pos_mean  = vis_np[pos, c].mean()
            vis_neg_mean  = vis_np[neg, c].mean()

            d_clip = clip_pos_mean - clip_neg_mean
            d_vis  = vis_pos_mean  - vis_neg_mean

            alpha[c] = 1.0 if d_clip > d_vis else 0.0

        self.alpha_per_cat[L] = alpha

    # -------------------------
    # Skor ve Tahmin
    # -------------------------
    def _combined_scores(self, L, idx_array):
        """Her kategori için α_c ile birleşik skorlar döndürür. (Q,C)"""
        if len(idx_array) == 0:
            return None
        clip = self._clip_scores(L, idx_array)                    # (Q,C)
        vis  = self._visual_scores(L, idx_array, self.best_k[L])  # (Q,C)
        a = torch.from_numpy(self.alpha_per_cat[L]).to(self.device).view(1,-1)  # (1,C)
        return a*clip + (1.0 - a)*vis

    def _restrict_by_parent(self, L, parent_pred):
        """L seviyesinde, parent_pred’in çocuklarına filtre uygula."""
        if L not in self.parent_child_map:
            return None  # filtre yok
        children = self.parent_child_map[L].get(parent_pred, [])
        return children if children else None

    def predict_one(self, image_path):
        """Tek görsel için hiyerarşik tahmin (L1→L4)."""
        img = self._load_image(image_path)
        if img is None:
            return None
        # Tek örnek için embed
        with torch.no_grad():
            clip = self._encode_image_clip(img).unsqueeze(0)      # (1,D)
            res  = self._encode_image_resnet(img).unsqueeze(0)    # (1,2048)

        # Skorları hesaplamak için geçici olarak buf’lara eklemeden kullan
        preds = {}
        parents = {}

        # L1
        L = 1
        txt = self.level_text_embeds[L]           # (C,D)
        cent = self.centroids.get(L, None)
        if cent is None:                          # garanti için
            self._build_centroids(L)
        # CLIP
        clip_s = cosine_sim_matrix(clip, txt).float()
        # Visual
        cent_s = torch.clamp(cosine_sim_matrix(res, self.centroids[L]).float(), 0.0, 1.0)
        # kNN (train set’e göre)
        k = self.best_k.get(L, 3)
        knn = self._knn_vote_probs(L, np.array([self.train_idx[0]] * 0 + np.array([], dtype=np.int32)), k)  # boş
        # tek örnek için kNN hesapla (pratik, ayrı fonk yazmak yerine hack yok)
        # daha basit: bütün train’e karşı benzerlik al
        ref_feats = self.img_resnet[self.train_idx]
        sims = cosine_sim_matrix(res, ref_feats).float().squeeze(0)
        topv, topi = torch.topk(sims, k=min(k, sims.numel()), largest=True)
        ytrain = self.labels[L][self.train_idx]
        C = len(self.level_categories[L])
        knn_vec = torch.zeros((1, C), dtype=torch.float32, device=self.device)
        cnt = Counter(int(ytrain[int(i)]) for i in topi.detach().cpu().numpy().tolist() if int(ytrain[int(i)]) >= 0)
        if cnt:
            s = float(sum(cnt.values()))
            for c, v in cnt.items():
                knn_vec[0, int(c)] = v / s

        vis = 0.5*cent_s + 0.5*knn_vec
        a = torch.from_numpy(self.alpha_per_cat[L]).to(self.device).view(1,-1)
        comb = a*clip_s + (1.0 - a)*vis
        i1 = torch.argmax(comb, dim=1).item()
        pred1 = self.level_categories[L][i1]
        preds[1] = pred1
        parents[2] = pred1

        # L2, L3, L4 sırayla ve parent filtresi ile
        for L in [2,3,4]:
            if L not in self.level_categories or len(self.level_categories[L]) == 0:
                continue
            # skorları hızlıca tekrar hesaplamak için mevcut img embed’lerini kullanıp aynı adımları uygularız
            txt = self.level_text_embeds[L]
            if L not in self.centroids:
                self._build_centroids(L)
            # clip
            clip_s = cosine_sim_matrix(clip, txt).float()
            # visual
            cent_s = torch.clamp(cosine_sim_matrix(res, self.centroids[L]).float(), 0.0, 1.0)
            # kNN
            k = self.best_k.get(L, 3)
            ref_feats = self.img_resnet[self.train_idx]
            sims = cosine_sim_matrix(res, ref_feats).float().squeeze(0)
            topv, topi = torch.topk(sims, k=min(k, sims.numel()), largest=True)
            ytrain = self.labels[L][self.train_idx]
            C = len(self.level_categories[L])
            knn_vec = torch.zeros((1, C), dtype=torch.float32, device=self.device)
            cnt = Counter(int(ytrain[int(i)]) for i in topi.detach().cpu().numpy().tolist() if int(ytrain[int(i)]) >= 0)
            if cnt:
                s = float(sum(cnt.values()))
                for c, v in cnt.items():
                    knn_vec[0, int(c)] = v / s
            vis = 0.5*cent_s + 0.5*knn_vec
            a = torch.from_numpy(self.alpha_per_cat[L]).to(self.device).view(1,-1)
            comb = a*clip_s + (1.0 - a)*vis

            # parent filtresi
            allowed = self._restrict_by_parent(L, parents.get(L))
            if allowed:
                mask = torch.zeros((1, C), dtype=torch.bool, device=self.device)
                idxs = [self.level_cat2idx[L][c] for c in allowed if c in self.level_cat2idx[L]]
                if idxs:
                    mask[:, idxs] = True
                    comb = torch.where(mask, comb, torch.full_like(comb, -1e9))
            i = torch.argmax(comb, dim=1).item()
            pred = self.level_categories[L][i]
            preds[L] = pred
            parents[L+1] = pred

        return preds

    # -------------------------
    # Eğitim & Değerlendirme
    # -------------------------
    def fit(self, val_ratio=0.2, seed=42):
        self._split_data(val_ratio=val_ratio, seed=seed)
        # L1-L4 için k ve α_c seç
        for L in [1,2,3,4]:
            if L not in self.labels or len(self.level_categories.get(L, [])) == 0:
                continue
            print(f"🔧 L{L} ayarlanıyor (k ve α_c)...")
            self._select_k_and_alphas(L)

    def evaluate(self):
        print("\n📈 Değerlendirme (validation üzerinde)...")
        metrics = {}
        for L in [1,2,3,4]:
            if L not in self.labels or len(self.level_categories.get(L, [])) == 0:
                continue
            y = self.labels[L]
            idx = self.val_idx[y[self.val_idx] >= 0]
            if len(idx) == 0:
                continue
            scores = self._combined_scores(L, idx)  # (Q,C)
            pred = torch.argmax(scores, dim=1).cpu().numpy()
            acc = float(np.mean(pred == y[idx]))
            metrics[L] = {"accuracy": acc, "count": int(len(idx))}
            print(f"  L{L}: acc={acc:.3f} (n={len(idx)})  | k={self.best_k.get(L)}")
        return metrics

    # -------------------------
    # Toplu Test (hiyerarşik)
    # -------------------------
    def test_hierarchical(self, num_samples=500):
        print("\n🎯 Hiyerarşik test başlıyor...")
        # rastgele num_samples seç
        N_all = len(self.img_ids)
        picks = np.random.choice(N_all, size=min(num_samples, N_all), replace=False)

        results = []
        for j in tqdm(picks, desc="Testing"):
            ds_idx = self.img_ids[j]
            row = self.df.iloc[ds_idx]
            img_path = row.get("imagePath_s")

            preds = self.predict_one(img_path)
            if preds is None:
                continue

            # ground-truth
            gt = {L: row.get(f"category{L}Name_en") for L in [1,2,3,4]}
            corr = {L: (preds.get(L) == gt[L]) if (gt[L] is not None and isinstance(gt[L], str)) else None
                    for L in [1,2,3,4]}
            hs = all(corr[L] for L in [1,2,3,4] if corr[L] is not None) if any(corr[L] is not None for L in [1,2,3,4]) else False

            results.append({
                "image_path": img_path,
                "pred_L1": preds.get(1), "gt_L1": gt[1], "ok_L1": corr[1],
                "pred_L2": preds.get(2), "gt_L2": gt[2], "ok_L2": corr[2],
                "pred_L3": preds.get(3), "gt_L3": gt[3], "ok_L3": corr[3],
                "pred_L4": preds.get(4), "gt_L4": gt[4], "ok_L4": corr[4],
                "hier_success": hs
            })

        # raporla
        print("\n====================================================================================================")
        print(f"📈 HİYERARŞİK TEST RAPORU - {len(results)} ÖRNEK")
        lvl_acc = {}
        for L in [1,2,3,4]:
            oks = [r[f"ok_L{L}"] for r in results if r[f"ok_L{L}"] is not None]
            if oks:
                acc = float(np.mean(oks))
                lvl_acc[L] = acc
                print(f"  L{L}: {acc:.3f}  (n={len(oks)})")
        hs = [r["hier_success"] for r in results]
        if hs:
            print(f"🏆 Hiyerarşik Başarı: {float(np.mean(hs)):.3f}  ({sum(hs)}/{len(hs)})")
        print("====================================================================================================")

        # kaydet
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(self.base_dir, "output_maxacc")
        os.makedirs(out_dir, exist_ok=True)
        out_csv = os.path.join(out_dir, f"hier_maxacc_results_{ts}_{len(results)}samples.csv")
        pd.DataFrame(results).to_csv(out_csv, index=False, encoding="utf-8")
        print(f"💾 Sonuçlar kaydedildi: {out_csv}")

        return results


# =========================
# Çalıştırma
# =========================
def main():
    predictor = HierarchicalMaxAccPredictor()
    if not predictor.setup():
        print("❌ Kurulum başarısız.")
        return

    # α_c ve k seçimi (validation ile)
    predictor.fit(val_ratio=0.2, seed=42)

    # Validation metrikleri
    predictor.evaluate()

    # Hiyerarşik test
    predictor.test_hierarchical(num_samples=4920)


if __name__ == "__main__":
    main()
