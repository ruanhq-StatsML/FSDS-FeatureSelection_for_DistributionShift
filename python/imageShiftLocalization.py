#!pip install albumentations --quiet
#!pip install open_clip_torch --quiet
#!pip install FrEIA --quiet
#!pip install torch --quiet
#!pip install tqdm --quiet


import numpy as np
import pandas as pd
#Reading in the data:
import os
from google.colab import drive
drive.mount('/content/drive')
if not os.path.exists('lake_or_pond_processed'):
    os.mkdir('lake_or_pond_processed')
!unzip -o "/content/drive/MyDrive/lake_or_pond_processed.zip" -d "lake_or_pond_processed"

import shutil
file_lists = [f for f in os.listdir('/content/lake_or_pond_processed/lake_or_pond_processed')
if f.endswith('.jpg')]
f_random = np.random.choice(np.arange(len(file_lists)), round(len(file_lists)//3), replace= False)
for i in f_random:
    shutil.move(os.path.join('/content/lake_or_pond_processed/lake_or_pond_processed', file_lists[i]),
                '/content/lake_or_pond_processed/lake_or_pond_processed_eval')


#Model and extracting the image representation.
from PIL import Image
import os
import numpy as np
import pandas as pd
import open_clip
Image.MAX_IMAGE_PIXELS = None
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
def extract_embedding(image_tensor, model):
    with torch.no_grad():
        features = model.forward_features(image_tensor)
        embedding = features.mean(dim=[-2, -1])
    return embedding
class ImageDataset(Dataset):
    def __init__(self, image_dir, preprocess):
        self.image_dir = image_dir
        self.preprocess = preprocess
        self.images = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff')
        self.image_paths = []
        # os.walk automatically traverses all nested subfolders
        for root, _, files in os.walk(image_dir):
            for file in files:
                if file.lower().endswith(valid_extensions):
                    full_path = os.path.join(root, file)
                    self.image_paths.append(full_path)
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
       img_path = os.path.join(self.image_dir, self.images[idx])
       img = Image.open(img_path).convert('RGB')
       file_name = os.path.relpath(img_path, self.image_dir)
       return self.preprocess(img), file_name


model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
model = model.eval()
train_dataset = ImageDataset(image_dir = '/content/lake_or_pond_processed/lake_or_pond_processed', preprocess = preprocess)
train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = False)
import os
import cv2
import zipfile
import shutil
import albumentations as A
CORRUPTIONS = {
    'gaussian': A.GaussNoise(std_range=(0.6, 0.8), p=1.0),
    'shot': A.ISONoise(color_shift=(0.0, 0.0), intensity=(0.75, 0.75), p=1.0),
    'impulse': A.SaltAndPepper(amount=(0.25, 0.25), p=1.0),
    'defocus': A.Defocus(radius=(8, 8), alias_blur=(0.25, 0.25), p=1.0),
    'glass': A.GlassBlur(sigma=1.5, max_delta=4, iterations=2, p=1.0),
    'motion': A.MotionBlur(blur_limit=25, p=1.0),
    'zoom': A.ZoomBlur(max_factor_range=(0.25, 0.95), step_factor_range=(0.2, 0.2), p=1.0),
    'fog': A.RandomFog(fog_coef_range=(0.25, 0.95), p=1.0),
    'frost': A.GlassBlur(sigma=2.5, max_delta=2, iterations=2, p=1.0),
    'snow': A.RandomSnow(snow_point_lower=2.0, snow_point_upper=0.2, p=1.0),
    'spatter': A.Spatter(mode="rain", p=1.0),
    'brightness': A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.0, p=1.0),
    'contrast': A.RandomBrightnessContrast(brightness_limit=0.0, contrast_limit=0.5, p=1.0),
    'elastic': A.ElasticTransform(alpha=50, sigma=10, alpha_affine=75, p=1.0),
    'pixelate': A.Downscale(scale_min=0.75, scale_max=0.75, p=1.0),
}
import os
import cv2
import shutil
import zipfile
import albumentations as A
def generate_corruptions(input_image_folder, output_root='corrupted_output', zip_name='corruptions.zip'):
    img_paths = [f for f in os.listdir(input_image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if os.path.exists(output_root):
        shutil.rmtree(output_root)
    os.makedirs(output_root, exist_ok=True)
    for name in CORRUPTIONS.keys():
        os.makedirs(os.path.join(output_root, name), exist_ok=True)
    for img_file in img_paths:
        img_full_path = os.path.join(input_image_folder, img_file)
        img = cv2.imread(img_full_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        base_name, ext = os.path.splitext(img_file)
        for idx, (name, transform) in enumerate(CORRUPTIONS.items(), 1):
            augmented = transform(image=img_rgb)
            corrupted_img = augmented['image']
            save_path = os.path.join(output_root, name, f"{base_name}_{name}{ext}")
            cv2.imwrite(save_path, cv2.cvtColor(corrupted_img, cv2.COLOR_RGB2BGR))
            print(f'finished image {img_file} image with {str(transform).split('(')[0]} perturbations')
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(output_root):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, start=os.path.dirname(output_root))
                zipf.write(file_path, arcname)
generate_corruptions(input_image_folder = '/content/lake_or_pond_processed/lake_or_pond_processed')
#Also incorporate other forms:
import numpy as np
import pandas as pd
#Reading in the data:
import os
from google.colab import drive
drive.mount('/content/drive')
if not os.path.exists('space_facility_processed.zip'):
    os.mkdir('space_facility_processed.zip')
!unzip -o "/content/drive/MyDrive/space_facility_processed.zip" -d "space_facility_processed"
from google.colab import userdata
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
HF_TOKEN = userdata.get('HF_TOKEN')
GITHUB_TOKEN = userdata.get('GITHUB_TOKEN')
from google.colab import drive
drive.mount('/content/drive')
import os
if not os.path.exists('fmow_subset'):
    os.mkdir('fmow_subset')
!unzip -o "/content/drive/MyDrive/fmow_subset.zip" -d "fmow_subset"
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from diffusers import AutoencoderKL
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
DATASET_PAIRS = [
    ("/content/fmow_subset/fmow_subset/space_facility", "/content/fmow_subset/fmow_subset/border_checkpoint", "part1"),
    ("/content/fmow_subset/fmow_subset/airport_processed", "/content/fmow_subset/fmow_subset/border_checkpoint", "part2"),
    ("/content/fmow_subset/fmow_subset/lake_or_pond", "/content/fmow_subset/fmow_subset/space_facility", "part3"),
    ("/content/fmow_subset/fmow_subset/space_facility", "/content/fmow_subset/fmow_subset/shipyard_processed", "part4"),
]

IMAGE_SIZE = 64
BATCH_SIZE = 32
NUM_WORKERS = 2
SEED = 42
TOP_K = [1, 5, 10, 20, 50]
MAX_HEATMAPS = 20
class ImageDataset(Dataset):
    def __init__(self, image_dir: str, max_image = 1000, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths: List[str] = []
        for root, _, files in os.walk(image_dir):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                    self.image_paths.append(os.path.join(root, f))
                if len(self.image_paths) > 1000:
                    break
        print(f"Found {len(self.image_paths)} images in {image_dir}")
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, os.path.basename(self.image_paths[idx])

def make_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

def encode_vae(model, x):
    with torch.no_grad():
        latent = model.encode(x * 2.0 - 1.0).latent_dist.mode()
        return latent.view(latent.size(0), -1)

def decode_vae(model, z):
    with torch.no_grad():
        img = model.decode(z.view(-1, 4, 8, 8)).sample
        return (img + 1.0) / 2.0

def encode_dataset(model, loader):
    z_list = []
    with torch.no_grad():
        for batch_imgs, _ in tqdm(loader, desc="Encoding"):
            z_list.append(encode_vae(model, batch_imgs.to(device)).cpu())
    return torch.cat(z_list, dim=0)

def rf_domain_classifier(z_train, z_eval, seed=42):
    X = np.vstack([z_train.numpy(), z_eval.numpy()])
    y = np.concatenate([np.zeros(len(z_train)), np.ones(len(z_eval))])
    rf = RandomForestClassifier(
        n_estimators=150, max_depth=10,
        max_features=round(np.sqrt(X.shape[1]) / X.shape[1], 3),
        min_samples_leaf=max(1, round(np.sqrt(X.shape[0]) / 2)),
        random_state=seed, n_jobs=-1, oob_score=True,
    )
    rf.fit(X, y)
    return rf.feature_importances_, float(rf.oob_score_)

def evaluate_region_selection(loader_eval, model, z_delta, importance, top_k_list):
    sorted_idx = np.argsort(-importance)
    z_delta = z_delta.to(device)
    results = {k: {"gini":[],"top10":[],"active":[]} for k in sorted(top_k_list)}
    with torch.no_grad():
        for batch_imgs, _ in tqdm(loader_eval, desc="  🔍 Region eval"):
            batch_imgs = batch_imgs.to(device)
            z_ref = encode_vae(model, batch_imgs)
            for k in top_k_list:
                top_idx = sorted_idx[:k]
                z_pert = z_ref.clone(); z_pert[:, top_idx] += z_delta[top_idx]
                img_pert = decode_vae(model, z_pert)
                heatmap = (batch_imgs - img_pert).abs().mean(dim=1)
                for b in range(len(batch_imgs)):
                    h = heatmap[b].cpu().numpy().flatten()#looking at the concentration in the top 10% percent, thanks
                    h_sorted = np.sort(h)
                    n = len(h_sorted)
                    gini = (2 * np.sum(np.arange(1, n+1) * h_sorted) / (n * np.sum(h_sorted))) - (n+1)/n
                    gini = float(np.clip(gini, 0, 1))
                    top10_ratio = h_sorted[:max(1, int(0.1*n))].sum() / (h.sum() + 1e-8)
                    active_area = (h > (h.mean() + h.std())).sum() / n
                    results[k]["gini"].append(float(np.clip(gini, 0, 1)))
                    results[k]["top10"].append(top10_ratio)
                    results[k]["active"].append(active_area)
    summary = {}
    for k in top_k_list:
        summary[k] = {
            "gini_coeff": float(np.mean(results[k]["gini"])),
            "top10_ratio": float(np.mean(results[k]["top10"])),
            "active_area": float(np.mean(results[k]["active"])),
        }
        print(f"K={k:2d}|Gini:{summary[k]['gini_coeff']:.4f}|Top10%:{summary[k]['top10_ratio']:.4f}|Active:{summary[k]['active_area']:.4f}")
    return summary

def save_visualization(img, img_pert, heatmap, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    def to_np(t):
        return t.squeeze(0).permute(1, 2, 0).cpu().numpy()
    axes[0].imshow(to_np(img).clip(0, 1)); axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(to_np(img_pert).clip(0, 1)); axes[1].set_title("Perturbed"); axes[1].axis("off")
    axes[2].imshow(to_np(img).clip(0, 1)); axes[2].imshow(heatmap, cmap="jet", alpha=0.5)
    axes[2].set_title("Heatmap (Gini insight)"); axes[2].axis("off")
    plt.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚡ Using device: {device}")
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device).eval()

overall_records = []

for idx, (train_dir, eval_dir, exp_name) in enumerate(DATASET_PAIRS, 1):
    for beta in (1 + np.arange(4)):
        train_domain = str(train_dir).split("/")[-1]
        eval_domain = str(eval_dir).split("/")[-1]
        print("=" * 70)
        BASE_OUTPUT_DIR = f"/content/saved_heatmaps_{train_domain}_{eval_domain}"   # 输出根目录
        if not os.path.exists(BASE_OUTPUT_DIR):
            os.makedirs(BASE_OUTPUT_DIR)
        print("=" * 70)
        print(f"[{idx}/{len(DATASET_PAIRS)}] >>> Experiment: {exp_name}")
        print(f"Train: {train_dir}\n  Eval:{eval_dir}")
        print("=" * 70)
        out_dir = Path(BASE_OUTPUT_DIR) / exp_name
        out_dir.mkdir(parents=True, exist_ok=True)
        transform = make_transform()
        loader_train = DataLoader(ImageDataset(train_dir, max_image = 1000, transform = make_transform()), batch_size=BATCH_SIZE,
                                  shuffle=True, num_workers=NUM_WORKERS, pin_memory=(device.type=="cuda"))
        loader_eval = DataLoader(ImageDataset(eval_dir, max_image = 1000, transform = make_transform()), batch_size=BATCH_SIZE,
                                shuffle=False, num_workers=NUM_WORKERS, pin_memory=(device.type=="cuda"))
        z_train, z_eval = encode_dataset(vae, loader_train), encode_dataset(vae, loader_eval)
        importance, oob_auc = rf_domain_classifier(z_train, z_eval, seed=SEED)
        vimp_rank = np.argsort(-importance)
        z_delta = z_eval.mean(0) - z_train.mean(0)
        print(f"OOB AUC: {oob_auc:.4f} | Top 5 dims: {vimp_rank[:5].tolist()}")
        summary = evaluate_region_selection(loader_eval, vae, z_delta, importance, TOP_K)
        with open(out_dir / f"summary_{beta}.json", "w") as f:
            json.dump({str(k): v for k, v in summary.items()}, f, indent=2)
        torch.save(z_train, out_dir / f"z_train_{beta}.pt"); torch.save(z_eval, out_dir / f"z_eval_{beta}.pt")
        torch.save(z_delta.cpu(), out_dir / f"z_delta_{beta}.pt")
        np.save(out_dir / f"vimp_rank_{beta}.npy", vimp_rank)
        #look AT top vimp especially the summary here
        #summary = evaluate_region_selection(loader_eval, vae, z_delta, importance, TOP_K)
        pd.DataFrame({"dim": np.arange(len(importance)), "importance": importance,
                      "rank": np.argsort(vimp_rank)+1}).to_csv(out_dir / f"vimp_table_{beta}.csv", index=False)
        gini_k5 = summary.get(5, {}).get("gini_coeff", 0)
        overall_records.append({"experiment": exp_name, "oob_auc": oob_auc, "gini_coeff_K5": gini_k5})
        max_k, top_idx = max(TOP_K), vimp_rank[:max(TOP_K)]
        count = 0
        for batch_imgs, _ in loader_eval:
            if count >= MAX_HEATMAPS: break
            batch_imgs = batch_imgs.to(device)
            z_ref = encode_vae(vae, batch_imgs)
            for b in range(min(len(batch_imgs), MAX_HEATMAPS - count)):
                z_pert = z_ref[b:b+1].clone()
                z_pert[:, top_idx] += beta * z_delta[top_idx].to(device)
                img_pert = decode_vae(vae, z_pert)
                heatmap = (batch_imgs[b:b+1] - img_pert).abs().mean(dim=1).squeeze(0).cpu().numpy()
                save_visualization(batch_imgs[b:b+1], img_pert, heatmap, out_dir / f"heatmap_{count:04d}_{beta}.png")
                count += 1
        print(f" Saved {count} heatmaps.")
        if gini_k5 >= 0.45:
            print(" Separable and localization attribution works well")
        else:
            print(" Separable and localization attribution didn't works well \n You'd better check the dimension again. ")
if overall_records:
    df_summary = pd.DataFrame(overall_records).sort_values("gini_coeff_K5", ascending=False)
    df_summary.to_csv(Path(BASE_OUTPUT_DIR) / f"GINI_RANKING_{beta}.csv", index=False)
    print(df_summary[["experiment", "gini_coeff_K5", "oob_auc"]].to_string(index=False))

df_sum2 = df_summary.groupby('experiment').agg(
    {'gini_coeff_K5': 'mean', 'oob_auc': 'mean'}
).reset_index(drop = True)
df_sum2['pairs'] = ['spacefacilitybordercheckpoint',
                    'airportprocessedbordercheckpoint',
                    'lakeorpondspacefacility',
                    'spacefacilityshipyardprocessed']
df_sum2.columns = ['gini', 'oobauc', 'pairs']
df_sum2.to_csv('imageresult1.csv')
df_sum3 = pd.read_csv("/content/saved_heatmaps_space_facility_shipyard_processed/GINI_RANKING_4.csv")
df_grp2 = df_sum3.groupby('experiment').agg(
    {'gini_coeff_K5': 'mean', 'oob_auc': 'mean'}
).reset_index(drop = True)
df_grp2['pairs'] = ['lakepondborder',
                    'spaceborder',
                    'airportship',
                    'lakeship']
df_grp2.columns = ['gini', 'oobauc', 'pairs']

df_whole = pd.concat([df_grp2, df_sum2], axis = 0)
np.round(df_whole,4).to_csv('GINIresultFMOW1.csv')








































