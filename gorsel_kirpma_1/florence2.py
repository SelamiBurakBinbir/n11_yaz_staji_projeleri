# =========================================
# Hücre 0 — Yollar & klasör hazırlığı
# =========================================
from pathlib import Path
import os, time, re

BASE_DIR      = Path(r"C:\Users\selam\Desktop\kod deneme\grounding_dino_deneme")
LABELS_FILE   = BASE_DIR / "labels_updated.txt"
OUT_DIR       = BASE_DIR / "outputs_2"
ANALYSIS_FILE = BASE_DIR / "output_analysis_2.txt"

OUT_DIR.mkdir(exist_ok=True)

# =========================================
# Hücre 1 — Model kurulumu & yardımcılar
# =========================================
import torch, matplotlib.pyplot as plt, matplotlib.patches as patches
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float32
MODEL  = "microsoft/Florence-2-base"

processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
model     = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=DTYPE, trust_remote_code=True
).to(DEVICE)

def _safe_name(txt: str) -> str:
    """Dosya-ismi olarak zararsız hâle getir."""
    txt = re.sub(r"\s+", "_", txt.strip())
    return re.sub(r"[^A-Za-z0-9_\-şŞıİçÇöÖüÜğĞ]", "_", txt)

# ---- Florence-2 inference ----
def florence_run(image: Image.Image, phrase: str):
    """Phrase-grounding modunda inference + saf model süresi (ms) döndür."""
    token  = "<CAPTION_TO_PHRASE_GROUNDING>"
    prompt = token + phrase

    inputs = processor(text=prompt, images=image, return_tensors="pt")
    # float tensörleri modele uyumlu dtype'a döndür
    for k, v in inputs.items():
        if torch.is_floating_point(v):
            inputs[k] = v.to(model.dtype)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # --- sadece model generate süresini ölç ---
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=512)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    runtime_ms = (time.perf_counter() - t0) * 1000

    raw    = processor.batch_decode(output, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(
        raw, task=token, image_size=image.size
    )[token]

    bboxes = parsed.get("bboxes", [])
    labels = parsed.get("labels") or [phrase] * len(bboxes)
    return raw, bboxes, labels, runtime_ms

# ---- Kutulu görseli kaydet ----
def save_result(image, bboxes, labels, save_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image); ax.axis("off"); ax.set_title("Kutulu Çıktı", fontsize=11)

    for box, lbl in zip(bboxes, labels):
        x0, y0, x1, y1 = box
        rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                 linewidth=2, edgecolor="red", facecolor="none")
        ax.add_patch(rect)
        ax.text(x0, y0 - 5, lbl, color="red", fontsize=8,
                bbox=dict(facecolor="white", alpha=0.6, pad=1))
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

# =========================================
# Hücre 2 — Toplu inference döngüsü
# =========================================
with open(LABELS_FILE, encoding="utf-8") as f_labels, \
     open(ANALYSIS_FILE, "w", encoding="utf-8") as f_out:

    for line in f_labels:
        line = line.strip()
        if not line or "|" not in line:
            continue  # boş veya hatalı satır

        prod_id, phrase, img_path = [p.strip() for p in line.split("|", 2)]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[{prod_id}] Görsel açılamadı → {e}")
            continue

        raw, bboxes, labels, runtime_ms = florence_run(image, phrase)

        # --- Kutulu görseli kaydet ---
        out_name = f"{prod_id}__{_safe_name(phrase)}.jpg"
        out_path = OUT_DIR / out_name
        save_result(image, bboxes, labels, out_path)

        # --- Analysis satırı ---
        if bboxes:
            obj_str = ", ".join(labels)
        else:
            obj_str = "WARNING_EMPTY"

        f_out.write(
            f"{prod_id} | {obj_str} | {runtime_ms:.2f}ms | {out_path}\n"
        )

        print(f"[{prod_id}] tamam → {len(bboxes)} kutu, "
              f"{runtime_ms:.2f} ms, {out_path.name}")
