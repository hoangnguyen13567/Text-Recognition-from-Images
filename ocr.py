import os
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import easyocr
import logging
import torchvision.models as models
import matplotlib.colors as mcolors

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==== CONFIG ====
input_image_path = "Train/img59.jpg"  # Thay bằng đường dẫn ảnh thực tế
output_vis_dir = "test_detect_recognition_vis"  # Thư mục lưu kết quả visualize
recognition_model_path = "cr_recognizer2.pt"  # Đường dẫn tới model recognition
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_seq_len = 50  # Độ dài tối đa của chuỗi dự đoán
beam_width = 5  # Số beam trong beam search

# Tạo thư mục lưu kết quả
os.makedirs(output_vis_dir, exist_ok=True)

# ==== CHARSET ====
charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,-!?'\""

# ==== VOCAB ====
checkpoint = torch.load(recognition_model_path, map_location=device)
vocab = checkpoint['vocab']
inv_vocab = {v: k for k, v in vocab.items()}

# ==== HÀM PHỤ ====
def crop_polygon(image, coords):
    """Cắt vùng đa giác từ ảnh."""
    coords = np.array(coords, dtype=np.float32)
    if coords.shape[0] != 4:
        x, y, w, h = cv2.boundingRect(coords.astype(np.int32))
        return image[y:y+h, x:x+w]
    width = int(max(np.linalg.norm(coords[0] - coords[1]), np.linalg.norm(coords[2] - coords[3])))
    height = int(max(np.linalg.norm(coords[0] - coords[3]), np.linalg.norm(coords[1] - coords[2])))
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(coords, dst)
    warped = cv2.warpPerspective(image, M, (width, height))
    return warped

def preprocess_crop(crop):
    """Chuẩn hóa ảnh crop trước khi đưa vào model recognition."""
    crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    crop_tensor = torch.tensor(crop_gray / 255.0, dtype=torch.float32).unsqueeze(0)  # [1, H, W]
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((64, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])  # Grayscale
    ])
    return transform(crop_tensor).unsqueeze(0).to(device)  # [1, 1, H, W]

def visualize_attention(crop, attention_weights, predicted_text, output_path):
    """Visualize attention weights for each character in the predicted text."""
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    
    # If no attention weights are available, just show the crop with the prediction
    if not attention_weights or len(attention_weights) == 0:
        logging.warning("No attention weights available for visualization")
        plt.figure(figsize=(5, 4))
        plt.imshow(crop_rgb)
        plt.title(f"Predicted: {predicted_text}")
        plt.axis("off")
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        return

    # Number of characters to visualize (excluding <sos> and <eos>)
    num_chars = len(predicted_text)
    if num_chars == 0:
        logging.warning("Predicted text is empty, skipping attention visualization")
        plt.figure(figsize=(5, 4))
        plt.imshow(crop_rgb)
        plt.title(f"Predicted: {predicted_text}")
        plt.axis("off")
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        return

    # Create a color map for each character
    colors = list(mcolors.TABLEAU_COLORS.values())  # Use Tableau colors for distinct hues
    if len(colors) < num_chars:
        colors = colors * (num_chars // len(colors) + 1)  # Repeat colors if needed

    # Set up the plot: 1 row for the crop, 1 row for each character's attention map
    num_rows = num_chars + 1  # +1 for the original crop
    plt.figure(figsize=(10, 2 * num_rows))

    # Plot the original crop with the predicted text
    plt.subplot(num_rows, 2, (1, 2))
    plt.imshow(crop_rgb)
    plt.title(f"Predicted: {predicted_text}")
    plt.axis("off")

    # Plot attention map for each character
    for i in range(num_chars):
        # Get the attention map for the i-th character (i+1 to skip <sos>)
        if i + 1 < len(attention_weights):
            attn_map = attention_weights[i + 1].squeeze(0).cpu().numpy()  # [H, W]
            attn_map = cv2.resize(attn_map, (crop.shape[1], crop.shape[0]))

            # Plot the crop with the attention map overlay
            plt.subplot(num_rows, 2, (2 * (i + 1) + 1, 2 * (i + 1) + 2))
            plt.imshow(crop_rgb)
            plt.imshow(attn_map, cmap='jet', alpha=0.5)  # Use jet colormap for attention
            plt.title(f"Attention for char '{predicted_text[i]}' (pos {i+1})")
            plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def decode_prediction(logits, inv_vocab):
    """Giải mã output của model recognition thành text."""
    pred_ids = logits.argmax(dim=-1).cpu().numpy()[0]  # [seq_len]
    text = ''
    for id in pred_ids:
        if id == vocab['<eos>']:
            break
        if id != vocab['<sos>'] and id != vocab['<pad>']:
            text += inv_vocab.get(id, '')
    return text.strip()

def beam_search_decode(model, img, vocab, inv_vocab, beam_width=5, max_seq_len=50):
    """Thực hiện beam search để dự đoán chuỗi text."""
    model.eval()
    with torch.no_grad():
        # Khởi tạo beam với <sos>
        beams = [(torch.tensor([[vocab['<sos>']]], dtype=torch.long).to(device), 0.0, [])]  # (tgt_input, log_prob, attn_maps)
        for _ in range(max_seq_len):
            new_beams = []
            for tgt_input, log_prob, attn_maps in beams:
                logits, attn_map = model(img, tgt_input, return_attn=True)
                log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)  # [batch=1, vocab_size]
                topk_probs, topk_ids = log_probs[0].topk(beam_width)
                for prob, token_id in zip(topk_probs, topk_ids):
                    token_id = token_id.item()
                    new_prob = log_prob + prob.item()
                    new_tgt_input = torch.cat([tgt_input, torch.tensor([[token_id]], dtype=torch.long).to(device)], dim=1)
                    new_attn_maps = attn_maps + [attn_map] if attn_map is not None else attn_maps
                    if token_id == vocab['<eos>']:
                        # Trả về beam tốt nhất khi gặp <eos>
                        logits = model(img, new_tgt_input, return_attn=False)
                        return logits, new_attn_maps
                    new_beams.append((new_tgt_input, new_prob, new_attn_maps))
            # Giữ lại beam_width beams tốt nhất
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
        # Nếu không gặp <eos>, trả về beam tốt nhất
        best_beam = beams[0]
        logits = model(img, best_beam[0], return_attn=False)
        return logits, best_beam[2]

# ==== MODEL DEFINITION ====
class CNNEncoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        layers = list(resnet.children())[:-2]
        self.backbone = nn.Sequential(*layers)
        self.conv_proj = nn.Conv2d(2048, d_model, kernel_size=1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):  # x: [B, 1, H, W]
        if x.size(1) == 1:
            x = x.expand(-1, 3, -1, -1)
        feat = self.backbone(x)
        feat = self.conv_proj(feat)
        feat = self.dropout(feat)
        B, C, H, W = feat.shape
        return feat.permute(0, 2, 3, 1).reshape(B, H * W, C), (H, W)

class CustomDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.3):
        super().__init__(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.cross_attn_weights = None

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                tgt_is_causal=True, memory_is_causal=False):
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        attn_output, attn_weights = self.multihead_attn(tgt, memory, memory,
                                                        attn_mask=memory_mask,
                                                        key_padding_mask=memory_key_padding_mask,
                                                        need_weights=True)
        self.cross_attn_weights = attn_weights
        tgt = tgt + self.dropout2(attn_output)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

class OCRRecognizer(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_layers=4, num_heads=4, ff_dim=512, max_len=2048):
        super().__init__()
        self.encoder = CNNEncoder(d_model)
        self.pos_embed_enc = nn.Parameter(torch.randn(1, max_len, d_model))  # Encoder positional embedding
        self.pos_embed_dec = nn.Parameter(torch.randn(1, max_len, d_model))  # Decoder positional embedding
        self.token_embed = nn.Embedding(vocab_size, d_model)
        decoder_layer = CustomDecoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=ff_dim, dropout=0.3)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, img, tgt_input, return_attn=False):
        B = img.size(0)
        enc, (H, W) = self.encoder(img)
        S = enc.size(1)
        if S > self.pos_embed_enc.size(1):
            raise ValueError(f"Encoder output S={S} > max_len={self.pos_embed_enc.size(1)}")
        memory = enc + self.pos_embed_enc[:, :S, :]  # Use pos_embed_enc
        memory = memory.permute(1, 0, 2)

        tgt = self.token_embed(tgt_input)
        T = tgt.size(1)
        if T > self.pos_embed_dec.size(1):
            raise ValueError(f"Target sequence T={T} > max_len={self.pos_embed_dec.size(1)}")
        tgt = tgt + self.pos_embed_dec[:, :T, :]  # Use pos_embed_dec
        tgt = tgt.permute(1, 0, 2)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(0)).to(tgt.device)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask)
        out = output.permute(1, 0, 2)
        logits = self.fc_out(out)
        if return_attn:
            attn_map = self.decoder.layers[-1].cross_attn_weights
            if attn_map is not None:
                attn_map = attn_map.mean(dim=1).reshape(tgt_input.size(0), H, W)  # Mean over heads
            return logits, attn_map
        return logits

# Load model recognition
model = OCRRecognizer(vocab_size=len(vocab)).to(device)
try:
    model.load_state_dict(checkpoint['model'])
    model.eval()
    logging.info(f"✅ Loaded recognition model from {recognition_model_path}")
except Exception as e:
    logging.error(f"❌ Failed to load recognition model: {e}")
    exit()

# ==== XỬ LÝ ẢNH ĐẦU VÀO ====
reader = easyocr.Reader(['en'], gpu=True)
image = cv2.imread(input_image_path)
if image is None:
    logging.error(f"❌ Không thể đọc ảnh: {input_image_path}")
    exit()

# Chuẩn hóa ảnh
image = cv2.convertScaleAbs(image, alpha=1.5, beta=0)
h, w = image.shape[:2]
if h > 2000 or w > 2000:
    scale = min(2000/h, 2000/w)
    image = cv2.resize(image, (int(w*scale), int(h*scale)))

# Phát hiện vùng text bằng EasyOCR
try:
    detected = reader.readtext(image, text_threshold=0.5, low_text=0.5)
    detected_regions = [res[0] for res in detected]
except Exception as e:
    logging.warning(f"❌ EasyOCR fail: {e}")
    detected_regions = []

logging.info(f"🔍 Detected {len(detected_regions)} text regions")

# Sao chép ảnh để visualize
image_copy = image.copy()

# ==== PREDICTION VÀ VISUALIZATION ====
for i, coords in enumerate(detected_regions):
    # Crop vùng text
    crop = crop_polygon(image, coords)
    if crop is None or crop.shape[0] <= 5 or crop.shape[1] <= 5:
        logging.warning(f"⚠️ Crop {i} không hợp lệ, bỏ qua")
        continue

    # Chuẩn hóa crop và đưa vào model recognition
    crop_tensor = preprocess_crop(crop)
    
    # Beam search decoding
    logits, attn_maps = beam_search_decode(model, crop_tensor, vocab, inv_vocab, beam_width=beam_width, max_seq_len=max_seq_len)

    # Giải mã dự đoán
    predicted_text = decode_prediction(logits, inv_vocab)
    logging.info(f"✅ Crop {i}: Predicted text: {predicted_text}")

    # Visualize attention
    attention_vis_path = os.path.join(output_vis_dir, f"crop_{i}_attention.jpg")
    visualize_attention(crop, attn_maps, predicted_text, attention_vis_path)

    # Vẽ vùng và text dự đoán lên ảnh gốc
    poly_pts = np.array(coords, dtype=np.int32)
    cv2.polylines(image_copy, [poly_pts], isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.putText(image_copy, predicted_text, tuple(poly_pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

# Lưu ảnh kết quả
output_image_path = os.path.join(output_vis_dir, "final_output.jpg")
cv2.imwrite(output_image_path, image_copy)
logging.info(f"🌟 Đã lưu kết quả tại {output_image_path}")