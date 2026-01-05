import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torchvision.models as models
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchmetrics.text import CharErrorRate, WordErrorRate
from collections import Counter

# ====================== CONFIG ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
label_file = "labels.txt"
crop_folder = "images_crop"
epochs = 100
batch_size = 64
max_len = 2048
patience = 3

# ====================== VOCAB BUILDER ======================
def build_vocab(label_file):
    if not os.path.exists(label_file):
        raise FileNotFoundError(f"Label file not found: {label_file}")
    vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2}
    total_samples = 0
    filtered_samples = 0
    with open(label_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            total_samples += 1
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"Invalid format at line {i}: {line.strip()}")
            text = parts[1]
            if '#' in text:  # Bỏ qua các mẫu có "#"
                continue
            filtered_samples += 1
            for char in text:
                if char not in vocab:
                    vocab[char] = len(vocab)
    inv_vocab = {v: k for k, v in vocab.items()}
    print(f"Total samples before filtering '#': {total_samples}")
    print(f"Total samples after filtering '#': {filtered_samples}")
    return vocab, inv_vocab

# ====================== DATASET ======================
class OCRDataset(Dataset):
    def __init__(self, crop_folder, label_file, vocab, img_size=(256, 64)):
        self.crop_folder = crop_folder
        self.vocab = vocab
        self.img_size = img_size
        self.data = []
        if not os.path.exists(crop_folder):
            raise FileNotFoundError(f"Crop folder not found: {crop_folder}")
        with open(label_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                parts = line.strip().split(maxsplit=1)
                if len(parts) != 2:
                    raise ValueError(f"Invalid format at line {i}: {line.strip()}")
                img_path, text = parts
                if '#' in text:  # Bỏ qua các mẫu có "#"
                    continue
                img_path = os.path.join(crop_folder, img_path)
                if os.path.exists(img_path):
                    self.data.append((img_path, text))
                else:
                    print(f"Warning: Image not found at {img_path}, skipping")
        if not self.data:
            raise ValueError("No valid data found after processing")
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(15),
            transforms.GaussianBlur(kernel_size=5),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, text = self.data[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.resize(img, self.img_size)
        img = img / 255.0
        img_tensor = torch.tensor(img, dtype=torch.float32)
        img_tensor = self.transform(img_tensor)
        tokens = [self.vocab['<sos>']] + [self.vocab[c] for c in text] + [self.vocab['<eos>']]
        target = torch.tensor(tokens, dtype=torch.long)
        return img_tensor, target, text

# ====================== COLLATE ======================
def collate_fn(batch):
    images, targets, texts = zip(*batch)
    images = torch.stack(images)
    max_len = max(len(t) for t in targets)
    padded_targets = torch.zeros((len(targets), max_len), dtype=torch.long)
    for i, t in enumerate(targets):
        padded_targets[i, :len(t)] = t
    return images, padded_targets, texts

# ======= CNN ENCODER =======
class CNNEncoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        layers = list(resnet.children())[:-2]
        self.backbone = nn.Sequential(*layers)
        self.conv_proj = nn.Conv2d(2048, d_model, kernel_size=1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        if x.size(1) == 1:
            x = x.expand(-1, 3, -1, -1)
        feat = self.backbone(x)
        feat = self.conv_proj(feat)
        feat = self.dropout(feat)
        B, C, H, W = feat.shape
        return feat.permute(0, 2, 3, 1).reshape(B, H * W, C), (H, W)

# ======= CUSTOM DECODER LAYER =======
class CustomDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.5):
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

# ======= OCR RECOGNIZER =======
class OCRRecognizer(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_layers=4, num_heads=4, ff_dim=512, max_len=2048):
        super().__init__()
        self.encoder = CNNEncoder(d_model)
        self.pos_embed_enc = nn.Parameter(torch.randn(1, max_len, d_model))
        self.pos_embed_dec = nn.Parameter(torch.randn(1, max_len, d_model))
        self.token_embed = nn.Embedding(vocab_size, d_model)
        decoder_layer = CustomDecoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=ff_dim, dropout=0.5)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, img, tgt_input, return_attn=False):
        B = img.size(0)
        enc, (H, W) = self.encoder(img)
        S = enc.size(1)
        if S > self.pos_embed_enc.size(1):
            raise ValueError(f"Encoder output S={S} > max_len={self.pos_embed_enc.size(1)}")
        memory = enc + self.pos_embed_enc[:, :S, :]
        memory = memory.permute(1, 0, 2)

        tgt = self.token_embed(tgt_input)
        T = tgt.size(1)
        if T > self.pos_embed_dec.size(1):
            raise ValueError(f"Target sequence T={T} > max_len={self.pos_embed_dec.size(1)}")
        tgt = tgt + self.pos_embed_dec[:, :T, :]
        tgt = tgt.permute(1, 0, 2)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(0)).to(tgt.device)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask)
        out = output.permute(1, 0, 2)
        logits = self.fc_out(out)
        if return_attn:
            attn_map = self.decoder.layers[-1].cross_attn_weights
            if attn_map is not None:
                attn_map = attn_map.mean(dim=1).reshape(tgt_input.size(0), H, W)
            return logits, attn_map
        return logits

# === DATA ===
vocab, inv_vocab = build_vocab(label_file)
dataset = OCRDataset(crop_folder, label_file, vocab, img_size=(256, 64))

# Thống kê dataset sau khi lọc
char_counts = Counter()
for _, text in dataset.data:
    char_counts.update(text)
print("Character frequencies after filtering '#':", char_counts)
print("Total samples after filtering '#':", len(dataset))

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# === MODEL ===
model = OCRRecognizer(vocab_size=len(vocab), d_model=256, max_len=max_len, ff_dim=512, num_layers=4)
model.to(device)

# === OPTIMIZER ===
criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'], label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# Warm-up scheduler
class WarmUpScheduler:
    def __init__(self, optimizer, warmup_epochs, base_lr, final_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.final_lr = final_lr

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr + (self.final_lr - self.base_lr) * epoch / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

warmup_scheduler = WarmUpScheduler(optimizer, warmup_epochs=5, base_lr=1e-5, final_lr=3e-4)

# === METRICS ===
cer_metric = CharErrorRate()
wer_metric = WordErrorRate()

# === TRAIN LOOP ===
train_losses = []
val_losses = []
best_val_loss = float('inf')
early_stop_counter = 0

for epoch in range(1, epochs + 1):
    # Warm-up step
    warmup_scheduler.step(epoch)

    # Train
    model.train()
    total_train_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for images, targets, _ in pbar:
        images, targets = images.to(device), targets.to(device)
        tgt_input = targets[:, :-1]
        tgt_output = targets[:, 1:]
        logits = model(images, tgt_input)
        loss = criterion(logits.view(-1, logits.size(-1)), tgt_output.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
        pbar.set_postfix(loss=loss.item())
    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation
    model.eval()
    total_val_loss = 0
    preds, targets = [], []
    with torch.no_grad():
        for images, tgt, texts in val_loader:
            images, tgt = images.to(device), tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            logits = model(images, tgt_input)
            loss = criterion(logits.view(-1, logits.size(-1)), tgt_output.reshape(-1))
            total_val_loss += loss.item()
            pred_ids = logits.argmax(dim=-1).cpu().numpy()
            pred_texts = [''.join(inv_vocab[id] for id in pred if id != vocab['<pad>'] and id != vocab['<eos>']) for pred in pred_ids]
            preds.extend(pred_texts)
            targets.extend(texts)
    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    # Calculate CER and WER
    cer = cer_metric(preds, targets)
    wer = wer_metric(preds, targets)
    print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, CER = {cer:.4f}, WER = {wer:.4f}")

    # Scheduler step
    scheduler.step(avg_val_loss)

    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        early_stop_counter = 0
        torch.save({
            'model': model.state_dict(),
            'vocab': vocab,
        }, "cr_recognizer2.pt")
        print(f"💾 Saved best model with Val Loss = {best_val_loss:.4f}")
    else:
        early_stop_counter += 1
        print(f"Early stopping counter: {early_stop_counter}/{patience}")
        if early_stop_counter >= patience:
            print(f"Early stopping triggered after {epoch} epochs")
            break

# === PLOT LOSS ===
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.grid(True)
plt.legend()
plt.savefig('loss_curve.png')
print("Saved loss curve to 'loss_curve.png'")
plt.show()

print("Final model saved to cr_recognizer2.pt")