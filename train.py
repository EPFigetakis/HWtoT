import sys 
import os 
import torch 
import torch.nn as nn
import torch.optim as optim 
from tqdm import tqdm 
import editdistance
import wandb
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torch.utils.data import DataLoader 

# Custom Library Imports 
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Python File Imports 
from data import collate_fn
from data import HWLD
from data import create_label
from model import CRNN




def cer(preds, targets):
    total_dist, total_chars = 0, 0
    for p, t in zip(preds, targets):
        dist = editdistance.eval(p, t)
        total_dist += dist
        total_chars += len(t)
    return total_dist / total_chars if total_chars > 0 else 1.0

def decode_predictions(logits, charset, blank=0):
    # Greedy decode, collapse repeats and remove blanks
    probs = logits.softmax(2)
    pred_indices = probs.argmax(2)  # [T, B]
    pred_indices = pred_indices.permute(1, 0)  # [B, T]

    results = []
    for seq in pred_indices:
        prev = blank
        text = ""
        for idx in seq:
            idx = idx.item()
            if idx != blank and idx != prev:
                text += charset[idx]
            prev = idx
        results.append(text)
    return results

def train(model, train_loader, val_loader, charset,num_epochs=20,lr=1e-3,device="cuda" if torch.cuda.is_available() else "cpu",save_path="crnn_best.pth"):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    best_val_cer = float("inf")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for batch in loop:
            images = batch["images"].to(device)  # [B, 1, 32, W]
            targets = batch["label"].to(device)
            label_lengths = batch["label_lengths"].to(device)
            input_lengths = batch["input_lengths"].to(device)
            #label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)

            # Flatten labels into a 1D tensor for CTC
            #targets = torch.cat(labels).to(device)

            logits = model(images)  # [T, B, C]
            #input_lengths = torch.full(size=(logits.size(1),), fill_value=logits.size(0), dtype=torch.long)

            loss = criterion(logits.log_softmax(2), targets, input_lengths, label_lengths)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        # Validation step
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                images = batch["images"].to(device)
                labels = batch["label_strs"]
                logits = model(images)
                decoded = decode_predictions(logits.cpu(), charset)
                all_preds.extend(decoded)
                all_targets.extend(labels)
                if epoch == 0 or (epoch+1) % 5 == 0:
                    for i in range(min(3, len(all_preds))):
                        wandb.log({f"Prediction {i+1}": wandb.Table(data=[[all_preds[i], all_targets[i]]],columns=["Prediction", "Ground Truth"])})

        val_cer = cer(all_preds, all_targets)
        scheduler.step(val_cer)
        avg_train_loss = epoch_loss / len(train_loader)
        wandb.log({"epoch": epoch + 1,"train_loss": avg_train_loss,"val_cer": val_cer,"learning_rate": scheduler.optimizer.param_groups[0]['lr']})
        print(f"Epoch {epoch+1}: Train Loss = {epoch_loss / len(train_loader):.4f}, Val CER = {val_cer:.4f}")

        # Save best model
        if val_cer < best_val_cer:
            best_val_cer = val_cer
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved best model with CER = {val_cer:.4f}")











if __name__ == "__main__":
    wandb.init(project="handwriting-crnn", name="crnn-run-1")
    
    create_label.create_dataset_json(image_dir="data/cropped",label_dir="data/labels",output_file="dataset.json")
    create_label.convert_json_to_csv("dataset.json","dataset.csv")

    dataset = HWLD.HandwritingLineDataset("dataset.csv")
    indices = list(range(len(dataset)))
    train_idx,val_idx = train_test_split(indices,test_size=0.3)
    train_data = Subset(dataset,train_idx)
    val_data = Subset(dataset, val_idx)

    train_loader = DataLoader(train_data,batch_size=4,shuffle=True,collate_fn=collate_fn.ctc_collate_fn)
    val_loader = DataLoader(val_data, batch_size=4, shuffle=True, collate_fn=collate_fn.ctc_collate_fn)

    charset = dataset.get_charset()
    num_classes = len(charset) + 1
    num_epochs = 15
    lr=1e-3
    model = CRNN.CRNN(img_height=32,num_classes=num_classes)
    wandb.config.update({"epochs": num_epochs,"learning_rate": lr,"batch_size": train_loader.batch_size,"optimizer": "Adam","loss": "CTCLoss"})
    wandb.watch(model, log="all", log_freq=10)
    train(model=model,train_loader=train_loader,val_loader=val_loader, lr=lr,charset=charset,num_epochs=num_epochs)
   

