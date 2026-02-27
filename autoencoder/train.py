import os
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from dataset import Autoencoder_dataset
from model import Autoencoder
from torch.utils.tensorboard import SummaryWriter
import argparse


torch.autograd.set_detect_anomaly(True)
SCRIPT_DIR = Path(__file__).resolve().parent
CKPT_ROOT = SCRIPT_DIR / "ckpt"

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def cos_loss(network_output, gt):
    return 1 - F.cosine_similarity(network_output, gt, dim=0).mean()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--encoder_dims',
                    nargs = '+',
                    type=int,
                    default=[256, 128, 64, 32, 3],
                    )
    parser.add_argument('--decoder_dims',
                    nargs = '+',
                    type=int,
                    default=[16, 32, 64, 128, 256, 256, 512],
                    )
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument("--vlm", type=str, default = "CLIP")
    parser.add_argument(
        '--input_dir_name',
        type=str,
        default='',
        help='Input feature folder name under dataset_path. If empty, use vlm default.',
    )
    parser.add_argument(
        '--ckpt_dir_name',
        type=str,
        default='',
        help='Checkpoint subdir name under ckpt/. If empty, use dataset_name.',
    )
    args = parser.parse_args()
    dataset_path = args.dataset_path
    num_epochs = args.num_epochs
    if args.input_dir_name:
        data_dir = os.path.join(dataset_path, args.input_dir_name)
    else:
        if args.vlm == "clip_fine":
            data_dir = f"{dataset_path}/language_features_fine"
        else:
            data_dir = f"{dataset_path}/language_features"

    if args.ckpt_dir_name:
        ckpt_dir_name = args.ckpt_dir_name
    else:
        input_tag = Path(data_dir).name
        ckpt_dir_name = f"{input_tag}/{args.dataset_name}"
    ckpt_dir = str(CKPT_ROOT / ckpt_dir_name)
    print('Loading VLM features from:', data_dir)
    print('Saving checkpoints to:', ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)

    train_dataset = Autoencoder_dataset(data_dir)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=16,
        drop_last=False
    )

    test_loader = DataLoader(
        dataset=train_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=16,
        drop_last=False  
    )
    
    encoder_hidden_dims = args.encoder_dims
    decoder_hidden_dims = args.decoder_dims

    model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims).to("cuda:0")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    logdir = ckpt_dir
    tb_writer = SummaryWriter(logdir)
    print(logdir)

    best_eval_loss = 100.0
    best_epoch = 0
    for epoch in tqdm(range(num_epochs)):
        model.train()
        for idx, feature in enumerate(train_loader):
            data = feature.to("cuda:0")
            outputs_dim3 = model.encode(data)
            outputs = model.decode(outputs_dim3)
            
            l2loss = l2_loss(outputs, data) 
            cosloss = cos_loss(outputs, data)
            loss = l2loss + cosloss * 0.001
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_iter = epoch * len(train_loader) + idx
            tb_writer.add_scalar('train_loss/l2_loss', l2loss.item(), global_iter)
            tb_writer.add_scalar('train_loss/cos_loss', cosloss.item(), global_iter)
            tb_writer.add_scalar('train_loss/total_loss', loss.item(), global_iter)
            tb_writer.add_histogram("feat", outputs, global_iter)

        if epoch > 95:
            eval_loss = 0.0
            model.eval()
            for idx, feature in enumerate(test_loader):
                data = feature.to("cuda:0")
                with torch.no_grad():
                    outputs = model(data) 
                loss = l2_loss(outputs, data) + cos_loss(outputs, data)
                eval_loss += loss * len(feature)
            eval_loss = eval_loss / len(train_dataset)
            print("eval_loss:{:.8f}".format(eval_loss))
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(ckpt_dir, 'best_ckpt.pth'))
                
            if epoch % 10 == 0:
                torch.save(model.state_dict(), os.path.join(ckpt_dir, f'{epoch}_ckpt.pth'))
            
    print(f"best_epoch: {best_epoch}")
    print("best_loss: {:.8f}".format(best_eval_loss))
