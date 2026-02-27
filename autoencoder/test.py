import os
from pathlib import Path
import numpy as np
import torch
import argparse
import shutil
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import Autoencoder_dataset
from model import Autoencoder

SCRIPT_DIR = Path(__file__).resolve().parent
CKPT_ROOT = SCRIPT_DIR / "ckpt"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, required=True)
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
    parser.add_argument("--vlm", type=str, default = "CLIP")
    parser.add_argument(
        '--input_dir_name',
        type=str,
        default='',
        help='Input feature folder name under dataset_path. If empty, use vlm default.',
    )
    parser.add_argument(
        '--output_dir_name',
        type=str,
        default='',
        help='Output dim3 folder name under dataset_path. If empty, use vlm default.',
    )
    parser.add_argument(
        '--ckpt_dir_name',
        type=str,
        default='',
        help='Checkpoint subdir name under ckpt/. If empty, use dataset_name.',
    )
    args = parser.parse_args()
    
    dataset_name = args.dataset_name
    encoder_hidden_dims = args.encoder_dims
    decoder_hidden_dims = args.decoder_dims
    dataset_path = args.dataset_path
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
        ckpt_dir_name = f"{input_tag}/{dataset_name}"
    ckpt_path = str(CKPT_ROOT / ckpt_dir_name / "best_ckpt.pth")

    if args.output_dir_name:
        output_dir = os.path.join(dataset_path, args.output_dir_name)
    else:
        if args.vlm == "clip_fine":
            output_dir = f"{dataset_path}/language_features_fine_dim3"
        else:
            output_dir = f"{dataset_path}/language_features_dim3"
        
    print('Dataset Name:', dataset_name)
    print('Features Path:', data_dir)
    print('Output Path:', output_dir)
    print('Use Checkpoint:', ckpt_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # copy the segmentation map
    for filename in os.listdir(data_dir):
        if filename.endswith("_s.npy"):
            source_path = os.path.join(data_dir, filename)
            target_path = os.path.join(output_dir, filename)
            shutil.copy(source_path, target_path)

    checkpoint = torch.load(ckpt_path)
    train_dataset = Autoencoder_dataset(data_dir)

    test_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=256,
        shuffle=False, 
        num_workers=16, 
        drop_last=False   
    )

    model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims).to("cuda:0")

    model.load_state_dict(checkpoint)
    model.eval()

    for idx, feature in tqdm(enumerate(test_loader)):
        data = feature.to("cuda:0")
        with torch.no_grad():
            outputs = model.encode(data).to("cpu").numpy()  
        if idx == 0:
            features = outputs
        else:
            features = np.concatenate([features, outputs], axis=0)

    os.makedirs(output_dir, exist_ok=True)
    start = 0
    
    for k,v in train_dataset.data_dic.items():
        path = os.path.join(output_dir, k)
        np.save(path, features[start:start+v])
        start += v
