import torch
import wandb
import argparse
import random
import datetime
from torch.utils.data import Subset
import numpy as np
import itertools
import sys
from AttenUnet.AttenUnet_4M import Att_Unet
sys.path.append('../')
from CTDataset import StrokeAI



import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from CTDataset import StrokeAI
from model.pytorch3dunet.unet3d.model import UNet3D, ResidualUNet3D
from model.pytorch3dunet.unet3d.losses import DiceLoss
#from model.SwinUnet3D.SwinUnet_3DV2 import SwinUnet3D
import os
import json

from monai.inferers import sliding_window_inference

sys.path.append('./Diff_UNet/')
sys.path.append('./Diff_UNet/BraTS2020')
#from BraTS2020.train import DiffUNet

torch.manual_seed(42)

# sometimes, the world_size is not equal to the real size of data
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def reduce_loss(loss, rank, world_size):
    """
    Reduce the loss across all processes so that every process has the average loss.

    Args:
    loss (float): The loss to reduce.
    rank (int): The rank of the current process in the distributed training setup.
    world_size (int): The total number of processes in the distributed training setup.

    Returns:
    float: The reduced loss.
    """
    reduced_loss = torch.tensor(loss, device="cuda:{}".format(rank))
    dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
    reduced_loss = reduced_loss / world_size
    return reduced_loss.item()


def main(rank, world_size):

    setup(rank, world_size)
    args = parse_args()
    print(args)
    args_dict = vars(args)

    # Training Parameters
    gradient_accumulation_steps = args.gradient_accumulation_steps
    batch_size= args.batch_size
    learning_rate = args.learning_rate
    epochs = args.epochs
    current_time = datetime.datetime.now()
    timestamp = current_time.strftime("%Y%m%d_%H%M")

    # Dataset Parameters
    # construct train dataset and test dataset
    train_dataset =  StrokeAI(CT_root="/scratch4/rsteven1/AISD/image",
                       DWI_root="/scratch4/rsteven1/DWI_coregis_20231208",  #DWI
                       ADC_root="/scratch4/rsteven1/ADC_coregis_20231228",  # ADC
                       label_root="/scratch4/rsteven1/AISD/mask", 
                       MRI_type = 'ADC',
                       mode = 'train',
                       map_file= "/home/agoyal19/3D-Lesion-Segmentation/GenrativeMethod/efficient_ct_dir_name_to_XNATSessionID_mapping.json",
                       bounding_box=args.bounding_box,
                       instance_normalize=args.instance_normalize, 
                       padding=args.padding, 
                       slicing=args.slicing,
                       crop=args.crop,
                       random_crop_ratio=args.random_crop_ratio,
                       RotatingResize = args.RotatingResize)

    test_dataset = StrokeAI(CT_root="/scratch4/rsteven1/AISD/image",
                       DWI_root="/scratch4/rsteven1/DWI_coregis_20231208",  #DWI
                       ADC_root="/scratch4/rsteven1/ADC_coregis_20231228",  # ADC
                       label_root="/scratch4/rsteven1/AISD/mask", 
                       MRI_type = 'ADC',
                       mode = 'test',
                       map_file= "/home/agoyal19/3D-Lesion-Segmentation/GenrativeMethod/efficient_ct_dir_name_to_XNATSessionID_mapping.json",
                       bounding_box=args.bounding_box,
                       instance_normalize=args.instance_normalize, 
                       padding=args.padding, 
                       slicing=args.slicing,
                       crop=args.crop,
                       RotatingResize = args.RotatingResize)

    print(f"Size of train_dataset: {len(train_dataset)}")
    print(f"Size of test_dataset: {len(test_dataset)}") 


    # train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_seed=42, indices_file='../dataset_indices.json', args=args)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank) 
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=train_sampler, num_workers=3)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, sampler=test_sampler, num_workers=3)

    # Model initialization
    if args.model_name == '3D_Unet':
        model = UNet3D(in_channels=1, out_channels=1, final_sigmoid=True).to(rank)
    elif args.model_name == 'Att_Unet':
        model = Att_Unet().to(rank)
    elif args.model_name == 'Swin_Unet':
        model = SwinUnet3D(hidden_dim=96, layers=(2, 2, 6, 2), heads=(3, 6, 9, 12), num_classes=1, window_size=3).to(rank)
    elif args.model_name == 'Diff_Unet':
        model = DiffUNet()
    else:
        assert False

    num_parameters = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_parameters}")

    model = DDP(model.to(rank), device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    if args.resuming:
        checkpoint = torch.load(args.checkpoint_path, map_location=lambda storage, loc: storage.cuda(rank))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Load saved model and optimizer")


    # loss and optimizer selection
    loss = DiceLoss(normalization='none')
    best_test_loss = 100
    # adversarial_loss =torch.nn.CrossEntropyLoss()

    if rank == 0 and args.wandb:
        wandb.init(
            # set the wandb project where this run will be logged
            project="StrokeAI",

            name=f'{timestamp}_{args.model_name}',
            
            # track hyperparameters and run metadata
            config = args_dict
        )

    print("start training")
    for epoch in range(epochs):
    
        if rank == 0 and args.wandb:
            print(f"The current epoch is {epoch}")
            wandb.log({"epoch": epoch})


        # Training loop
        model.train()
        total_train_loss = 0
        train_samples = 0
        # optimizer.zero_grad()

        for batch_idx, sample in enumerate(train_loader):

            # image = sample['ct'].to(rank)
            # label = sample['label'].to(rank)
            # x_start = label
            # x_start = (x_start) * 2 -1
            # x_t, t, noise = model(x=x_start, pred_type="q_sample")
            # pred_xstart = model(x=x_t, step=t, image=image, pred_type="denoise")
            # pred = torch.sigmoid(pred_xstart)

            # regular training
            pred = model(sample['ct'].to(rank))
            label = sample['label'].to(rank)
            pred = model(sample['ct'].to(rank))
            label = sample['label'].to(rank)

            loss_ = loss(pred, label) / gradient_accumulation_steps
            loss_.backward()
            total_train_loss += loss_.item() * sample['ct'].size(0) * gradient_accumulation_steps
            train_samples += sample['ct'].size(0)

            if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

        avg_train_loss = reduce_loss(total_train_loss / train_samples, rank, world_size)
        if rank == 0 and args.wandb:
            print(f"Average Training Loss for Epoch {epoch}: {avg_train_loss}")
            wandb.log({"epoch": epoch, "average_train_loss": avg_train_loss})

        # Testing loop
        model.eval()
        dice_scores = []
        with torch.no_grad():
            for batch_idx, sample in enumerate(test_loader):

                if args.crop:
                    pred = sliding_window_inference(sample['ct'].to(rank), [56, 56, 56], args.batch_size, model)
                else:
                    pred = model(sample['ct'].to(rank))

                pred = pred > (pred.max() + pred.min()) / 2
                label = sample['label'].to(rank)

                # Compute Dice score manually for logging
                intersection = (pred & label).sum(dim=(1,2,3,4)).float()
                union = pred.sum(dim=(1,2,3,4)) + label.sum(dim=(1,2,3,4)).float()
                dice_batch = (2. * intersection) / (union + 1e-6)

                dice_scores.extend(dice_batch.cpu().numpy().tolist())

        # Calculate mean and std
        mean_dice = np.mean(dice_scores)
        std_dice = np.std(dice_scores)

        if rank == 0:
            print(f"Dice Score: {mean_dice:.4f} ± {std_dice:.4f}")
            if args.wandb:
                wandb.log({"dice_mean": mean_dice, "dice_std": std_dice, "epoch": epoch})

    cleanup()

if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    print(f"There are {world_size} CUDA device")
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)


def parse_args():
    parser = argparse.ArgumentParser(description="StrokeAI Training Script")

    # Model Parameters
    ### You have to give a name of the model name
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model to use')

    # log info
    parser.add_argument('--wandb', action='store_true', help='log data to wandb')

    # Dataset Parameters
    parser.add_argument('--CT_root', type=str, default='/scratch4/rsteven1/AISD/image', help='Root directory for CT images')
    parser.add_argument('--DWI_root', type=str, default='/scratch4/rsteven1/DWI_coregis_20231208', help='Root directory for DWI images')
    parser.add_argument('--ADC_root', type=str, default='/scratch4/rsteven1/ADC_coregis_20231228', help='Root directory for ADC images')
    parser.add_argument('--label_root', type=str, default='/scratch4/rsteven1/AISD/mask', help='Root directory for label images')
    parser.add_argument('--MRI_type', type=str, default='ADC', choices=['ADC', 'DWI', 'Other'], help='Type of MRI images')
    parser.add_argument('--map_file', type=str, default= "/home/agoyal19/3D-Lesion-Segmentation/GenrativeMethod/efficient_ct_dir_name_to_XNATSessionID_mapping.json", help='Path to the map file')
    parser.add_argument('--bounding_box', action='store_true', help='Whether to use bounding box')
    parser.add_argument('--padding', action='store_true', help='Whether to use padding')
    parser.add_argument('--slicing', action='store_true', help='Whether to use slicing')
    parser.add_argument('--instance_normalize', action='store_true', help='Whether to use instance normalization')
    parser.add_argument('--crop', action='store_true', help='Crop size, if any(need to upgrade)')
    parser.add_argument('--random_crop_ratio', type=float, default=0.0, help='the ratio of random crop compared to lesion crop')
    parser.add_argument('--RotatingResize', action='store_true', help='Whether to use rotating resize')


    # Training Parameters
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate for the optimizer')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training and testing')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Number of gradient accumulation steps')


    # Add more arguments as needed
    parser.add_argument('--resuming', action='store_true', help='Continue trianing from previous checkpoint')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path for model checkpoint')



    return parser.parse_args()