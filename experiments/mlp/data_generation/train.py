import os, shutil, sys

int_arg = None
if len(sys.argv) > 1:
    try:
        int_arg = int(sys.argv[1])
    except:
        int_arg = None
        
run_dir = None
if len(sys.argv) > 2:
    try:
        run_dir = str(sys.argv[2])
    except:
        run_dir = None

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
import logging
import warnings
import h5py
import uuid
from pathlib import Path
from datetime import datetime

from trainer import MLPWrapper
from datamodule import ExpressionDataModule

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=pl.utilities.warnings.PossibleUserWarning)

if run_dir is None:
    run_dir = 'run-' + str(uuid.uuid4())
    retry_count = 0
    while os.path.exists(f'./{run_dir}/') and retry_count < 5:
        run_dir = uuid.uuid4()
        retry_count += 1
    if retry_count == 5:
        print("somehow failed to choose run dir")
        exit(1)
    elif retry_count > 0:
        print("WARNING: had to retry for unique run dir. concerning.")

if not os.path.exists(f'./{run_dir}/'):
    os.mkdir(f'./{run_dir}/')
os.chdir(f'./{run_dir}/')

# expression_list = []
# parameter_list = []
# loss_list = []

expression_file = "./expressions.csv"
parameter_file = "./parameters.h5"

if not os.path.exists(expression_file):
    with open(expression_file, 'a') as f:
        f.write(',expr,index,best_mse_loss,best_scaled_mse_loss\n')

num_nns = int_arg or 300 # goal for this run
nns_saved = 0 # so far
num_trained = 0 # so far

print(f"Training {num_nns} models...")

success_rate = 1.0
emergency_stop = False

while nns_saved < num_nns and not emergency_stop:
    os.mkdir(f'./lightning_logs/')

    model = MLPWrapper(
        input_dim=2,
        hidden_dims=[16,],
        output_dim=1,
        activation="relu",
        bias=True,
    )

    TINY_EXPR = ["i0", "i1", "-i0", "-i1", "(i0 + i1)", "(i0 - i1)", "(i1 - i0)", "-(i0 + i1)"]
    datamodule = ExpressionDataModule(
        input_dim=2,
        output_dim=1,
        batch_size=256,
        dataset_size=1024,
        split_ratios=[0.625, 0.25, 0.125],
        expr_list=[TINY_EXPR[np.random.choice(8)]],
        reloadable=True,
        # sampler_kwargs={
        #     "depth_decay": 0.3, 
        #     "var_decay": 0.0,
        #     "const_prob": 0.0, 
        #     "modifier_prob": 0.0,
        #     "leaf_prob": 0.05,
        #     "operators": ["+", "-", "*"],
        #     # "modifiers": [],
        #     "operator_probs": [1./3, 1./3, 1./3],
        #     # "modifier_probs": [],
        #     "const_rounding": 0,
        # }
    )

    print(str(datetime.now()) + f': Model {num_trained}')
    print(datamodule.expr_list)

    trainer = pl.Trainer(
        default_root_dir=f"./",
        enable_checkpointing=True,
        max_epochs=3000,
        reload_dataloaders_every_n_epochs=10,
        logger=[pl.loggers.CSVLogger(save_dir=f"./")],
        callbacks=[
            pl.callbacks.EarlyStopping(monitor="valid/loss", patience=500, mode="min", stopping_threshold=1e-4),
            pl.callbacks.ModelCheckpoint(save_last=False, save_top_k=1, monitor="valid/loss", mode="min")
        ],
        enable_progress_bar=False,
        enable_model_summary=False,
        log_every_n_steps=3,
        gpus=int(torch.cuda.is_available()),
    )
    trainer.fit(model=model, datamodule=datamodule)

    log_df = pd.read_csv(f'./lightning_logs/version_0/metrics.csv')
    best_loss = log_df['valid/loss'].min()

    scaling = torch.var(datamodule.val.tensors[1])
    best_perc_error = (best_loss / scaling).item()

    if best_perc_error > 2.5e-2:
        print(f'Last epoch: {model.current_epoch}')
        print(f'Failed on {datamodule.expr_list}. Best loss: {best_loss}, best loss percent: {best_perc_error}')

        success_rate = success_rate * 0.95

    else:
        file_names = os.listdir(f'./lightning_logs/version_0/checkpoints/')
        if len(file_names) != 1:
            raise ValueError(f"{file_names}")

        best_epoch = file_names[0].split('-')[0].split('=')[1]
        print(f'Best epoch: {best_epoch}, last epoch: {model.current_epoch}')

        best_model = MLPWrapper.load_from_checkpoint(
            f'./lightning_logs/version_0/checkpoints/{file_names[0]}',
            input_dim=2,
            hidden_dims=[16,],
            output_dim=1,
            activation="relu",
            bias=True,
        )
        parameters = torch.nn.utils.parameters_to_vector(best_model.parameters())
        parameters = parameters.cpu().detach().numpy()

        # expression_list.append(datamodule.expr_list)
        # parameter_list.append(parameters)
        # loss_list.append((best_loss, best_perc_error))
        with h5py.File(parameter_file, 'a') as h5f:
            if 'counter' not in h5f.keys():
                h5f.create_dataset('counter', data=np.array([0], dtype=int))
            if 'nn_parameters' not in h5f.keys():
                h5f.create_dataset('nn_parameters', (50, parameters.size), maxshape=(None, parameters.size))
            curr_counter = h5f['counter'][()].item()
            if h5f['nn_parameters'].shape[0] < curr_counter + 1:
                h5f['nn_parameters'].resize((h5f['nn_parameters'].shape[0] + 50, parameters.size))
            h5f['nn_parameters'][curr_counter, :] = parameters
            h5f['counter'][:] = curr_counter + 1

        with open(expression_file, 'a') as f:
            f.write(','.join([str(curr_counter)] + datamodule.expr_list + [str(curr_counter), str(best_loss), str(best_perc_error)]) + '\n')

        nns_saved += 1
        print(f'Success on {datamodule.expr_list}. Best loss: {best_loss}, best loss percent: {best_perc_error}')

        success_rate = success_rate * 0.95 + 0.05

    num_trained += 1
    print(f'Smoothed success rate: {success_rate}')
    if success_rate < 0.2:
        emergency_stop = True

    print('')

    shutil.rmtree(f"./lightning_logs/")


