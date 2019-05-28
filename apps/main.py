import sys

#sys.path.append("./devito")

import argparse
import toml
import logging

import numpy as np

from sklearn.metrics import accuracy_score

import torch
from torch.autograd import Variable
from torch.optim import Adam, SGD, lr_scheduler
import torch.nn as nn

from seisgan.fwi import layers
from seisgan.networks import GeneratorMultiChannel, HalfChannels, DiscriminatorUpsampling, HalfChannelsTest

import os
import errno


def make_dir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

parser = argparse.ArgumentParser()
parser.add_argument("--working_dir", type=str, default="./", help="Set run_name")
parser.add_argument("--config_file", type=str, default="3_sources_1_local.toml", help="Set config file")
parser.add_argument("--config_folder", type=str, default="config", help="Set config file")
parser.add_argument("--run_name", type=str, default="test_1", help="Set Run Name")
args = parser.parse_args()

config_path = os.path.expandvars(args.working_dir+args.config_folder+"/"+args.config_file)

config = toml.load(config_path)
working_dir = args.working_dir
out_folder = args.run_name
run_name = args.run_name

loss_names = ["Disc", "Facies", "Vp", "FWI"]
if config["optimization"]["use_well"]:
    loss_names = ["Disc", "Facies", "FWI"]
elif not config["optimization"]["use_well"]:
    loss_names = ["Disc", "FWI"]

log_path = os.path.expandvars(args.working_dir+"logs")

latent_variables_out_path = os.path.expandvars(working_dir+out_folder+"/"+run_name+"_latents_")
shots_out_path = os.path.expandvars(working_dir+out_folder+"/"+run_name+"_shots_")
losses_out_path = os.path.expandvars(working_dir+out_folder+"/"+run_name+"_losses_")

make_dir(log_path)
make_dir(os.path.expandvars(working_dir+out_folder))

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# create a file handler
handler = logging.FileHandler(filename=os.path.expandvars(working_dir+"logs"+"/"+run_name+"_log.log"))
if config["debugging"]["print_to_console"]:
    handler = logging.StreamHandler(sys.stdout)

handler.setLevel(logging.INFO)

# create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(handler)

logger.info('Started Logging')

generator_path = os.path.expandvars(working_dir+"checkpoints/generator_facies_multichannel_4_6790.pth")
discriminator_path = os.path.expandvars(working_dir+"checkpoints/discriminator_facies_multichannel_4_6790.pth")
minsmaxs_path = os.path.expandvars("../icml_2018/synthetics/half_circle_channels/half_circle_facies_vp_rho_mean_std_min_max.npy")
testimgs_path = os.path.expandvars("../icml_2018/synthetics/half_circle_channels/test_half_circle_facies_vp_rho.npy")

logger.info('Expanded Variables')

tn = lambda n: n.data.cpu().numpy()

configuration = {"t0": 0.,
                 "tn": config["simulation"]["simulation_time"],
                 "shape": (128, config["simulation"]["bottom_padding"]+64+config["simulation"]["top_padding"]),
                 "nbpml": 50,
                 "origin": (0., 0.),
                 "spacing": (10., 1.),
                 "source_min_x": 3,
                 "source_min_y": 10,
                 "nshots": config["simulation"]["sources"],
                 "f0": config["simulation"]["wavelet_frequency"],
                 "nreceivers": 128,
                 "rec_min_x": 3,
                 "rec_min_y": 10
                 }

generator = GeneratorMultiChannel()
new_state_dict = torch.load(generator_path)
generator.load_state_dict(new_state_dict)
generator.cpu()
generator.eval()

discriminator = DiscriminatorUpsampling()
new_state_dict = torch.load(discriminator_path)
discriminator.load_state_dict(new_state_dict)
discriminator.cpu()
discriminator.eval()

logger.info('Loaded Networks')

minsmaxs = np.load(minsmaxs_path)

half_channel_gen = HalfChannels(generator, min_vp=minsmaxs[2, 1], max_vp=minsmaxs[3, 1], vp_bottom=1.8, vp_top=1.8, top_size=config["simulation"]["top_padding"])
half_channel_test = HalfChannelsTest(min_vp=minsmaxs[2, 1], max_vp=minsmaxs[3, 1], vp_bottom=1.8, vp_top=1.8, top_size=config["simulation"]["top_padding"])

gt_image = np.load(testimgs_path)[config["test_image_id"]]

x_gt = torch.from_numpy(gt_image).float().unsqueeze(0)
x_gt_facies = x_gt
with torch.no_grad():
    x_gt, x_geo_gt = half_channel_test(x_gt)

logger.info('Loaded GT facies')

fwi_model_config = layers.FWIConfiguration(configuration, tn(x_gt)[0, 0])
fwi_loss = layers.FWILoss(fwi_model_config)
gt_sum = np.sum([x_i.data for x_i in fwi_loss.true_ds], 0)
if config["storage"]["store_gt_waveform"]:
    np.save(shots_out_path+"gt.npy", gt_sum)
logger.info('Defined Layers')

num_count = 0
lr = config["optimization"]["learning_rate"]
while num_count < config["num_runs"]:
    logger.info('Started Optimization: '+str(num_count))

    z_star = torch.randn(1, 50, 1, 2)
    z_star.requires_grad=True

    optimizer = SGD([z_star], lr=config["optimization"]["learning_rate"], weight_decay=1e-5)
    losses = []
    zs = []
    shots = []
    losses_total = []
    pred_sum = None
    latent_diverged = False
    acc = 0.0
    for i in range(0, config["optimization"]["max_iter"]):
        loss_vars = []
        optimizer.zero_grad()
        x_star, x_geo = half_channel_gen(z_star)

        if config["optimization"]["use_disc"]:
            d = -config["optimization"]["lambda_perceptual"]*discriminator(x_geo).mean()
            loss_vars.append(d)

        if config["optimization"]["use_well"]:
            for channel, lambd, loss_type, transform, name in zip([0], [config["optimization"]["lambda_well"]],
                                                                  [torch.nn.functional.binary_cross_entropy],
                                                                  [layers.to_probability],
                                                                  ["Facies"]):
                for well in [64]:
                    well_loss = lambd * layers.well_loss(x_geo, x_gt_facies.float(), well, channel,
                                                         loss=loss_type, transform=transform)
                    nn.utils.clip_grad_norm_(z_star, 5.0)
                    acc = accuracy_score(tn(x_gt_facies)[:, 0, :, 64].flatten().astype(int),
                                         np.where(layers.to_probability(tn(x_geo)[:, 0, :, 64]).flatten() > 0.5, 1, 0))
                    print(acc)
                    loss_vars.append(well_loss)

        l = config["optimization"]["lambda_fwi"]*fwi_loss(x_star)
        loss_vars.append(l)

        pred_sum = fwi_loss.smooth_ds
        error = np.linalg.norm(pred_sum-gt_sum)
        rel_error = error/np.linalg.norm(gt_sum)
        losses.append(rel_error)

        total_losses_sum = sum(loss_vars)
        total_losses_sum.backward()

        optimizer.step()

        z_star.data += 2*lr*torch.randn(1, 50, 1, 2)
        zs.append(z_star.data.numpy().copy())

        for param_group in optimizer.param_groups:
            param_group['lr'] -= (config["optimization"]["learning_rate"]-config["optimization"]["final_learning_rate"])/config["optimization"]["max_iter"]
            lr = param_group['lr']
        print(z_star.std())
        if z_star.std().data.numpy() > 5.0:
            logger.info(
                'NOT COMPLETED Optimization, Latent Space Diverged')
            latent_diverged = True
            break

        for ls, name in zip(loss_vars, loss_names):
            logger.info('Optimization: ' + str(num_count) + ' Iteration: ' + str(i) + " " +name+': ' + str(tn(ls)))
        logger.info('Optimization: ' + str(num_count) + ' Iteration: ' + str(i) + " " + "Relative Error" + ': ' + str(rel_error))
        if config["optimization"]["use_well"]:
            logger.info('Optimization: ' + str(num_count) + ' Iteration: ' + str(i) + ' Accuracy: ' + str(acc))

        if not config["optimization"]["error_termination"]:
            if rel_error <= config["optimization"]["seismic_relative_error"] and (config["optimization"]["use_well"] and acc >= config["optimization"]["well_accuracy"]):
                logger.info('COMPLETED Optimization: ' + str(num_count) + ' Iteration: ' + str(i) + ' Loss: ' + str(rel_error))

                zs_out = np.array(zs)
                np.save(latent_variables_out_path+str(num_count)+".npy", zs_out)
                if config["storage"]["store_final_reconstruction_waveform"]:
                    np.save(shots_out_path + str(num_count) + ".npy", pred_sum)
                np.save(losses_out_path+str(num_count)+".npy", np.array(losses))
                num_count += 1
                break

            elif rel_error <= config["optimization"]["seismic_relative_error"] and not config["optimization"]["use_well"]:
                logger.info('COMPLETED Optimization: ' + str(num_count) + ' Iteration: ' + str(i) + ' Loss: ' + str(rel_error))

                zs_out = np.array(zs)
                np.save(latent_variables_out_path + str(num_count) + ".npy", zs_out)
                if config["storage"]["store_final_reconstruction_waveform"]:
                    np.save(shots_out_path + str(num_count) + ".npy", pred_sum)
                np.save(losses_out_path + str(num_count) + ".npy", np.array(losses))
                num_count += 1
                break
        fwi_loss.reset()
    print(acc)
    if config["optimization"]["error_termination"] and not latent_diverged:
        if rel_error <= config["optimization"]["seismic_relative_error"] and (config["optimization"]["use_well"] and acc >= config["optimization"]["well_accuracy"]):
            logger.info(
                'COMPLETED Optimization: ' + str(num_count) + ' Iteration: ' + str(i) + ' Loss: ' + str(rel_error))

            zs_out = np.array(zs)
            np.save(latent_variables_out_path + str(num_count) + ".npy", zs_out)
            if config["storage"]["store_final_reconstruction_waveform"]:
                np.save(shots_out_path + str(num_count) + ".npy", pred_sum)
            np.save(losses_out_path + str(num_count) + ".npy", np.array(losses))
            num_count += 1

        elif rel_error <= config["optimization"]["seismic_relative_error"] and not config["optimization"]["use_well"]:
            logger.info(
                'COMPLETED Optimization: ' + str(num_count) + ' Iteration: ' + str(i) + ' Loss: ' + str(rel_error))

            zs_out = np.array(zs)
            np.save(latent_variables_out_path + str(num_count) + ".npy", zs_out)
            if config["storage"]["store_final_reconstruction_waveform"]:
                np.save(shots_out_path + str(num_count) + ".npy", pred_sum)
            np.save(losses_out_path + str(num_count) + ".npy", np.array(losses))
            num_count += 1

        else:
            logger.info(
                'NOT COMPLETED Optimization: ' + str(num_count) + ' Iteration: ' + str(i) + ' Loss: ' + str(rel_error))