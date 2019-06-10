import sys

#sys.path.append("./devito")

import argparse
import toml
import logging

import numpy as np

from sklearn.metrics import accuracy_score

import torch
from torch.optim import Adam, SGD, lr_scheduler
import torch.nn as nn

from seisgan.fwi import layers
from seisgan.networks import GeneratorMultiChannel, HalfChannels, DiscriminatorUpsampling, HalfChannelsTest
from seisgan.optimizers import MALA, SGHMC
from seisgan.utils import set_seed
import os
import errno
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

def add_seismic_to_writer(tag, writer, seismic, iteration, sum=True):

    if sum:
        noisy_shot_data = torch.from_numpy(np.array([shot.data[:] for shot in seismic])).sum(0).unsqueeze(0)
    else:
        noisy_shot_data = torch.from_numpy(np.array([shot.data[:] for shot in seismic])).unsqueeze(0)

    max_seis = noisy_shot_data.max()
    min_seis = noisy_shot_data.min()
    noisy_shot_data = ((noisy_shot_data - min_seis) / (max_seis - min_seis)) * 255.

    writer.add_image(tag, noisy_shot_data, iteration, dataformats='CHW')

    return True

def add_model_to_writer(tag, writer, model, iteration):
    max_model = model.max()
    min_model = model.min()
    model = ((model - min_model) / (max_model - min_model)) * 255.

    model = torch.from_numpy(model).unsqueeze(0)
    writer.add_image(tag, model, iteration, dataformats='CHW')

    return True

# Writer will output to ./runs/ directory by default

def make_dir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def compute_prior_loss(z, alpha=1.):
    """

    Computes prior loss according to Creswell 2016

    :param z: latent vector
    :param alpha: weight of prior loss
    :return: log probability of the gaussian latent variables
    """
    pdf = torch.distributions.Normal(0, 1)
    logProb = pdf.log_prob(z.view(1, -1)).sum(dim=1)
    prior_loss = -alpha*logProb
    return prior_loss

parser = argparse.ArgumentParser()
parser.add_argument("--working_dir", type=str, default="./", help="Set run_name")
parser.add_argument("--config_file", type=str, default="2_sources_well_local.toml", help="Set config file")
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
model_try = 0
while num_count < config["num_runs"]:
    set_seed(model_try)
    logger.info('Started Optimization: '+str(num_count))
    writer = SummaryWriter(log_dir="./runs2/run_"+str(int(model_try)))
    z_star = torch.randn(1, 50, 1, 2)
    z_star.requires_grad = True

    optimizer = MALA([z_star], lr=config["optimization"]["learning_rate"], weight_decay=0.0)#SGHMC([z_star], lr=config["optimization"]["learning_rate"], nu=0.1)#MALA([z_star], lr=config["optimization"]["learning_rate"], weight_decay=0.0)#SGD([z_star], lr=config["optimization"]["learning_rate"], weight_decay=1e-5) #Adam([z_star], lr=config["optimization"]["learning_rate"])
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

                    acc = accuracy_score(tn(x_gt_facies)[:, 0, :, 64].flatten().astype(int),
                                         np.where(layers.to_probability(tn(x_geo)[:, 0, :, 64]).flatten() > 0.5, 1, 0))
                    print(acc)

                    well_loss.backward(retain_graph=True)
                    if i < 20:
                        nn.utils.clip_grad_norm_(z_star, 10.0)
                    writer.add_scalar("well_acc", acc, global_step=i)
                    writer.add_scalar("well_loss", well_loss, global_step=i)
                    writer.add_scalar("well_grad_norm", z_star.grad.norm(), global_step=i)

        add_seismic_to_writer("seismic_noise", writer, fwi_loss.true_ds, i)

        prior_loss = compute_prior_loss(z_star, 1.0)
        prior_loss.backward(retain_graph=True)
        writer.add_scalar("prior_grad_norm", z_star.grad.norm(), global_step=i)

        writer.add_scalar("prior_loss", prior_loss, global_step=i)
        l = config["optimization"]["lambda_fwi"]*fwi_loss(x_star)
        loss_vars.append(l)

        pred_sum = fwi_loss.smooth_ds

        add_seismic_to_writer("seismic_synth", writer, fwi_loss.smooth_ds, i, sum=False)
        print(x_star.size())
        add_model_to_writer("model", writer, x_geo[0, 0].detach().cpu().numpy(), i)
        rmse_noise = np.sqrt(np.mean((fwi_loss.noisy_ds-fwi_loss.clean_ds)**2))
        rmse_inversion = np.sqrt(np.mean((pred_sum-fwi_loss.clean_ds)**2))

        print(rmse_noise)
        print(rmse_inversion)
        print(rmse_inversion/rmse_noise)

        error = np.linalg.norm(pred_sum-gt_sum)
        rel_error = error/np.linalg.norm(gt_sum)

        print(np.linalg.norm(fwi_loss.noisy_ds-fwi_loss.clean_ds)/np.linalg.norm(fwi_loss.clean_ds))

        losses.append(rel_error)

        total_losses_sum = sum(loss_vars)
        total_losses_sum.backward()

        optimizer.step()

        zs.append(z_star.data.numpy().copy())

        #for param_group in optimizer.param_groups:
        #    param_group['lr'] -= (config["optimization"]["learning_rate"]-config["optimization"]["final_learning_rate"])/config["optimization"]["max_iter"]
        #    lr = param_group['lr']
        for param_group in optimizer.param_groups:
            #param_group['lr'] -= (config["optimization"]["learning_rate"]-config["optimization"]["final_learning_rate"])/config["optimization"]["max_iter"]
            param_group['lr'] = config["optimization"]["learning_rate"]*(1-(i+1)/float(config["optimization"]["max_iter"]))**(0.9)
            lr = param_group['lr']

        print(lr)

        print(z_star.std())

        writer.add_scalar("lr", lr, global_step=i)
        writer.add_scalar("z_std", z_star.std(), global_step=i)
        writer.add_scalar("z_grad_norm", z_star.grad.norm(), global_step=i)
        writer.add_scalar("rel_error", rel_error, global_step=i)

        if z_star.std().data.numpy() > 5.0:
            logger.info(
                'NOT COMPLETED Optimization, Latent Space Diverged')
            latent_diverged = True
            model_try += 1
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
                model_try += 1
                break

            elif rel_error <= config["optimization"]["seismic_relative_error"] and not config["optimization"]["use_well"]:
                logger.info('COMPLETED Optimization: ' + str(num_count) + ' Iteration: ' + str(i) + ' Loss: ' + str(rel_error))

                zs_out = np.array(zs)
                np.save(latent_variables_out_path + str(num_count) + ".npy", zs_out)
                if config["storage"]["store_final_reconstruction_waveform"]:
                    np.save(shots_out_path + str(num_count) + ".npy", pred_sum)
                np.save(losses_out_path + str(num_count) + ".npy", np.array(losses))
                num_count += 1
                model_try += 1
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
            model_try += 1

        elif rel_error <= config["optimization"]["seismic_relative_error"] and not config["optimization"]["use_well"]:
            logger.info(
                'COMPLETED Optimization: ' + str(num_count) + ' Iteration: ' + str(i) + ' Loss: ' + str(rel_error))

            zs_out = np.array(zs)
            np.save(latent_variables_out_path + str(num_count) + ".npy", zs_out)
            if config["storage"]["store_final_reconstruction_waveform"]:
                np.save(shots_out_path + str(num_count) + ".npy", pred_sum)
            np.save(losses_out_path + str(num_count) + ".npy", np.array(losses))
            num_count += 1
            model_try += 1

        else:
            logger.info(
                'NOT COMPLETED Optimization: ' + str(num_count) + ' Iteration: ' + str(i) + ' Loss: ' + str(rel_error))

