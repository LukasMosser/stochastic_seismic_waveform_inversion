import sys

import argparse
import logging
import os

import numpy as np
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from seisgan.fwi import layers
from seisgan.networks import GeneratorMultiChannel, HalfChannels, HalfChannelsTest
from seisgan.optimizers import MALA, SGHMC
from seisgan.utils import set_seed, make_dir, tn, output_losses, output_to_tensorboard


def parse_args(argv):
    # Writer will output to ./runs/ directory by default
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Random Seed")
    parser.add_argument("--run_name", type=str, default="test", help="Set Run Name")
    parser.add_argument("--test_image_id", type=int, default=0, help="Which test image to use as gt")
    parser.add_argument("--num_runs", type=int, default=1, help="How many inversions to perform")

    parser.add_argument("--working_dir", type=str, default="./", help="Set working directory")
    parser.add_argument("--out_folder", type=str, default="./", help="Output folder")
    parser.add_argument("--discriminator_path", type=str, default="./", help="Path to discriminator")
    parser.add_argument("--generator_path", type=str, default="./", help="Path to generator")
    parser.add_argument("--minsmaxs_path", type=str, default="./", help="Path to test set min-max-values")
    parser.add_argument("--testimgs_path", type=str, default="./", help="Path to test set models")

    parser.add_argument("--store_gt_waveform", action="store_true", help="Stores GT Waveform at the beginning of the run")
    parser.add_argument("--store_final_reconstruction_waveform", action="store_true", help="Store the waveform of the final inverted model")

    parser.add_argument("--sources", type=int, default=2, help="How many acoustic sources to use")
    parser.add_argument("--wavelet_frequency", type=float, default=1e-2, help="Wavelet Peak Frequency [hz]")
    parser.add_argument("--simulation_time", type=float, default=1e3, help="Shot recording time")
    parser.add_argument("--top_padding", type=int, default=32, help="Pad GAN domain above")
    parser.add_argument("--bottom_padding", type=int, default=32, help="Pad GAN domain below")
    parser.add_argument("--noise_percent", type=float, default=0.02, help="Percent of noise to add to observed shot data")

    parser.add_argument("--use_well", action="store_true", help="Use well loss")
    parser.add_argument("--well_accuracy", type=float, default=0.95, help="Target Well Accuracy")
    parser.add_argument("--well_position", type=int, default=64, help="Position of the well")

    parser.add_argument("--learning_rate", type=float, default=1e-2, help="Learning Rate")
    parser.add_argument("--final_learning_rate", type=float, default=0.00001, help="Final Learning Rate")
    parser.add_argument("--max_iter", type=int, default=200, help="Maximum number of MALA iterations")
    parser.add_argument("--lambda_perceptual", type=float, default=1.0, help="Weight of perceptual loss")
    parser.add_argument("--lambda_fwi", type=float, default=1.0, help="Weight of fwi loss")
    parser.add_argument("--lambda_well", type=float, default=1.0, help="Weight of well loss")

    parser.add_argument("--tensorboard", action="store_true", help="Use tensorboard")
    parser.add_argument("--print_to_console", action="store_true", help="Stdout to console")
    args = parser.parse_args()

    return args


def main(args):
    set_seed(args.seed)

    working_dir = args.working_dir
    out_folder = args.run_name
    run_name = args.run_name+"_"+str(args.seed)

    monitored_variables = []
    if args.use_well:
        monitored_variables += ["Well-Loss", "Well-Accuracy", "gradient/Well-Loss-Grad-Norm"]
    monitored_variables += ["Pior-Loss", "gradient/Prior-Loss-Grad-Norm", "FWI", "FWI-L1", "FWI-L2", "Shot-Noise-Norm", "gradient/FWI-Grad-Norm", "gradient/Total-Grad-Norm", "Learning-Rate", "Latent-Variable-Std-Dev"]

    log_path = os.path.expandvars(working_dir+"/"+out_folder+"/"+"logs")

    latent_variables_out_path = os.path.expandvars(working_dir+"/"+out_folder+"/"+run_name+"_latents_")
    shots_out_path = os.path.expandvars(working_dir+"/"+out_folder+"/"+run_name+"_shots_")
    losses_out_path = os.path.expandvars(working_dir+"/"+out_folder+"/"+run_name+"_losses_")
    errors_out_path = os.path.expandvars(working_dir+"/"+out_folder+"/"+run_name+"_errors_")
    gt_norms_out_path = os.path.expandvars(working_dir+"/"+out_folder+"/"+run_name+"_gtnorms_")

    make_dir(log_path)
    make_dir(os.path.expandvars(working_dir+"/"+out_folder))

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # create a file handler
    handler = logging.FileHandler(filename=os.path.expandvars(working_dir+"/"+out_folder+"/"+"logs"+"/"+run_name+"_log.log"))
    if args.print_to_console:
        handler = logging.StreamHandler(sys.stdout)

    handler.setLevel(logging.INFO)

    # create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)

    logger.info('Started Logging')

    generator_path = os.path.expandvars(args.generator_path)
    discriminator_path = os.path.expandvars(args.discriminator_path)
    minsmaxs_path = os.path.expandvars(args.minsmaxs_path)
    testimgs_path = os.path.expandvars(args.testimgs_path)

    logger.info('Expanded Variables')

    configuration = {"t0": 0.,
                     "tn": args.simulation_time,
                     "shape": (128, args.bottom_padding+64+args.top_padding),
                     "nbpml": 50,
                     "origin": (0., 0.),
                     "spacing": (10., 1.),
                     "source_min_x": 3,
                     "source_min_y": 10,
                     "nshots": args.sources,
                     "f0": args.wavelet_frequency,
                     "nreceivers": 128,
                     "rec_min_x": 3,
                     "rec_min_y": 10,
                     "noise_percent": args.noise_percent
                     }

    generator = GeneratorMultiChannel()
    new_state_dict = torch.load(generator_path)
    generator.load_state_dict(new_state_dict)
    generator.cpu()
    generator.eval()

    logger.info('Loaded Networks')

    minsmaxs = np.load(minsmaxs_path)

    half_channel_gen = HalfChannels(generator, min_vp=minsmaxs[2, 1], max_vp=minsmaxs[3, 1], vp_bottom=1.8, vp_top=1.8, top_size=args.top_padding)
    half_channel_test = HalfChannelsTest(min_vp=minsmaxs[2, 1], max_vp=minsmaxs[3, 1], vp_bottom=1.8, vp_top=1.8, top_size=args.top_padding)

    gt_image = np.load(testimgs_path)[args.test_image_id]

    x_gt = torch.from_numpy(gt_image).float().unsqueeze(0)
    x_gt_facies = x_gt
    with torch.no_grad():
        x_gt, x_geo_gt = half_channel_test(x_gt)

    logger.info('Loaded GT facies')

    fwi_model_config = layers.FWIConfiguration(configuration, tn(x_gt)[0, 0])
    fwi = layers.FWILoss(fwi_model_config)
    gt_sum = np.sum([x_i.data for x_i in fwi_model_config.true_ds], 0)

    if args.store_gt_waveform:
        np.save(shots_out_path+"gt.npy", gt_sum)
    logger.info('Defined Layers')

    num_count = 0
    lr = args.learning_rate
    model_try = 0
    while num_count < args.num_runs:
        seed = np.random.randint(low=0, high=2**31)
        set_seed(seed)
        logger.info('Started Optimization: '+str(num_count))

        if args.tensorboard:
            writer = SummaryWriter(log_dir=os.path.expandvars(working_dir+"/"+"tensorboard"+"/run_"+str(int(seed))))

        z_star = torch.randn(1, 50, 1, 2)
        z_star.requires_grad = True

        # Initialize MALA sampler
        optimizer = MALA([z_star], lr=args.learning_rate, weight_decay=0.0)
        losses, shots = [], []
        zs = [z_star.detach().numpy().copy()]

        pred_sum, latent_diverged = None, False

        acc = 0.0
        for i in range(0, args.max_iter):
            # Reset Gradients
            optimizer.zero_grad()

            # Forward Pass Latent Space to get model representation
            x_star, x_geo = half_channel_gen(z_star)

            monitor_losses = []
            current_grad_norm = 0.

            # If optimizing the wells, compute binary cross-entropy at well location
            if args.use_well:
                for channel, lambd, loss_type, transform, name in zip([0], [args.lambda_well],
                                                                      [torch.nn.functional.binary_cross_entropy],
                                                                      [layers.to_probability],
                                                                      ["Facies"]):
                    for well in [args.well_position]:
                        well_loss = layers.well_loss(x_geo, x_gt_facies.float(), well, channel, loss=loss_type, transform=transform)

                        acc = accuracy_score(tn(x_gt_facies)[:, 0, :, args.well_position].flatten().astype(int),
                                             np.where(layers.to_probability(tn(x_geo)[:, 0, :, args.well_position]).flatten() > 0.5, 1, 0))

                        well_loss.backward(retain_graph=True)  # keep graph active so we can continue to backprop
                        monitor_losses.append(tn(well_loss))
                        monitor_losses.append(acc)

                        well_loss_grad_norm = tn(z_star.grad.norm())
                        monitor_losses.append(well_loss_grad_norm)
                        current_grad_norm += well_loss_grad_norm # Store temporary grad norm so we can see which variables are contributing to loss

            # Compute Prior loss based on Gaussian latent space prior
            prior_loss = layers.compute_prior_loss(z_star, 1.0)
            prior_loss.backward(retain_graph=True) # Kep graph active so we can continue to backprop
            prior_loss_grad_norm = tn(z_star.grad.norm())-current_grad_norm
            monitor_losses.extend([tn(prior_loss), prior_loss_grad_norm])
            current_grad_norm += prior_loss_grad_norm

            # Compute FWI-Loss
            fwi = layers.FWILoss(fwi_model_config)
            fwi_loss = args.lambda_fwi*fwi(x_star)
            pred_sum = fwi.smooth_ds
            l1_norm = np.linalg.norm(gt_sum-pred_sum, 1)
            l2_norm = np.linalg.norm(pred_sum-gt_sum)
            monitor_losses.extend([fwi_loss.item(), l1_norm, l2_norm, fwi_model_config.noise_norm])

            fwi_loss.backward()
            fwi_grad_norm = tn(z_star.grad.norm())-current_grad_norm
            monitor_losses.append(fwi_grad_norm)
            current_grad_norm += fwi_grad_norm
            monitor_losses.append(current_grad_norm)

            # Step MALA forward one iteration
            optimizer.step()

            # Anneal Step Size
            for param_group in optimizer.param_groups:
                param_group['lr'] -= (args.learning_rate-args.final_learning_rate)/args.max_iter
                lr = param_group['lr']
            monitor_losses.append(lr)

            # Monitor the latent space after the MALA step
            latent_space_standard_dev = z_star.std().data.numpy()
            zs.append(z_star.data.numpy().copy())
            monitor_losses.append(latent_space_standard_dev)

            # Output to logger or tensorboard
            losses.append(monitor_losses)
            output_losses(logger, monitor_losses, monitored_variables, num_count, i)
            if args.tensorboard:
                output_to_tensorboard(writer, monitor_losses, monitored_variables, i)

            # Check whether the latent space has diverged if yes, reset
            if latent_space_standard_dev > 5.0:
                logger.info('NOT COMPLETED Optimization, Latent Space Diverged')
                output_losses(logger, monitor_losses, monitored_variables, num_count, i)
                latent_diverged = True
                model_try += 1
                break

        if ((not args.use_well) or (args.use_well and acc >= args.well_accuracy)) and not latent_diverged:
            logger.info('COMPLETED Optimization')
            output_losses(logger, monitor_losses, monitored_variables, num_count, i)

            zs_out = np.array(zs)
            np.save(latent_variables_out_path + str(num_count) + ".npy", zs_out)
            if args.store_final_reconstruction_waveform:
                np.save(shots_out_path + str(num_count) + ".npy", pred_sum)
            np.save(losses_out_path + str(num_count) + ".npy", np.array(losses))
            num_count += 1
        else:
            logger.info('NOT COMPLETED Optimization')
            output_losses(logger, monitor_losses, monitored_variables, num_count, i)
        model_try += 1
        fwi.reset()

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)

