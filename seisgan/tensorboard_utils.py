import torch
import numpy as np

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