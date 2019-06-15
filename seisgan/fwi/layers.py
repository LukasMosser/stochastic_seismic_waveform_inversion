import numpy as np
import torch
import torch.autograd as autograd
import torch.nn.functional as F
from devito import Function, clear_cache

from .pde.seismic.model import Model
from .pde.seismic.acoustic import AcousticWaveSolver
from .pde.seismic import RickerSource, Receiver

from cached_property import cached_property

class TimeAxis(object):
    """ Data object to store the time axis. Exactly three of the four key arguments
        must be prescribed. Because of remainder values it is not possible to create
        a time axis that exactly adhears to the inputs therefore start, stop, step
        and num values should be taken from the TimeAxis object rather than relying
        upon the input values.
        The four possible cases are:
        start is None: start = step*(1 - num) + stop
        step is None: step = (stop - start)/(num - 1)
        num is None: num = ceil((stop - start + step)/step);
                     because of remainder stop = step*(num - 1) + start
        stop is None: stop = step*(num - 1) + start
    :param start:(Optional) Start of time axis.
    :param step: (Optional) Time interval.
    :param: num: (Optional) Number of values (Note: this is the number of intervals + 1).
                 stop value is reset to correct for remainder.
    :param stop: (Optional) End time.
    """
    def __init__(self, start=None, step=None, num=None, stop=None):
        try:
            if start is None:
                start = step*(1 - num) + stop
            elif step is None:
                step = (stop - start)/(num - 1)
            elif num is None:
                num = int(np.ceil((stop - start + step)/step))
                stop = step*(num - 1) + start
            elif stop is None:
                stop = step*(num - 1) + start
            else:
                raise ValueError("Only three of start, step, num and stop may be set")
        except:
            raise ValueError("Three of args start, step, num and stop may be set")

        if not isinstance(num, int):
            raise TypeError("input argument must be of type int")

        self.start = start
        self.stop = stop
        self.step = step
        self.num = num

    def __str__(self):
        return "TimeAxis: start=%g, stop=%g, step=%g, num=%g" % \
               (self.start, self.stop, self.step, self.num)

    def _rebuild(self):
        return TimeAxis(start=self.start, stop=self.stop, num=self.num)

    @cached_property
    def time_values(self):
        return np.linspace(self.start, self.stop, self.num)


class FWIConfiguration(object):
    def __init__(self, config, ground_truth_vp):
        super(FWIConfiguration, self).__init__()
        self.config = config
        self.dtype = np.float32

        self.target = Model(m=ground_truth_vp, origin=self.config["origin"], shape=self.config["shape"],
                            dtype=self.dtype, spacing=self.config["spacing"], nbpml=self.config["nbpml"])


        self.input_m = Model(m=ground_truth_vp, origin=self.config["origin"], shape=self.config["shape"],
                            dtype=self.dtype, spacing=self.config["spacing"],
                            nbpml=self.config["nbpml"])

        self.source_locations = np.empty((self.config["nshots"], 2), dtype=np.float32)
        self.source_locations[:, 1] = self.config["source_min_y"]

        self.source_locations[:, 0] = np.linspace(self.config["source_min_x"],
                                                  self.config["spacing"][0]*self.config["shape"][0] - self.config["source_min_x"],
                                                  num=self.config["nshots"])

        if self.config["nshots"] == 1:
            self.source_locations[:, 0] = np.array([int(self.config["shape"][0]/2.)])
        
        dt = self.target.critical_dt  # Time step from model grid spacing
        self.nt = int(1 + (self.config["tn"]-self.config["t0"]) / dt)  # Discrete time axis length
        self.time = np.linspace(self.config["t0"], self.config["tn"], self.nt)  # Discrete modeling time

        dt = self.target.critical_dt# * (1.73 if kernel == 'OT4' else 1.0)
        t0 = 0.0
        self.time = TimeAxis(start=t0, stop=self.config["tn"], step=dt)

        self.src = self.create_source()
        self.rec = self.create_recorders()

        self.solver = AcousticWaveSolver(self.target, self.src, self.rec, space_order=4)

        self.true_ds = []
        for i in range(self.config["nshots"]):
            # Update source location
            self.src.coordinates.data[0, :] = self.source_locations[i, :]

            # Generate synthetic data from true model
            true_d, _, _ = self.solver.forward(src=self.src, m=self.target.m)
            self.true_ds.append(true_d)

        self.clean_ds = np.zeros((self.time.num, 128))
        for x in self.true_ds:
            self.clean_ds += x.data[:]

        self.noise_norm = 0.
        #Added noise modification to MATG submission
        for i in range(len(self.true_ds)):
            std = self.true_ds[i].data[:].std()
            noise = self.config["noise_percent"]*std*np.random.randn(*self.true_ds[i].data[:].shape)
            self.noise_norm += 0.5*np.linalg.norm(noise)**2
            self.true_ds[i].data[:] += noise

        self.noisy_ds = np.zeros((self.time.num, 128))
        for x in self.true_ds:
            self.noisy_ds += x.data[:]

    def create_source(self):
        #
        #  Define time discretization according to grid spacing
        src = RickerSource(name='src', grid=self.target.grid, f0=self.config["f0"], time_range=self.time)
        src.coordinates.data[0, :] = np.array(self.target.domain_size)
        src.coordinates.data[:, 1] = self.config["source_min_y"]
        src.coordinates.data[0, 0] = self.config["source_min_x"]
        return src

    def create_recorders(self):
        rec = Receiver(name='rec', grid=self.target.grid, npoint=self.config["nreceivers"], time_range=self.time)
        rec.coordinates.data[:, 1] = self.config["rec_min_y"]
        rec.coordinates.data[:, 0] = np.linspace(0, self.config["nreceivers"]*self.config["spacing"][0], num=self.config["nreceivers"])
        return rec


class FWILoss(autograd.Function):
    def __init__(self, configuration):
        super(FWILoss, self).__init__()

        self.config = configuration
        #self.solver = AcousticWaveSolver(self.config.target, self.config.src, self.config.rec, space_order=4)
        self.gradient, self.residual = None, None

        self.residual = Receiver(name='rec', grid=self.config.target.grid,
                                 time_range=self.config.time, coordinates=self.config.rec.coordinates.data)

        self.smooth_ds = np.zeros((self.config.time.num, 128))

    def forward(self, x):
        clear_cache()

        self.config.input_m.vp = x[0, 0].cpu().detach().numpy()

        # Create symbols to hold the gradient and residual
        grad = Function(name="grad", grid=self.config.input_m.grid)

        objective = 0.

        self.smooth_ds = np.zeros((self.config.time.num, 128))
        for i in range(self.config.config["nshots"]):
            clear_cache()
            # Update source location
            self.config.src.coordinates.data[0, :] = self.config.source_locations[i, :]

            # Compute smooth data and full forward wavefield u0
            smooth_d, u0, _ = self.config.solver.forward(src=self.config.src, m=self.config.input_m.m, save=True)

            self.smooth_ds += smooth_d.data[:]
            # Compute gradient from data residual and update objective function
            self.residual.data[:] = smooth_d.data[:] - self.config.true_ds[i].data[:]

            objective += 0.5*np.linalg.norm(self.residual.data.reshape(-1))**2

            self.config.solver.gradient(rec=self.residual, u=u0, m=self.config.input_m.m, grad=grad)

        gradient = torch.from_numpy(grad.data.copy())[self.config.config["nbpml"]:-self.config.config["nbpml"],
                                                      self.config.config["nbpml"]:-self.config.config["nbpml"]]
        self.gradient = gradient
        self.save_for_backward(gradient)

        del grad

        return torch.from_numpy(np.array([objective])).float()

    def backward(self, grad_output):
        gradient = self.saved_tensors[0]
        grad_input = (gradient/gradient.abs().max()).unsqueeze(0).unsqueeze(0)
        return grad_input, None

    def reset(self):
        self.smooth_ds = np.zeros((self.config.nt, 128))


def to_probability(x):
    return x/2.+0.5

def well_loss_old(x_geo_hat, x_geo, well_pos, channel, loss=F.binary_cross_entropy, transform=None):

    wells_hat = x_geo_hat[:, channel, :, well_pos]
    if transform is not None:
        wells_hat = transform(wells_hat)

    wells = x_geo[:, channel, :, well_pos]

    loss_value = loss(wells_hat, wells, reduction="mean")
    return loss_value


def well_loss(x_geo_hat, x_geo, well_pos, channel, loss=F.binary_cross_entropy, transform=None):

    wells_hat = x_geo_hat[:, channel, :, well_pos]
    if transform is not None:
        wells_hat = transform(wells_hat)

    wells = x_geo[:, channel, :, well_pos]

    loss_value = loss(wells_hat, wells, reduction="sum")
    return loss_value

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
