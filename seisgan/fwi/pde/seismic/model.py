import numpy as np

from devito import Grid, Function, Constant


def damp_boundary(damp, nbpml, spacing):
    """Initialise damping field with an absorbing PML layer.

    :param damp: Array data defining the damping field
    :param nbpml: Number of points in the damping layer
    :param spacing: Grid spacing coefficent
    """
    dampcoeff = 1.5 * np.log(1.0 / 0.001) / (40.)
    ndim = len(damp.shape)
    for i in range(nbpml):
        pos = np.abs((nbpml - i + 1) / float(nbpml))
        val = dampcoeff * (pos - np.sin(2*np.pi*pos)/(2*np.pi))
        if ndim == 2:
            damp[i, :] += val/spacing[0]
            damp[-(i + 1), :] += val/spacing[0]
            damp[:, i] += val/spacing[1]
            damp[:, -(i + 1)] += val/spacing[1]
        else:
            damp[i, :, :] += val/spacing[0]
            damp[-(i + 1), :, :] += val/spacing[0]
            damp[:, i, :] += val/spacing[1]
            damp[:, -(i + 1), :] += val/spacing[1]
            damp[:, :, i] += val/spacing[2]
            damp[:, :, -(i + 1)] += val/spacing[2]


class Model(object):
    """The physical model used in seismic inversion processes.

    :param origin: Origin of the model in m as a tuple in (x,y,z) order
    :param spacing: Grid size in m as a Tuple in (x,y,z) order
    :param shape: Number of grid points size in (x,y,z) order
    :param m: Square Slowness in s**2/km**2
    :param nbpml: The number of PML layers for boundary damping
    :param rho: Density in kg/cm^3 (rho=1 for water)
    :param epsilon: Thomsen epsilon parameter (0<epsilon<1)
    :param delta: Thomsen delta parameter (0<delta<1), delta<epsilon
    :param theta: Tilt angle in radian
    :param phi: Asymuth angle in radian

    The :class:`Model` provides two symbolic data objects for the
    creation of seismic wave propagation operators:

    :param m: The square slowness of the wave
    :param damp: The damping field for absorbing boundarycondition
    """
    def __init__(self, origin, spacing, shape, m, nbpml=20, dtype=np.float32,
                 epsilon=None, delta=None, theta=None, phi=None):
        self.shape = shape
        self.nbpml = int(nbpml)
        self.origin = origin

        shape_pml = np.array(shape) + 2 * self.nbpml
        # Physical extent is calculated per cell, so shape - 1
        extent = tuple(np.array(spacing) * (shape_pml - 1))
        self.grid = Grid(extent=extent, shape=shape_pml,
                         origin=origin, dtype=dtype)

        # Create square slowness of the wave as symbol `m`
        if isinstance(m, np.ndarray):
            self.m = Function(name="m", grid=self.grid)

        # Set model velocity, which will also set `m`
        self.vp = m #We assign vp here with the square slowness which internally computes the true vp
        #We need to do this because the adjoint will compute gradients with respect to square slowness
        #Which is then backpropped through our network in the last layer where vp gets turned into m

        # Create dampening field as symbol `damp`
        self.damp = Function(name="damp", grid=self.grid)
        damp_boundary(self.damp.data, self.nbpml, spacing=self.spacing)

        # Additional parameter fields for TTI operators
        self.scale = 1.

        if epsilon is not None:
            if isinstance(epsilon, np.ndarray):
                self.epsilon = Function(name="epsilon", grid=self.grid)
                self.epsilon.data[:] = self.pad(1 + 2 * epsilon)
                # Maximum velocity is scale*max(vp) if epsilon > 0
                if np.max(self.epsilon.data) > 0:
                    self.scale = np.sqrt(np.max(self.epsilon.data))
            else:
                self.epsilon = 1 + 2 * epsilon
                self.scale = epsilon
        else:
            self.epsilon = 1

        if delta is not None:
            if isinstance(delta, np.ndarray):
                self.delta = Function(name="delta", grid=self.grid)
                self.delta.data[:] = self.pad(np.sqrt(1 + 2 * delta))
            else:
                self.delta = delta
        else:
            self.delta = 1

        if theta is not None:
            if isinstance(theta, np.ndarray):
                self.theta = Function(name="theta", grid=self.grid)
                self.theta.data[:] = self.pad(theta)
            else:
                self.theta = theta
        else:
            self.theta = 0

        if phi is not None:
            if isinstance(phi, np.ndarray):
                self.phi = Function(name="phi", grid=self.grid)
                self.phi.data[:] = self.pad(phi)
            else:
                self.phi = phi
        else:
            self.phi = 0

    @property
    def dim(self):
        """
        Spatial dimension of the problem and model domain.
        """
        return self.grid.dim

    @property
    def spacing(self):
        """
        Grid spacing for all fields in the physical model.
        """
        return self.grid.spacing

    @property
    def spacing_map(self):
        """
        Map between spacing symbols and their values for each :class:`SpaceDimension`
        """
        return self.grid.spacing_map

    @property
    def dtype(self):
        """
        Data type for all assocaited data objects.
        """
        return self.grid.dtype

    @property
    def shape_domain(self):
        """Computational shape of the model domain, with PML layers"""
        return tuple(d + 2*self.nbpml for d in self.shape)

    @property
    def domain_size(self):
        """
        Physical size of the domain as determined by shape and spacing
        """
        return tuple((d-1) * s for d, s in zip(self.shape, self.spacing))

    @property
    def critical_dt(self):
        """Critical computational time step value from the CFL condition."""
        # For a fixed time order this number goes down as the space order increases.
        #
        # The CFL condtion is then given by
        # dt <= coeff * h / (max(velocity))
        coeff = 0.38 if len(self.shape) == 3 else 0.42
        return coeff * np.min(self.spacing) / (self.scale*np.max(self.vp))

    @property
    def vp(self):
        """:class:`numpy.ndarray` holding the model velocity in km/s.

        .. note::

        Updating the velocity field also updates the square slowness
        ``self.m``. However, only ``self.m`` should be used in seismic
        operators, since it is of type :class:`Function`.
        """
        return self._vp

    @vp.setter
    def vp(self, vp):
        """Set a new velocity model and update square slowness

        :param vp : new velocity in km/s
        """
        self._vp = 1./np.sqrt(vp) #convert square slowness to vp m=1./vp**2 -> 1./sqrt(m)

        # Update the square slowness according to new value
        if isinstance(vp, np.ndarray):
            self.m.data[:] = self.pad(vp)

    def pad(self, data):
        """Padding function PNL layers in every direction for for the
        absorbing boundary conditions.

        :param data : Data array to be padded"""
        pad_list = [(self.nbpml, self.nbpml) for _ in self.shape]
        return np.pad(data, pad_list, 'edge')
