# Copyright (C) 2018  Carl-Johan Haster, Michael Pürrer, Charlie Hoy
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""This module provides model classes that assume the noise is Gaussian,
   and the use of a Reduced Order Quadrature (ROQ) likelihood
"""

import numpy
import logging
from abc import (ABCMeta, abstractmethod)

from pycbc import transforms, distributions
from pycbc.waveform import generator, NoWaveformError
from pycbc.detector import Detector

from .base import BaseModel, SamplingTransforms


class GaussianNoiseROQ(BaseModel):
    r"""
    write helpful help here!
    """
    name = 'gaussian_noise_roq'

    def __init__(self, variable_params, waveform_generator,
                 roq_weights_linear, roq_weights_quadratic,
                 roq_freq_linear, roq_freq_quadratic,
                 roq_time_steps, lognl,
                 **kwargs):


        self._waveform_generator = waveform_generator
        self._roq_weights_linear = {ifo: rwl.copy() for (ifo, rwl) in roq_weights_linear.items()}
        # NEED TO BE A SPLINE
        self._roq_weights_quadratic = {ifo: rwq.copy() for (ifo, rwq) in roq_weights_quadratic.items()}
        # NEED TO BE A SPLINE
        self._roq_freqs_linear = {ifo: rfl.copy() for (ifo, rfl) in roq_freq_linear.items()}
        self._roq_freqs_quadratic = {ifo: rfq.copy() for (ifo, rfq) in roq_freq_quadratic.items()}
        self._roq_time_steps = {ifo: rts.copy() for (ifo, rts) in roq_time_steps.items()}
        self._lognl = {ifo: nl.copy() for (ifo, nl) in lognl.items()}

        # figure out the length of the linear and quadratic bases, diff for each IFO
        self._len_linear = 12
        self._len_quadratic = 12
        # figure out the number of timesteps? also how much time it's covering?
        self_time_steps = 12

        self.detectors = {ifo: Detector(ifo) for ifo in lognl.items()}


        # check that the linear weights and waveform generator have the same detectors
        if (sorted(self._waveform_generator.detectors.keys()) !=
                sorted(self._roq_weights_linear.keys())):
            raise ValueError(
                "waveform generator's detectors ({0}) does not "
                "match the linear weights ({1})".format(
                    ','.join(sorted(self._waveform_generator.detector_names)),
                    ','.join(sorted(self._roq_weights_linear.keys()))))
        # check that the quadratic weights and waveform generator have the same detectors
        if (sorted(self._waveform_generator.detectors.keys()) !=
                sorted(self._roq_weights_quadratic.keys())):
            raise ValueError(
                "waveform generator's detectors ({0}) does not "
                "match the quadratic weights ({1})".format(
                    ','.join(sorted(self._waveform_generator.detector_names)),
                    ','.join(sorted(self._roq_weights_quadratic.keys()))))
        # check that the linear freqs and waveform generator have the same detectors
        if (sorted(self._waveform_generator.detectors.keys()) !=
                sorted(self._roq_freqs_linear.keys())):
            raise ValueError(
                "waveform generator's detectors ({0}) does not "
                "match the linear freqs ({1})".format(
                    ','.join(sorted(self._waveform_generator.detector_names)),
                    ','.join(sorted(self._roq_freqs_linear.keys()))))
        # check that the quadratic freqs and waveform generator have the same detectors
        if (sorted(self._waveform_generator.detectors.keys()) !=
                sorted(self._roq_freqs_quadratic.keys())):
            raise ValueError(
                "waveform generator's detectors ({0}) does not "
                "match the quadratic freqs ({1})".format(
                    ','.join(sorted(self._waveform_generator.detector_names)),
                    ','.join(sorted(self._roq_freqs_quadratic.keys()))))
        # check that the quadratic freqs and waveform generator have the same detectors
        if (sorted(self._waveform_generator.detectors.keys()) !=
                sorted(self._lognl.keys())):
            raise ValueError(
                "waveform generator's detectors ({0}) does not "
                "match the logNoise likelihood ({1})".format(
                    ','.join(sorted(self._waveform_generator.detector_names)),
                    ','.join(sorted(self._lognl.keys()))))

    @property
    def waveform_generator(self):
        """Returns the waveform generator that was set."""
        return self._waveform_generator

    @property
    def default_stats(self):
        """The stats that ``get_current_stats`` returns by default."""
        return ['logjacobian', 'logprior', 'loglr'] + \
               ['{}_cplx_loglr'.format(det) for det in self._lognl] + \
               ['{}_optimal_snrsq'.format(det) for det in self._lognl]
    @property
    def lognl(self):
        """The log likelihood of the model assuming the data is noise.

        This will initially try to return the ``current_stats.lognl``.
        If that raises an ``AttributeError``, will call `_lognl`` to
        calculate it and store it to ``current_stats``.
        """
        return self._trytoget('lognl', self._lognl)

    @property
    def loglr(self):
        """The log likelihood ratio at the current parameters.

        This will initially try to return the ``current_stats.loglr``.
        If that raises an ``AttributeError``, will call `_loglr`` to
        calculate it and store it to ``current_stats``.
        """
        return self._trytoget('loglr', self._loglr)

    def _nowaveform_loglr(self):
        """Convenience function to set loglr values if no waveform generated.
        """
        for det in self._lognl:
            setattr(self._current_stats, '{}_cplx_loglr'.format(det),
                    -numpy.inf)
            # snr can't be < 0 by definition, so return 0
            setattr(self._current_stats, '{}_optimal_snrsq'.format(det), 0.)
        return -numpy.inf

    def _loglr(self):
        r"""Computes the log likelihood ratio (for the ROQ likelihood),

        .. math::

            \log \mathcal{L}(\Theta) =
            \sum_i w^l_i*h_i(\Theta) -
            \sum_j w^q_j*h_j(\Theta)

        at the current parameter values :math:`\Theta` for the linear
        and quadratic ROQ weights

        Returns
        -------
        float
            The value of the log likelihood ratio.
        """
        params = self.current_params
        try:
            wfs_l_plus, wfs_l_cross = self._waveform_generator.generate(self._roq_freqs_linear, **params)
            wfs_q_plus, wfs_q_cross = self._waveform_generator.generate(self._roq_freqs_quadratic, **params)
        except NoWaveformError:
            return self._nowaveform_loglr()
        lr = 0.

        timeshift = self._waveform_generator.epoch - self.current_params['tc']

        for det in self._lognl:

            timedelay = self.detectors[det].time_delay_from_earth_center(self.current_params['ra'],
                         self.current_params['dec'], self.current_params['tc'])

            hd = 0.+0j
            for i in xrange(self._len_linear):
                hd += self._roq_weights_linear[det][i](timeshift + timedelay)*(wfs_l_plus[i] + wfs_l_cross[i]).conj()

            hh = 0.
            for j in xrange(self._len_quadratic):
                hh += self._roq_weights_quadratic[det][j](timeshift + timedelay)*((wfs_q_plus[j] + wfs_q_cross[j])*\
                                                                                 (wfs_q_plus[j] + wfs_q_cross[j]).conj()).real

            weight_l = self._roq_weights_linear[det](timeshift + timedelay)

            #timeshift =  (epoch - (*(REAL8 *) LALInferenceGetVariable(model->params, "time"))) + timedelay;


    @classmethod
    def _init_args_from_config(cls, cp):
        """Helper function for loading parameters."""
        section = "model"
        prior_section = "prior"
        vparams_section = 'variable_params'
        sparams_section = 'static_params'
        constraint_section = 'constraint'
        # check that the name exists and matches
        name = cp.get(section, 'name')
        if name != cls.name:
            raise ValueError("section's {} name does not match mine {}".format(
                             name, cls.name))
        # get model parameters
        variable_params, static_params = distributions.read_params_from_config(
            cp, prior_section=prior_section, vargs_section=vparams_section,
            sargs_section=sparams_section)
        # get prior
        prior = cls.prior_from_config(cp, variable_params, prior_section,
                                      constraint_section)
        args = {'variable_params': variable_params,
                'static_params': static_params,
                'prior': prior}
        # get any other keyword arguments provided
        args.update(cls.extra_args_from_config(cp, section,
                                               skip_args=['name']))
        return args

    @classmethod
    def roq_from_config(cls, cp, section):
        roq_args = [opt for opt in cp.options(section)
                     if opt.startswith('roq')]



    @classmethod
    def from_config(cls, cp, **kwargs):
        """Initializes an instance of this class from the given config file.

        Parameters
        ----------
        cp : WorkflowConfigParser
            Config file parser to read.
        \**kwargs :
            All additional keyword arguments are passed to the class. Any
            provided keyword will over ride what is in the config file.
        """
        args = cls._init_args_from_config(cp)
        # try to load sampling transforms
        try:
            sampling_transforms = SamplingTransforms.from_config(
                cp, args['variable_params'])
        except ValueError:
            sampling_transforms = None
        args['sampling_transforms'] = sampling_transforms
        args.update(kwargs)
        return cls(**args)