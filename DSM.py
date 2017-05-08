#! /usr/bin/env python2
# -*- coding: utf-8 -*-
# tested on:    Python 2.7.12 (default, Nov 19 2016, 06:48:10) 
#               [GCC 5.4.0 20160609] on linux2
"""
Created on Thu Apr 27 15:55:26 2017

@author: Michele Dei (michele.dei@imb-cnm.csic.es)

DSM module
"""

# IMPORTS
import numpy as np
import matplotlib.pyplot as plt

# PatternPSD class
class PatternPSD():
    """
    Recognize signal versus noise and distortion patterns in power spectral
    densities (PSDs)
    """
    def __init__(self, sxx, 
                 ftone=None, 
                 fsamp=1.0,  
                 bandwidth=None, 
                 leakage=4, **kwargs):
        
        if not type(sxx) == np.ndarray:
            try:
                sxx = np.array(sxx)
            except:
                return
        
        self.fsamp = fsamp             # sampling frequency
        if ftone:
            self.ftone = ftone         # input tone frequency
        else:
            # auto detect input tone (as strongest)
            self.ftone = 0.5*fsamp*sxx.argmax()/float(len(sxx)-1)
        assert(ftone <= 0.5*fsamp)
        
        # bandwidth parameter handling
        if bandwidth is None:
            self.bandwidth = [0, 0.5*fsamp] # bandwidth
        elif isinstance(bandwidth, (list, tuple, np.ndarray)):
            self.bandwidth = bandwidth
        else:
            assert(isinstance(bandwidth, (int, float)))
            self.bandwidth = [0, bandwidth]
        assert(len(self.bandwidth)==2)
        self.bandwidth[1] = min(self.bandwidth[1], 0.5*fsamp)   
                
        # leakage handling
        self.leakage = leakage         # spectral leakage, number of bins
        
        #try: # DEBUG
        if 1:
            # each method __calculate_X will append the X attribute 
            self.__calculate_pattern(sxx)
            self.__calculate_SNDR(sxx)
            self.__calculate_SDR(sxx)
            self.__calculate_SNR(sxx)
            self.__calculate_SFDR(sxx)
        #except:
        #    return None # DEBUG
    
    def out(self, what={'SNDR':True, 'SFDR':True, 'SNR':True, 'SDR':True},):
        s = ''
        for key in what:
            if hasattr(self, key):
                s = s+key+'={0:.1f}dB; '.format(getattr(self, key))
        return s
    
    def __calculate_pattern(self, x):
        
        # pattern of psd:
        #             0 = in-band noise
        #             1 = signal, mutiples: harmonics
        #            -1 = out of band noise
    
        bl, bu = self.bandwidth
        fn = 0.5*self.fsamp
        ft = self.ftone
        lk = self.leakage
        
        #try: # DEBUG
        if 1:
            l = len(x)                 # number of samples
            i_bl = int((l-1)/fn*bl)    # lower interval bandwidth index
            i_bu = int((l-1)/fn*bu)    # upper interval bandwidth index
            i_in = int(l*ft/fn)        # input tone index              
            if i_in > 0:
                nh = max(int(i_bu/i_in),1) # number of harmonics
            else:
                nh = 0
                print('PatternPSD.__calculate_pattern(): i_in = 0') # DEBUG
            ptt = -1*np.ones(l, dtype=np.int8)
            ptt[i_bl:i_bu].fill(0)
            for i in range(1, nh+1):
                il, iu = max(0, i*i_in-lk), min(i*i_in+lk+1, l)
                ptt[il:iu].fill(i)
            
            self.pattern = ptt
            self._index_in = i_in
            self._index_bw = np.array([i_bl, i_bu], np.int8)
            self._harmonics_number = nh
            return ptt
        #except:
        #    print('__calculate_pattern@PatternPSD() failed') # DEBUG
        #    return
    
    def __calculate_SNDR(self, x):
        # SNDR:       signal-to-noise-and-distortion ratio [dB]
        sum1 = np.sum(x[np.where(self.pattern == 1)])
        sum0 = np.sum(x[np.logical_or(self.pattern == 0, self.pattern > 1)])    
        self.SNDR = 10.0*np.log10(sum1/sum0)
        return self.SNDR
    
    def __calculate_SDR(self, x):
        # SDR:       signal-to-distortion ratio [dB]
        sum1 = np.sum(x[np.where(self.pattern == 1)])
        sum2 = np.sum(x[np.where(self.pattern > 1)])
        self.SDR = 10.0*np.log10(sum1/sum2)
        return self.SDR          
    
    def __calculate_SNR(self, x):
        # SNR:       signal-to-noise ratio [dB]
        sum1 = np.sum(x[np.where(self.pattern == 1)])
        sum2 = np.sum(x[np.where(self.pattern == 0)])
        self.SNR = 10.0*np.log10(sum1/sum2)
        return self.SNR
    
    def __calculate_SFDR(self, x):
        # SFDR:      spurious-free-dynamic-range [dB]
        l = len(x)
        lk = self.leakage
        ix = min(l-1, max(0, self._index_in))
        xx = np.copy(x)
        xx1 = x[ix]
        xx = np.where(self.pattern<0, 0, xx)
        xx[max(0, ix-lk):min(ix+lk+1, l)].fill(0)
        self._index_spurious = xx.argmax()
        self.SFDR = 10.0*np.log10(xx1/max(xx))

        return self.SFDR

# PatternPSDSweep()
class PatternPSDSweep():
    """
    Based on PatternPSD class, returns the S**R sweeping the OSR
    """
    def __init__(self, sxx, 
                 ftone=None, 
                 fsamp=1.0,
                 leakage=4, 
                 bw_low=0,
                 df_steps=1,
                 silent=True):
        p = PatternPSD(sxx=sxx, ftone=ftone, fsamp=fsamp, leakage=leakage)
        self.ftone = p.ftone
        df = fsamp/float(len(sxx))
        self.freqs = np.arange(bw_low+df, 0.5*fsamp, df*df_steps)
        self.OSR = (0.5*fsamp)/self.freqs
        self.SNDR = np.zeros_like(self.freqs)
        self.SFDR = np.zeros_like(self.freqs)
        self.SDR = np.zeros_like(self.freqs)
        self.SNR = np.zeros_like(self.freqs)
        for i, f in enumerate(self.freqs):
            if not(silent):
                print(' * PatternPSDSweep, processing {0:d} of {1:d}'.format(i, len(self.freqs)))
            p = PatternPSD(sxx=sxx, ftone=ftone, fsamp=fsamp, leakage=leakage,
                           bandwidth=[bw_low, f])
            self.SNDR[i] = p.SNDR
            self.SFDR[i] = p.SFDR
            self.SDR[i] = p.SDR
            self.SNR[i] = p.SNR

# ColorPSD class
class ColorPSD():
    """
    Provides an easy way to color customization for the PlotPSD class    
    """
    
    def __init__(self, 
        rgb_signal       = ((0.5, 0.6, 0.2), (0.2, 0.6, 0.5)),
        rgb_harmonics    = ((0.7, 0.0, 0.2), (0.7, 0.3, 0.2)),
        rgb_noise_in     = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        rgb_noise_out    = ((0.8, 0.8, 0.8), (0.8, 0.8, 0.8)),
        rgb_spurious     = ((0.6, 0.1, 0.1), (0.6, 0.1, 0.1))   ):
        
        self.rgb_signal     = rgb_signal
        self.rgb_harmonics  = rgb_harmonics
        self.rgb_noise_in   = rgb_noise_in
        self.rgb_noise_out  = rgb_noise_out
        self.rgb_spurious   = rgb_spurious
        
        self.i = 1
        self.n = 1
    
    def __mix(self, i, n, attr):
        a = getattr(self, attr)
        mix = lambda c: ((n-i)*a[0][c]+i*a[1][c])/float(n)
        return tuple(map(mix, range(3)))

    def signal(self):
        return self.__mix(self.i, self.n, 'rgb_signal')

    def harmonics(self):
        return self.__mix(self.i, self.n, 'rgb_harmonics')

    def noise_in(self):
        return self.__mix(self.i, self.n, 'rgb_noise_in')

    def noise_out(self):
        return self.__mix(self.i, self.n, 'rgb_noise_out')

    def spurious(self):
        return self.__mix(self.i, self.n, 'rgb_spurious')
        
    def set_i_over_n(self, i, n):
        self.i = i
        self.n = n

# PlotPSD class
class PlotPSD():
    """
    Provides an obect based on matplotlib.subplots for visualization of power 
    spectral densities (PSDs). It returns the respective .fig and .ax method
    for further graphical manipulation.
    It relies on the PatternPSD class for signal versus noise and distortion 
    patterns and on the ColorPSD class for color customization
    It also defines internally the following graphical classes:
        - SlopeLegend: useful for sigma-delta or general colored spectra
        - RatioLegend: prints on plot the SNDR, SNR, SDR, SFDR from PatternPSD
        - AutoLabels: default x-y labels  
    All this internal classes provide a .show method
    """
    def __init__(self, sxx, fsamp=1.0,  
        pattern=None, color=None, 
        dB_at_fin=None, marker='v', figsize=(6,6), fig=None, ax=None, **kwargs):
        # sxx is intended to be passed as linear absolute measure
        # Pattern and Color must be instances of PatternPSD and ColorPSD, resp.
        # (28-04-2017): added figsize, fig, ax
        if not isinstance(pattern, PatternPSD):           
            pattern = PatternPSD(sxx=sxx, fsamp=fsamp, **kwargs)
        self.pattern = pattern # DEBUGGED! INDENTATION WRONG
        p = pattern.pattern
        i_in = pattern._index_in
        i_sp = pattern._index_spurious
        
        if not isinstance(color, ColorPSD):
            color = ColorPSD()

        frq = np.linspace(0, 0.5*fsamp, len(sxx))   
    
        dBn = 0
        if not(dB_at_fin is None):
            dBn = dB_at_fin - 10.0*np.log10(sxx[i_in])

        dB = lambda x: 10*np.log10(x) + dBn
        
        self.fig, self.ax = plt.subplots(figsize=figsize)
        if not(fig==None) and not(ax==None):
            self.fig = fig
            self.ax = ax
        # piece wise analysis and plot following the pattern
        changes = np.where(np.diff(p)!=0)[0]  
        # where return tuples of np.array. 
        begin_index = np.hstack( (0,changes) )
        # element 0 is prepended
        end_index = np.hstack( (changes, len(p)-1) )
        # last element is postpended
        color_dic = {-1 : color.noise_out(),
                      0 : color.noise_in(), 
                      1 : color.signal()}
        for b,e in zip(begin_index, end_index):
            if p[e] in color_dic.keys():
                c = color_dic[p[e]]    
            else:
                c = color.harmonics()
            if e < len(p)-1:
                e += 1
            # HOT-FIX DEBUG:
            # the following check as been assed as hot-fix: 
            # would return a traceback if [b,e] = [0,1]. Why?
            if len(frq[b:e])>1:
                self.ax.semilogx(frq[b:e], dB(sxx[b:e]), color=c)
        self.ax.semilogx(frq[i_in], dB(sxx[i_in]), marker, color=color.signal())
        self.ax.semilogx(frq[i_sp], dB(sxx[i_sp]), marker, color=color.spurious())
        self.slope_legend = self.SlopeLegend(self.ax, color=color.noise_in())
        self.ratio_legend = self.RatioLegend(self.ax, self.pattern, color=color.noise_in())
        self.labels = self.AutoLabels(self.ax)
    
    def decorate(self, **kwargs):
        self.ax.grid(True)
        self.slope_legend.show()
        self.ratio_legend.show()
        self.labels.show()
    
    class SlopeLegend():
        def __init__(self, ax, slope_dB_dec=None, linewidth=2, color=(0,0,0), fontsize=10):
            # slope is given as dB/dec
            self.ax = ax
            self.linewidth = linewidth
            self.color = color
            self.fontsize = fontsize
            self.slope_dB_dec = slope_dB_dec
                
        def show(self, location='sw', ax=None):            
            if not(self.slope_dB_dec==None):
                dp = {'linewidth': self.linewidth, 'color': self.color}
                dt = {'fontsize': self.fontsize, 'color': self.color,
                      'horizontalalignment': 'right', 
                      'verticalalignment': 'top'}
                txt = str(self.slope_dB_dec) + ' dB/dec'
                
                if ax==None:
                    ax = self.ax.axis()
                if location == 'sw':
                    x = ax[1]*np.array([0.1, 0.5, 0.5, 0.1])
                    y = ax[2]*np.ones(4)+np.array([0,0,0.5*self.slope_dB_dec,0])+10
                    tx, ty = x[1], y[0]-1
                # TO DO: implement more locations and negative slopes   
                try:
                    self.ax.plot(x[0:2], y[0:2], **dp)
                    self.ax.plot(x[1:3], y[1:3], **dp)
                    self.ax.plot(x[2:4], y[2:4], **dp)
                    self.ax.text(tx, ty, txt, **dt)
                except:
                    return
        
    class RatioLegend():
        def __init__(self, ax, pattern, linewidth=2, color=(0,0,0), fontsize=10):
            self.ax = ax
            self.pattern = pattern
            self.linewidth = linewidth
            self.color = color
            self.fontsize = fontsize
        
        def show(self, ratios=('SNDR','SNR','SDR','SFDR'), location='ne', ax=None):
            dt = {'fontsize': self.fontsize, 'color': self.color,
                  'horizontalalignment': 'left', 
                  'verticalalignment': 'top'}
            try:
                txt = ''
                for r in ratios:
                    if hasattr(self.pattern, r):
                        txt += r + '=%.2f'%getattr(self.pattern, r)+'dB\n'
            except:
               return
            
            if ax == None:
                ax = self.ax.axis()

            if location == 'ne':
                x = ax[0]*1.2
                y = ax[3]-3
            # TO DO: implement more locations
            
            try:
                self.ax.text(x, y, txt, **dt)
            except:
                return
                
    class AutoLabels():
        def __init__(self, ax):
            self.ax = ax
                
        def show(self, xlabel='Frequency (Hz)', 
            ylabel='Power Spectral Density (dB)', title='', **kwargs):
            self.ax.set_xlabel(xlabel, **kwargs)
            self.ax.set_ylabel(ylabel, **kwargs)
            if not(title==''):
                self.ax.set_title(title, **kwargs)

################### TOPOLOGIES FOR DELTA-SIGMA MODULATORS #####################

# StateVariable class
class StateVariable(object):
    """
    Class for common operations on internal state variables of the modulators.
    Allows clamping to given bounds. Common basic operation are supported:
    +,-,* operators are defined
    This class should be extended as needed.
    Also provides noise to be included in state variables defined as arrays,
    implementing an auto-noise (normal, zero mean) by the __getitem__ method
    and the _sigma attribute.
    """
    def __init__(self, x, bounds, sigma=0.0):
        self._bounds = bounds
        self._value = self.clamp(x)
        self._sigma = sigma
        
    # Supported types: built-in numbers, np.array
    def clamp(self, x):
        if type(x) == np.ndarray:
            return np.array([min(max(e, self._bounds[0]), self._bounds[1]) for e in x])
        else:
            return min(max(x, self._bounds[0]), self._bounds[1])
    
    # operators: +, -, *
    def __add__(self, x):
        return StateVariable(self.clamp(self._value + x), self._bounds)
    
    def __radd__(self, x):
        return StateVariable(self.clamp(self._value + x), self._bounds)
    
    def __sub__(self, x):
        return StateVariable(self.clamp(self._value - x), self._bounds)
    
    def __rsub__(self, x):
        return StateVariable(self.clamp(-self._value + x), self._bounds)
    
    def __mul__(self, x):
        return StateVariable(self.clamp(self._value * x), self._bounds)
        
    def __rmul__(self, x):
        return StateVariable(self.clamp(self._value * x), self._bounds)        

    # indexing for np.array container
    def __setitem__(self, k, x):
        self._value[k] = self.clamp(x)
        if self._sigma > 0:
             self._value[k] += self._sigma*np.random.normal(0, 1)
    
    def __getitem__(self, k):
        return self._value[k]

    # representation        
    def __str__(self):
        return str(self._value)
        
    def __repr__(self):
        return self.__str__()

# DeltaSigmaModulator class: parent of: DeltaSigmaModulator_2_, _3_, _4_
class DeltaSigmaModulator(object):
    """
    General class for Delta-Sigma-Modulator objects.
    Provided methods: __init__, thresholds, quantize, init_state.
    init_state method uses the StateVariable class.
    more_attributes should be a dictionary to assign attributes to object at 
    init
    """    
    def __init__(self, topology='ff', n_dac_levels=2, clamping=False, 
        internal_range=(-0.5, 0.5), more_attributes=None, **kwds):
        
        self.topology = topology
        self.internal_range = internal_range
        self.n_dac_levels = n_dac_levels
        self.thresholds(n_dac_levels)
        self.clamping = clamping
        if more_attributes:
            for key in more_attributes:
                try:
                    assert isinstance(key, str)
                    setattr(self, key, more_attributes[key])
                except AssertionError:
                    pass

    def thresholds(self, n_dac_levels):
        fullscalepp = self.internal_range[1]-self.internal_range[0]
        # dac levels:
        self.dac_levels = np.array(map(
            lambda x: fullscalepp/float(n_dac_levels-1)*x + self.internal_range[0],
            range(0, n_dac_levels)))
        # quantizer thresholds:
        self.quantizer_thresholds = np.array(map(
            lambda x: 0.5*(self.dac_levels[x-1] + self.dac_levels[x]), 
            range(1, n_dac_levels)))

    def quantize(self, x):
        # TO DO: multiquantizer structures? A dedicated quantizer class?
        idx = 0
        stop = False
        while not(stop):
            if x > self.quantizer_thresholds[idx]:
                idx += 1
            else:
                stop = True
            if idx > self.n_dac_levels - 2:
                stop = True        
        return idx, self.dac_levels[idx]
        
    def limit(self, x):
        return max(min(x, self.internal_range[1]), self.internal_range[0])
        
    #def init_state(self, s, sigma=0.0):
        # s: state variable. sigma: thermal noise standard deviation
        #print 'init_state@DeltaSigmaModulator' # DEBUG
        #print s # DEBUG        
        #return StateVariable(s, self.internal_range, sigma) if (self.clamping or sigma!=0) else s
 
# DeltaSigmaModulator child class: for SECOND ORDER modulators
class DeltaSigmaModulator_2_(DeltaSigmaModulator):
    """
    Second order single-loop topologies            
        topology = 'ff' | 'mf' 
                   'ff' : feed-forward
                   'mf' : multiple-distributed feed-back        
    """    
    modulator_order = 2
    state_variables = ['s1', 's2']
    
    def __init__(self, coeff_i, coeff_x, ic, **kwds):
        self.coeff_i = coeff_i # integrator coefficients
        self.coeff_x = coeff_x # feed-forward | feed-back coefficients
        self.ic = ic           # initial conditions
        super(DeltaSigmaModulator_2_, self).__init__(**kwds)
    
    def init_state_variables(self, length, sigma_arr=(0,0)):
        # initialize the state variables
        #self.s1 = self.init_state(np.zeros(length), sigma_arr[0]) # integrator 1
        #self.s2 = self.init_state(np.zeros(length), sigma_arr[1]) # integrator 2
        self.s1 = np.zeros(length)
        self.s2 = np.zeros(length)
        self.od = np.zeros(length)                      # digital output
        self.dac = np.zeros(length)                     # feed-back dac
        
    def apply_initial_conditions(self):
        # initial conditions
        self.s1[-1] = self.ic[0]
        self.s2[-1] = self.ic[1]
        self.od[-1], self.dac[-1] = self.quantize(self.s2[-1])        
    
    def __call__(self, stim, init_state_variables=True, 
                 apply_initial_conditions=True, sigma_arr=(0,0)):
        # sigma_arr: array containing the thermal noise standard deviation for
        # for each state variable, assigned in the same order as declared in 
        # the state_variable attribute     
        if init_state_variables: self.init_state_variables(len(stim), sigma_arr)
        if apply_initial_conditions: self.apply_initial_conditions()
        # run simulation for a given input stimulus 'stim'
        if self.topology == 'ff':
            return self.run_ff(stim)
        elif self.topology == 'mf':
            return self.run_mf(stim)
            
    def run_mf(self, stim):
        """
        multiple-distributed feed-back topology : equations
        """
        for k in xrange(len(stim)):
            # first integrator
            self.s1[k] = self.s1[k-1] + \
                self.coeff_i[0]*stim[k-1] - \
                self.coeff_x[0]*self.dac[k-1]
            # second integrator
            self.s2[k] = self.s2[k-1] + \
                self.coeff_i[1]*self.s1[k-1] - \
                self.coeff_x[1]*self.dac[k-1]
            # Quantizer and DAC
            self.od[k], self.dac[k] = self.quantize(self.s2[k])
            
    def run_ff(self, stim):
        """
        feed-forward topology : equations
        """        
        for k in xrange(len(stim)):
            # first integrator
            self.s1[k] = self.s1[k-1] + \
                self.coeff_i[0]*(stim[k-1] - self.dac[k-1])
            # second integrator
            self.s2[k] = self.s2[k-1] + \
                self.coeff_i[1]*self.s1[k-1]
            # summer node
            sn = stim[k] + \
                self.coeff_x[0]*self.s1[k] + \
                self.coeff_x[1]*self.s2[k]
            # Quantizer and DAC
            self.od[k], self.dac[k] = self.quantize(sn)

#################################### MAIN #####################################

if __name__ == '__main__':
    
    description = """
    * DSM module
    
    This module provides general classes for the high-level modeling of 
    discrete-time Delta-Sigma Modulators.
    
    A short example is also provided to clarify the use of the incuded classes. 
    """
    print(description)
    
    # SPECIFIC IMPORTS
    from scipy.signal import periodogram, welch
    
    # Operating conditions 
    fs = 1e6
    osr = 128
    amp_dB = -12
    ncyc = 16
    bw = 0.5*fs/float(osr)
    ftone = bw/4
    
    # Example modulator definition
    mod = DeltaSigmaModulator_2_(topology='ff', 
                                 n_dac_levels=2, 
                                 clamping=True, 
                                 coeff_i=[0.25, 0.25],  
                                 coeff_x=[1.0, 1.0], 
                                 ic=[0.0, 0.0])

    # set harmonic stimulus stimulus
    t = (1/fs)*np.arange(0, ncyc*(1/ftone)*fs)
    stim_norm = np.sin(2*np.pi*ftone*t)
    stim = 10**(amp_dB/20.0)*stim_norm
    
    # modulator simulation 
    mod(stim)
    
    # PSD:
    f, sxx = periodogram(x=mod.od, fs=fs, window=('kaiser',15))
    p1 = PlotPSD(sxx, fsamp=fs, dB_at_fin=amp_dB, ftone=ftone, 
                 bandwidth=[0, bw], leakage=5)
    p1.slope_legend.slope_dB_dec = 40
    p1.decorate()
    p1.ax.set_title('2nd Order single-loop, single-bit DSM')
    
    # SNDR curve:
    # One-shot simulation function
    def one_shot(amp_dB, mod, stim_norm, ftone, bandwidth):
        mod(10**(amp_dB/20.0)*stim_norm)
        f, sxx = periodogram(x=mod.od, fs=fs, window=('kaiser',15))
        # Note: welch estimation can be used insted if averaged spetra are preferred
        p = PatternPSD(sxx=sxx, fsamp=fs, ftone=ftone, bandwidth=bandwidth, leakage=5)
        return p.SNDR, p.SNR, p.SDR, p.SFDR

    # SNDR curve simulation
    amp_dBs = np.hstack((np.arange(-100,-20,20), np.arange(-24,2,2)))
    sim = np.vectorize(one_shot, excluded=['mod', 'stim_norm', 'ftone', 'bandwidth'])
    SNDR1, SNR1, SDR1, SFDR1 = sim(amp_dB=amp_dBs, mod=mod, stim_norm=stim_norm, ftone=ftone, bandwidth=[0, bw])
    
    # SNDR curve plot
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    ax.plot(amp_dBs, SNDR1)
    ax.set_title('2nd Order single-loop, single-bit DSM')
    ax.grid(True)
    ax.set_xlabel('Amplituted [dBFS]')
    ax.set_ylabel('SNDR [dB]')