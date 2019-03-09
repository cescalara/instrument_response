import numpy as np
import matplotlib
from matplotlib import pyplot as plt

class EffectiveArea(object):
    """
    Simple effective area as a function of enegry.
    """

    def __init__(self, scale, index, energy_break):
        """
        Simple effective area as a function of enegry.

        Aeff(E) = A * E**I * exp(-E/Eb) 

        @param scale S in the above expression [m^2]
        @param index I in the above expression
        @param energy_break Eb in the above expression [TeV]
        """

        self.scale = scale
        self.index = index
        self.energy_break = energy_break

        E = np.linspace(1.0, energy_break * 10, 1e4)
        self.maximum = max(self.evaluate(E))

    def evaluate(self, E):
        """
        Effective area as a function of energy.

        @param E energy [TeV]
        """

        return self.scale * E**self.index * np.exp(-E/self.energy_break)

    def interaction_probability(self, E):
        """
        P(interacts | E) is related to the effective area. If we have P(interaction) = 1 at the maximum, then 
        the interaction probability is just effective area / max(effective area).
        """

        return self.evaluate(E) / self.maximum
        

class Calorimeter(object):
    """
    Model simple interactions in a toy calorimeter.
    """

    def __init__(self, scale, energy_break, detection_uncertainty):        
        """
        Model simple interactions in a toy calorimeter.
        """

        self.scale = scale
        self.energy_break = energy_break
        self.detection_uncertainty = detection_uncertainty

    def expected_number_of_secondaries(self, E):
        """
        Number of secondaries prodcued in the calorimeter is linear in energy.

        @param E energy in TeV
        """
        
        return self.scale * E / self.energy_break

    
    def detected_energy(self, number_of_detected_secondaries, energy_per_secondary):
        """
        Based on some calibration procedure, define the detected energy based on 
        the number of secondaries.
        """

        mean = number_of_detected_secondaries * energy_per_secondary
        mean[mean <= 0 ] = 0.0001
        standard_deviation = self.detection_uncertainty * mean
        
        detected_energy = np.random.normal(mean, standard_deviation)
        detected_energy[detected_energy < 0] = 0

        return detected_energy

    
class Response(object):
    """
    Detector response information.
    """

    def __init__(self, initial_energy, true_energy, detected_energy, nbins_true_energy, nbins_detected_energy, Aeff_max):
        """
        Detector response information.
        
        @param initial_energy initial energies from simulation [TeV]
        @param true_energy true energies from simulation [TeV]
        @param detected_energy detected energies from simulation [TeV]
        @param nbins_true_energy number of true energy bins
        @param nbins_detected_energy number of detected energy bins   
        @param Aeff_max maximum of the effective area in m^2
        """

        true_energy_bins = np.logspace(np.log(min(true_energy)), np.log(max(true_energy)),
                                       nbins_true_energy+1, base=np.e)
        detected_energy_bins = np.logspace(np.log(min(detected_energy)), np.log(max(detected_energy)),
                                           nbins_detected_energy+1, base=np.e)
        
        # Get energy dispersion histogram
        self.H, self.true_energy_bins, self.detected_energy_bins = np.histogram2d(true_energy, detected_energy,
                                                                   bins=[true_energy_bins, detected_energy_bins]);

        #self.true_energy_bins = np.exp(self.true_energy_bins)
        #self.detected_energy_bins = np.exp(self.detected_energy_bins)  

        h_init, _ = np.histogram(initial_energy, bins=self.true_energy_bins)
        h_true, _ = np.histogram(true_energy, bins=self.true_energy_bins)
        effective_area = h_true / h_init
    
        # For each bin, divide by input MC counts and multiply by corresponding
        # effective area factor
        self.matrix = np.zeros((nbins_true_energy, nbins_detected_energy))
        for i in range(nbins_true_energy):
            for j in range(nbins_detected_energy):
                self.matrix[i][j] = (self.H[i][j] / h_true[i]) * effective_area[i]

        # Include effective area
        self.matrix = self.matrix #* Aeff_max
        
        
    def show(self):
        """
        Show a ridgeplot of response in each bin.
        """

        # Colormap
        norm = matplotlib.colors.Normalize(vmin=min(np.log(self.true_energy_bins[:-1])), 
                                           vmax=max(np.log(self.true_energy_bins[:-1])))
        cmap = matplotlib.cm.get_cmap('viridis')
        
        y_values = []
        for i, E in enumerate(self.true_energy_bins[:-1]):
            y_values.append(self.matrix[i])
        y_values = np.array(y_values)
        x_values = self.detected_energy_bins[:-1]
            
        y_add = 0.0
        max_y = np.max([np.max(_) for _ in y_values])
        delta_y = 0.1 * max_y

        # Plot
        fig, ax = plt.subplots()
        fig.set_size_inches((10, 7))

        for i, y in enumerate(y_values):  
            idx = y > 0
            ax.fill_between(x_values[idx], y_add, y_add+y[idx], zorder=-10, 
                            color=cmap(norm(np.log(self.true_energy_bins[i]))), alpha = 0.5)
            y_add+=delta_y

        # Formatting
        ax.set_yticks([]);
        for s in ['left', 'right', 'top']:
            ax.spines[s].set_visible(False)
        ax.set_xlabel('$E_\mathrm{det}$ / TeV')
        ax.set_xscale('log')
        ax.set_title('P($E_\mathrm{det}$ | $E_\mathrm{true}$, detector model)');

        
