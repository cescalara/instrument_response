import numpy as np
import h5py

from .power_law import BoundedPowerLaw
from .detector import EffectiveArea, Calorimeter, Response


class ToySimulation(object):
    """
    A toy simulation used to demonstrate how an instrument response works.
    """

    def __init__(self, power_law_index, min_energy, max_energy):
        """
        A toy simulation used to demonstrate how an instrument response works.
       
        @param power_law_index spectral index of the injection energy spectrum
        @param min_energy minimum energy in TeV
        @param max_energy minimum energy in TeV
        """

        self.power_law = BoundedPowerLaw(power_law_index, min_energy, max_energy)

        # Define an effective area as a function of energy for the detector
        effective_area_scale = 10.0 # m^2
        effective_area_index = 2.0 # power law index of increase in E
        effective_area_break = 10.0 # TeV

        self.effective_area = EffectiveArea(effective_area_scale, effective_area_index, effective_area_break) 
        
        # Define detector properties
        calorimeter_scale = 1000.0
        calorimeter_break = 10.0 # TeV
        calorimeter_uncertainty = 0.2

        self.calorimeter = Calorimeter(calorimeter_scale, calorimeter_break, calorimeter_uncertainty)


    def run(self, N, T = 1):
        """
        Run the simulation
        
        @param N number of particles/photons to simulate
        @param T time period of simulation [yr]
        """

        self.N = N
        self.total_dN_dt = N / T # [yr^-1]
        self.total_dN_dtdA = N / (T * self.effective_area.maximum) # [yr^-1 m^-2] 
        
        # Sample energies from a power law
        self.true_energy = self.power_law.samples(self.N)
        self.initial_energy = self.true_energy
 
        # Effective area
        interaction_probability = self.effective_area.interaction_probability(self.true_energy)
        interacted_in_detector = np.zeros(len(interaction_probability))
        for i, p in enumerate(interaction_probability):
            interacted_in_detector[i] = np.random.choice([0, 1], p=[1-p, p])
        
        self.true_energy = self.true_energy[np.where(interacted_in_detector == 1.0)]

        # Event generation
        # Each photon/particle produces a number of secondary particles proportional to its energy.
        # The number of secondary particle follows a poisson distribution.
        expected_number_of_secondaries = self.calorimeter.expected_number_of_secondaries(self.true_energy)
        self.number_of_secondaries = np.random.poisson(expected_number_of_secondaries)

        # Assume all energy divided equally among secondaries
        energy_per_secondary = self.true_energy / self.number_of_secondaries

        # Some fraction of these secondaries hit the detector.
        # Others are lost/absorbed in the detector material, or just do not interact.
        self.detected_fraction = np.random.normal(0.5, 0.1, len(self.number_of_secondaries))
        self.number_of_detected_secondaries = self.detected_fraction * self.number_of_secondaries
        
        # Now digitise energy into bins based on number of detected secondaries (ie. pulse height)
        # This will be done based on some calibration procedure
        self.detected_energy = self.calorimeter.detected_energy(self.number_of_detected_secondaries, energy_per_secondary)

        
    def save(self, filename):
        """
        Save the results to file.
        """

        with h5py.File(filename, 'w') as f:
            f.create_dataset('initial_energy', data = self.initial_energy) 
            f.create_dataset('true_energy', data = self.true_energy)
            f.create_dataset('number_of_secondaries', data = self.number_of_secondaries)
            f.create_dataset('number_of_detected_secondaries', data = self.number_of_detected_secondaries)
            f.create_dataset('detected_energy', data = self.detected_energy)
            f.create_dataset('effective_area_maximum', data = self.effective_area.maximum)

        

    
