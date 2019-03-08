import numpy as np

from power_law import BoundedPowerLaw

class ToySimulation(object):
    """
    A toy simulation used to demonstrate how an instrument response works.
    """

    def __init__(self, power_law_index, min_energy, max_energy, energy_channels):
        """
        A toy simulation used to demonstrate how an instrument response works.
        """

        self.power_law = BoundedPowerLaw(power_law_index, min_energy, max_energy)
        self.energy_channels = energy_channels


    def run(self, N):
        """
        Run the simulation
        """

        self.N = N

        # Sample energies from a power law
        true_energy = self.power_law.sample(self.N)
        
        # Model detection effects
        
        # Save simulation outputs
