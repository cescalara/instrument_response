import numpy as np

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
        standard_deviation = self.detection_uncertainty * mean
        
        detected_energy = np.random.normal(mean, standard_deviation)
        detected_energy[detected_energy < 0] = 0

        return detected_energy
