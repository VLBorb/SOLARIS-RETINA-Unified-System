# Hybrid Solaris Unified System with Adaptive Mutation & Strength, Rescaled Fitness, Logging & 3D Visualization

# Automatic dependency installation (for Colab)
# !pip install numpy scipy matplotlib pandas torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # For 3D plotting
import torch

# Check GPU availability
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(device)}")
else:
    device = torch.device("cpu")
    print("GPU not available, using CPU.")

# --- Photonic Model Constants and Functions ---
c = 299_792_458  # Speed of light in m/s
h = 6.626e-34    # Planck's constant in J*s
alpha = 0.65     # System-specific exponent parameter

def spectral_activation(λ_nm):
    """Determines a spectral activation coefficient based on wavelength."""
    return np.select(
        [λ_nm >= 650, λ_nm >= 590, λ_nm >= 570, λ_nm >= 490, λ_nm >= 450, λ_nm >= 420, λ_nm >= 380],
        [1.0, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65], default=0.0)

def spectral_fractal_refraction(wavelength_nm, theta_rad, refractive_index, polarization_mag, entropy_param):
    """Calculates a conceptual 'fractal photonic refraction' value."""
    λ = wavelength_nm * 1e-9  # nm to m
    rho = (np.sin(theta_rad) / refractive_index) * np.cos(np.abs(polarization_mag))
    epsilon = 1 / (1 + entropy_param)
    sigma = spectral_activation(wavelength_nm)
    E_photon = h * (c / λ)
    core_value = (c / λ) * rho * epsilon * sigma
    # Avoid math domain error for log2 if core_value**alpha is -1, though unlikely with positive core_value
    if 1 + core_value**alpha <= 0:
        return 0.0 
    return np.log2(1 + core_value**alpha) * E_photon

def entropy_condensation(data):
    """Applies a logarithmic transformation to condense the data range."""
    return np.log1p(np.abs(data))

# --- Adaptive Retina Module ---
class AdaptiveRetina:
    def __init__(self, population_size=50, retina_size=20, initial_mutation_rate=0.1,
                 expected_raw_output_scale=1e-17, 
                 target_scaling_factor_for_agent_range=0.5):
        self.population_size = population_size
        self.retina_size = retina_size
        self.population = [np.random.rand(retina_size) for _ in range(population_size)]
        
        # Mutation parameters (scalar, adaptive)
        self.mutation_rate = initial_mutation_rate  # Probability of mutation
        self.mutation_strength = initial_mutation_rate # Magnitude of mutation (std dev of noise)
        
        self.current_diversity = 0.0 # For logging

        # Parameters for fitness target scaling
        self.expected_raw_output_scale = expected_raw_output_scale
        self.target_scaling_factor_for_agent_range = target_scaling_factor_for_agent_range

    def evaluate_fitness(self, raw_photonic_output):
        """
        Fitness is negative absolute difference between agent's mean and a scaled target_output.
        """
        # Re-scale fitness target to be more comparable to agent's mean value range [0,1]
        if self.expected_raw_output_scale == 0: # Avoid division by zero
            scaled_target = 0.0
        elif raw_photonic_output == 0: # Handle zero output directly
            scaled_target = 0.0
        else:
            scaled_target = (raw_photonic_output / self.expected_raw_output_scale) * \
                            self.target_scaling_factor_for_agent_range
        
        # Clip scaled_target to be within the possible range of np.mean(agent), e.g., [0,1]
        scaled_target = np.clip(scaled_target, 0.0, 1.0) 
        
        fitnesses = []
        for agent in self.population:
            mean_val = np.mean(agent)
            fitness = -np.abs(mean_val - scaled_target) # Compare with scaled_target
            fitnesses.append(fitness)
        return fitnesses

    def calculate_diversity(self):
        """Calculate diversity as the standard deviation of mean agent values."""
        if not self.population: return 0.0
        mean_vals = [np.mean(agent) for agent in self.population]
        return np.std(mean_vals)

    def evolve(self, fitnesses):
        """Evolves the population using elitism and adaptive mutation (rate and strength)."""
        # Elitism: top 20% survive, ensure at least one survivor
        elite_count = max(1, int(0.2 * self.population_size))
        if self.population_size == 0 : elite_count = 0

        elite_indices = np.argsort(fitnesses)[-elite_count:]
        survivors = [self.population[i] for i in elite_indices if i < len(self.population)]


        # Update diversity and adaptive mutation parameters
        diversity = self.calculate_diversity()
        self.current_diversity = diversity 

        # Adaptive mutation rate (probability) & strength (magnitude)
        # Rate/Strength increases as diversity decreases, clamped between 0.05 and 0.20.
        # Diversity is typically <= 0.5 for values in [0,1], clipping (1-diversity) is robust.
        adaptive_factor = 0.05 + 0.15 * (1 - np.clip(diversity, 0, 1))
        self.mutation_rate = adaptive_factor
        self.mutation_strength = adaptive_factor
        
        # Rebuild population
        new_population = survivors.copy()
        while len(new_population) < self.population_size:
            if not survivors: # Fallback if no survivors (e.g., population_size was 0)
                 parent = np.random.rand(self.retina_size)
            else:
                parent = survivors[np.random.randint(len(survivors))]
            
            # Inlined mutation logic:
            child_genes = []
            for gene in parent:
                if np.random.rand() < self.mutation_rate:
                    child_genes.append(gene + np.random.normal(0, self.mutation_strength))
                else:
                    child_genes.append(gene)
            child = np.clip(child_genes, 0, 1)
            new_population.append(child)
        self.population = new_population

# --- Solaris Core System ---
class SolarisUnifiedSystem:
    def __init__(self, population_size=50, retina_size=20, initial_mutation_rate=0.1,
                 refractive_index=1.336, polarization_mag=0.5, entropy_param=0.75,
                 expected_raw_output_scale=1e-17, target_scaling_factor_for_agent_range=0.5):
        
        self.retina = AdaptiveRetina(population_size, retina_size, initial_mutation_rate,
                                     expected_raw_output_scale, target_scaling_factor_for_agent_range)
        self.refractive_index = refractive_index
        self.polarization_mag = polarization_mag
        self.entropy_param = entropy_param

    def run_evolution_cycle(self, λ_spectrum, θ_angles):
        """Runs the main simulation cycle with logging."""
        condensed_outputs = []
        λ_mesh, θ_mesh = np.meshgrid(λ_spectrum, θ_angles, indexing='ij') # For 3D plot

        num_steps = len(λ_spectrum) * len(θ_angles)
        print(f"Starting evolution cycle for {num_steps} steps...")

        for i, λ_val in enumerate(λ_spectrum): # Renamed λ to λ_val to avoid conflict
            for j, θ_val in enumerate(θ_angles): # Renamed θ to θ_val
                current_step = i * len(θ_angles) + j + 1
                raw_output = spectral_fractal_refraction(
                    λ_val, θ_val,
                    self.refractive_index,
                    self.polarization_mag,
                    self.entropy_param
                )

                fitnesses = self.retina.evaluate_fitness(raw_output)
                self.retina.evolve(fitnesses)
                condensed_outputs.append(entropy_condensation(raw_output))

                # Log metrics periodically
                if current_step % (max(1, num_steps // 10)) == 0 or current_step == num_steps: # Log ~10 times + last step
                    diversity_log = self.retina.current_diversity
                    mutation_rate_log = self.retina.mutation_rate
                    mutation_strength_log = self.retina.mutation_strength
                    print(f"Step {current_step}/{num_steps}: RawOut={raw_output:.2e}, "
                          f"Div={diversity_log:.3f}, MutRate={mutation_rate_log:.3f}, MutStr={mutation_strength_log:.3f}")

        print("Evolution cycle completed.")
        return λ_mesh.flatten(), θ_mesh.flatten(), condensed_outputs

# --- Main Execution Block ---
if __name__ == "__main__":
    print("Initializing Hybrid Solaris Unified System (Finalized)...")
    solaris_system = SolarisUnifiedSystem(
        population_size=50,
        retina_size=20,
        initial_mutation_rate=0.1, # Affects both initial rate and strength
        refractive_index=1.33,
        polarization_mag=0.45,
        entropy_param=0.7,
        expected_raw_output_scale=1e-17, # For fitness target scaling
        target_scaling_factor_for_agent_range=0.75 # Aim to scale target towards [0, 0.75]
    )

    # Define simulation parameters (wavelengths in nm, angles in radians)
    λ_spectrum = np.linspace(380, 780, 25) # Number of wavelength points
    θ_angles = np.linspace(0, np.pi/2, 15) # Number of angle points

    print(f"Running simulation with {len(λ_spectrum)} wavelengths and {len(θ_angles)} angles.")
    λ_flat, θ_flat, final_condensed_outputs = solaris_system.run_evolution_cycle(λ_spectrum, θ_angles)

    # --- Displaying Results ---
    print("\n--- Simulation Results ---")
    output_df = pd.DataFrame({
        'Wavelength': λ_flat,
        'Angle': θ_flat,
        'CondensedOutput': final_condensed_outputs
    })
    print("First 5 data points:")
    print(output_df.head())
    print(f"\nStatistics for Condensed Output:\n{output_df['CondensedOutput'].describe()}")

    # --- 3D Visualization ---
    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(λ_flat, θ_flat, final_condensed_outputs, 
                         c=final_condensed_outputs, cmap='viridis', s=25, alpha=0.8, edgecolor='k', linewidth=0.5)
    ax.set_xlabel('Wavelength (nm)', fontsize=12)
    ax.set_ylabel('Angle (rad)', fontsize=12)
    ax.set_zlabel('Condensed Output', fontsize=12)
    ax.set_title('Hybrid Solaris: 3D Photonic Output Surface', fontsize=16)
    fig.colorbar(scatter, ax=ax, label='Condensed Output Value', pad=0.1) # Add colorbar
    plt.tight_layout()
    plt.show()

    # --- Optional: Plot final retina agent characteristics ---
    final_population = solaris_system.retina.population
    if final_population:
        avg_genes_final_pop = [np.mean(agent) for agent in final_population]
        plt.figure(figsize=(10,5))
        plt.hist(avg_genes_final_pop, bins=15, color='teal', edgecolor='black', alpha=0.7)
        plt.title('Distribution of Mean Values of Agents in Final Retina Population', fontsize=14)
        plt.xlabel('Mean Agent Value (Targeting Scaled Photonic Output)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
