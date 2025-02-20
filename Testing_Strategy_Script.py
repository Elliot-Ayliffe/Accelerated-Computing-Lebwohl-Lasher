"""
Elliot Ayliffe (14/02/25)

This script tests if a modified version of the Lebwohl-Lahser Model code performs correctly by 
comparing its output to that of the origial version.

PLOTS ENERGY AND ORDER: Reduced energy and Order parameter are plotted for both versions to allow 
                        visual comparion of trends.

CALCULATES AND COMPARES STATISTICS: The range, mean, and standard deviation of each variable (energy and order) 
                                    and computed for both versions for comparison. 
                                    
SAVES TEST RESULTS TO AN OUTPUT FILE: The computed stats are saved to an output file (.txt)


This serves as a validation check to ensure that any modifications made to accelerate the code do not 
compromise the accuracy/correctness of the simulation results.

Run at the command line:
python Testing_Strategy_Script.py <original_code_output_file.txt> <new_version_output_file.txt>

where:
    original_code_output_file.txt = the text file that is produced when you run the original (uneditied)
    Lebwohl-Lasher script 

    new_version_output_file.txt = the text file that is produced when you run the modified Lebwohl-Lasher script

    
Note: Both versions of the code (original and new) should be run with the same parameters (iterations, lattice size, reduced temperature)
      to ensure a valid comparison.
"""

import sys
import numpy as np 
import scipy.stats as stats
import matplotlib.pyplot as plt

def load_simulation_output_file(output_filname):
    """
    Reads in a output text file and extracts variables (Monte Carlo step, Energy, Order)

    Args:
        output_filename (str): output file path
    
    Returns:
        3 numpy arrays containing values only (MC steps, energy, order)
    """

    MCS, reduced_energy, order = [], [], []

    with open(output_filname, 'r') as output_file:
        # Loop over each line in the text file 
        for each_line in output_file: 
            columns = each_line.strip().split()     # split the current line into columns 

            # if the line contains 4 columns of numeric data, store the MC step, energy and order as lists
            # leave out the acceptence ratio column 
            if len(columns) == 4 and columns[0].isdigit():
                MCS.append(int(columns[0]))     # MC step = 1st column 
                reduced_energy.append(float(columns[2]))    # Energy = 3rd column
                order.append(float(columns[3]))     # Order = 4th column 

    # Convert the lists to numpy arrays to allow for plotting 
    return np.array(MCS), np.array(reduced_energy), np.array(order)


def calculate_statistics(variable_data):
    """
    Calculate the mean, standard deviation, and range of the inputted data (variable)

    Args:
        variable_data (numpy array): variable data (e.g. values of energy)

    Returns: 
        stats : a dictionary of the mean, SD and range of the dataset
    """
    stats = {"Mean": np.mean(variable_data), "SD": np.std(variable_data), "Range": (np.min(variable_data), np.max(variable_data))}
    
    return stats 

# def t_test(original_dataset, new_dataset):
#     """ 
#     Conducts a paired t-test to quantify any significant differences between the two datasets.
#     This helps determine whether the simulation is still working correctly using statistical evidence.
#     A high p-value = the datasets are similar and the new version of code likely works correctly

#     Args:
#         original_dataset (numpy array): dataset produced by the original code (e.g. energy or order)
#         new_dataset (numpy array): dataset produced by the new veriosn of code (e.g. energy or order)

#     Returns:
#         p-value and t-statistic
#     """
#     # perform t-test on data to determine any differences (incorrect output of the code)
#     t_statistic, p_value = stats.ttest_rel(original_dataset, new_dataset)
#     return t_statistic, p_value


def energy_order_plotter(MCS, energy_original, energy_new_version, order_original, order_new_version, filename_new_version):
    """
    Produces 2 plots (Reduced Energy vs MC step, Order vs MC step) that contains the values of the old and new simulations to 
    allow for easy comparison. This makes it easy to check whether the new version of the script still performs correctly 
    in reference to the original (expected) output.

    Args:
        MCS (numpy array): Monte carlo steps (iterations)
        energy_original (numpy array): the energy values outputted from the original code 
        energy_new_version (numpy array): the energy values outputted from the new version of code 
        order_original (numpy array): the order values outputted from the original code 
        order_new_version (numpy array): the order values outputted from the new version of code 
        filename_new_version (str): the filename used to save the figures 
    """

    fig, ax =  plt.subplots(1,2, figsize=(14, 6))

    # Plotting Reduced energy (for both old and new versions)
    ax[0].plot(MCS, energy_original, label="Original (unedited) Simulation", color='green', linestyle='--')
    ax[0].plot(MCS, energy_new_version, label="New Version Simulation", color='red', linestyle='-')
    ax[0].set_title("Reduced Energy vs Monte Carlo Step")
    ax[0].set_xlabel("MCS")
    ax[0].set_ylabel("Reduced Energy")
    ax[0].legend()

    # Plotting Order Parameter (for both old and new verions)
    ax[1].plot(MCS, order_original, label="Original (unedited) Simulation", color='green', linestyle='--')
    ax[1].plot(MCS, order_new_version, label="New Version Simulation", color='red', linestyle='-')
    ax[1].set_title("Order Parameter vs Monte Carlo Step")
    ax[1].set_xlabel("MCS")
    ax[1].set_ylabel("Order Parameter")
    ax[1].legend()          

    # For saving the figure 
    figure_filename = f"validation_plot_{filename_new_version}.png"
    plt.savefig(figure_filename, dpi=300)
    plt.tight_layout()
    plt.show()


def save_stats_to_file(results_filename, stats_original, stats_new_version):
    """
    Saves the calculated statistics to a text file

    Args:
        results_filename (str): the filename of the outputted text file 
        stats_original (dict): the stats (mean, SD and range) of the original simulation 
        stats_new_version (dict): the stats (mean, SD and range) of the new version simulation 
        energy_pvalue (float): p-value from comparing the energy values 
        energy_tstat (float): t-statistic from comparing the energy values 
        order_pvalue (float): p-value from comparing the order values 
        order_tstat (float): t-statistic from comparing the order values
    """

    # Open the file and set it to 'write' mode 
    with open(results_filename, "w") as results_file:
        
        # Create the contents of the output file 
        results_file.write("# NEW VERSION PERFORMANCE VALIDATION RESULTS\n")
        results_file.write("This is a test to check that a new modified version of the Lebwohl-Lasher code still\n works correctly and produces comparable outputs to the original program\n")
        results_file.write("#=======================================================================================")

        # Reduced Energy stats 
        results_file.write("\n# REDUCED ENERGY:\n")
        results_file.write(f"Original Code: Mean = {stats_original['Energy']['Mean']:.2f}, SD = {stats_original['Energy']['SD']:.3f}, Range = {stats_original['Energy']['Range']}\n")
        results_file.write(f"New Version: Mean = {stats_new_version['Energy']['Mean']:.2f}, SD = {stats_new_version['Energy']['SD']:.3f}, Range = {stats_new_version['Energy']['Range']}\n")
        

        # Order stats 
        results_file.write("\n# ORDER PARAMETER:\n")
        results_file.write(f"Original Code: Mean = {stats_original['Order']['Mean']:.2f}, SD = {stats_original['Order']['SD']:.3f}, Range = {stats_original['Order']['Range']}\n")
        results_file.write(f"New Version: Mean = {stats_new_version['Order']['Mean']:.2f}, SD = {stats_new_version['Order']['SD']:.3f}, Range = {stats_new_version['Order']['Range']}\n")
        

# Main function to run 
def main():
    
    # make sure the user enters correct arguments into the command line 
    if len(sys.argv) != 3:
        print("Wrong command line usage.")  # error message 
        print("python Testing_Strategy_Script.py <original_code_output_file.txt> <new_version_output_file.txt>")
        
        return 
    
    # get filenames from the arguments given in the command line 
    original_output_file, new_output_file = sys.argv[1], sys.argv[2]

    # load in both output files 
    MCS_original, energy_original, order_original = load_simulation_output_file(original_output_file)
    MCS_new_version, energy_new_version, order_new_version = load_simulation_output_file(new_output_file)

    # Calculate stats 
    stats_original = {"Energy": calculate_statistics(energy_original), "Order": calculate_statistics(order_original)}
    stats_new_version = {"Energy": calculate_statistics(energy_new_version), "Order": calculate_statistics(order_new_version)}


    # Save statistics to output text file 
    save_stats_to_file(f"Version_Test_Stats_{new_output_file}", stats_original, stats_new_version)

    # Plot energy and order to allow visual comparison between simulations
    energy_order_plotter(MCS_original, energy_original, energy_new_version, order_original, order_new_version, new_output_file)


if __name__ == "__main__":
    main()


