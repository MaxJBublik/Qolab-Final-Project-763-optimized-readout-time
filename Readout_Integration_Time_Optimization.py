# -*- coding: utf-8 -*-
"""
READOUT OPTIMIZATION: INTEGRATION TIME

This sequence measures the I/Q distributions for ground |g> and excited |e> states
across varying integration times to find the optimal readout duration that maximizes
measurement fidelity.

The optimization balances two competing effects:
1. SNR improvement (grows as sqrt(t)) - longer integration improves noise averaging
2. T1 decay (decays as exp(-t/T1)) - longer integration increases qubit relaxation probability

The node computes: F_measure(t) = (1/2)[1 + erf(SNR/(2*sqrt(2)))] * exp(-t/T1)
and identifies t_optimal = argmax F_measure(t)

Prerequisites:
    - Calibrated qubit pi pulse (x180) from power_rabi
    - Calibrated readout frequency from resonator_spectroscopy or readout_frequency_optimization
    - Measured T1 relaxation time from T1 measurement node
    - Set desired flux bias

Next steps:
    - Update the integration time in the state to t_optimal
    - Save the current state
"""

# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration, active_reset
from quam_libs.lib.qua_datasets import convert_IQ_to_V
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray, load_dataset
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from typing import Literal, Optional, List
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf

# %% {Node_parameters}
class Parameters(NodeParameters):
    """
    Parameters for the readout time optimization node.
    """
    qubits: Optional[List[str]] = None  # List of qubit names to calibrate; None = all active qubits
    num_averages: int = 1000  # Number of single-shot measurements per integration time (N_avg in abstract)
    min_integration_time_ns: int = 100  # Minimum integration time in nanoseconds
    max_integration_time_ns: int = 16100  # Maximum integration time in nanoseconds
    integration_time_step_ns: int = 200  # Step size for integration time sweep in nanoseconds
    flux_point_joint_or_independent: Literal["joint", "independent"] = "independent"  # Flux bias configuration
    reset_type: Literal["active", "thermal"] = "thermal"  # Qubit reset method between measurements
    simulate: bool = False  # Whether to simulate the QUA program instead of running
    simulation_duration_ns: int = 5000  # Duration of simulation in nanoseconds
    timeout: int = 100  # Timeout for QM session in seconds
    load_data_id: Optional[int] = None  # ID of previously saved data to reload and reanalyze
    multiplexed: bool = False  # Whether to measure all qubits simultaneously

description = """Typical Runtime w/Default Params:
5-10 seconds per qubit.

Optimizes readout integration time by balancing:
- SNR improvement (proportional to sqrt(t))
- T1 decay (proportional to exp(-t/T1))


"""

node = QualibrationNode(
    name="Readout_Integration_Time_Optimization", 
    description=description, 
    parameters=Parameters()
)

# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions
u = unit(coerce_to_integer=True)

# Instantiate the QuAM class from the state file
machine = QuAM.load()

# Get the relevant QuAM components
if node.parameters.qubits is None or node.parameters.qubits == "":
    qubits = machine.active_qubits  # Use all active qubits
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits]  # Use specified qubits
num_qubits = len(qubits)


# Create array of integration times to analyze (in nanoseconds)
integration_times_ns = np.arange(
    node.parameters.min_integration_time_ns,
    node.parameters.max_integration_time_ns + node.parameters.integration_time_step_ns,
    node.parameters.integration_time_step_ns,
)

# Use the MAXIMUM integration time for the actual measurement
# We'll analyze shorter times in post-processing
max_integration_time_ns = int(integration_times_ns[-1])
max_integration_time_cycles = max_integration_time_ns // 4

# Generate the OPX and Octave configurations
config = machine.generate_config()

# Open Communication with the QOP (Quantum Orchestration Platform)
if node.parameters.load_data_id is None:
    qmm = machine.connect()

# %% {QUA_program}
n_avg = node.parameters.num_averages  # Number of averages per integration time

flux_point = node.parameters.flux_point_joint_or_independent

with program() as readout_time_opt:
    # Use qua_declaration macro for consistent variable declaration
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    
    # Additional variables for ground and excited state measurements
    I_g = [declare(fixed) for _ in range(num_qubits)]  # In-phase component, ground state
    Q_g = [declare(fixed) for _ in range(num_qubits)]  # Quadrature component, ground state
    I_e = [declare(fixed) for _ in range(num_qubits)]  # In-phase component, excited state
    Q_e = [declare(fixed) for _ in range(num_qubits)]  # Quadrature component, excited state
    
    # Declare streams for ground and excited states
    I_g_st = [declare_stream() for _ in range(num_qubits)]
    Q_g_st = [declare_stream() for _ in range(num_qubits)]
    I_e_st = [declare_stream() for _ in range(num_qubits)]
    Q_e_st = [declare_stream() for _ in range(num_qubits)]
    
    # Loop counters
    shot = [declare(int) for _ in range(num_qubits)]  # Average counter
    
    # Set flux bias for all qubits (if multiplexed, set once; if not, set per qubit)
    if node.parameters.multiplexed:
        for i, qubit in enumerate(qubits):
            machine.set_all_fluxes(flux_point=flux_point, target=qubit)
    
    # Main measurement loop - iterate over each qubit
    for i, qubit in enumerate(qubits):
        
        # Set flux bias (if not multiplexed, set individually per qubit)
        if not node.parameters.multiplexed:
            machine.set_all_fluxes(flux_point=flux_point, target=qubit)
        
        # Single loop: average over many shots
        # We only measure once with the LONGEST integration time
        # Post-processing will analyze different integration durations
        with for_(shot[i], 0, shot[i] < n_avg, shot[i] + 1):
            save(shot[i], n_st)  # Save progress for live monitoring
            
            # =============================================================
            # MEASUREMENT 1: Ground state |g> (qubit in |0>)
            # =============================================================
            
            # Reset qubit to ground state using chosen method
            if node.parameters.reset_type == "active":
                active_reset(qubit, "readout")  # Active reset using measurement feedback
            else:
                qubit.resonator.wait(qubit.thermalization_time * u.ns)  # Passive thermal reset
                qubit.align()
            
            # Measure the resonator with the MAXIMUM integration time
            # This gives us the full integrated signal
            qubit.resonator.measure("readout", qua_vars=(I_g[i], Q_g[i]))
            
            # Save single-shot I/Q values to stream
            save(I_g[i], I_g_st[i])
            save(Q_g[i], Q_g_st[i])
            
            align()  # Ensure measurement completes before next step
            
            # =============================================================
            # MEASUREMENT 2: Excited state |e> (qubit in |1>)
            # =============================================================
            
            # Reset qubit to ground state again
            if node.parameters.reset_type == "active":
                active_reset(qubit, "readout")
            else:
                qubit.resonator.wait(qubit.thermalization_time * u.ns)
                qubit.align()
            
            # Apply pi (pi) pulse to excite qubit from |0> to |1>
            qubit.xy.play("x180")
            
            # Align all elements before measuring
            align()
            
            # Measure the resonator again with the MAXIMUM integration time
            qubit.resonator.measure("readout", qua_vars=(I_e[i], Q_e[i]))
            
            # Save single-shot I/Q values to stream
            save(I_e[i], I_e_st[i])
            save(Q_e[i], Q_e_st[i])
        
        # If not multiplexed, ensure sequential measurement per qubit
        if not node.parameters.multiplexed:
            align()
    
    # Stream processing: organize data and transfer to host
    with stream_processing():
        n_st.save("n")  # Save progress counter
        
        for i in range(num_qubits):
            # Save all individual measurements (not just average)
            # This allows us to calculate standard deviation in Python
            I_g_st[i].buffer(n_avg).save_all(f"I_g{i + 1}")
            Q_g_st[i].buffer(n_avg).save_all(f"Q_g{i + 1}")
            I_e_st[i].buffer(n_avg).save_all(f"I_e{i + 1}")
            Q_e_st[i].buffer(n_avg).save_all(f"Q_e{i + 1}")

# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns * 4)
    job = qmm.simulate(config, readout_time_opt, simulation_config)
    
    # Get the simulated samples and plot them for all controllers
    samples = job.get_simulated_samples()
    fig, ax = plt.subplots(nrows=len(samples.keys()), sharex=True)
    for i, con in enumerate(samples.keys()):
        plt.subplot(len(samples.keys()), 1, i + 1)
        samples[con].plot()
        plt.title(con)
    plt.tight_layout()
    
    # Save the figure
    node.results = {"figure": plt.gcf()}
    node.machine = machine
    node.save()

elif node.parameters.load_data_id is None:
    # Execute the QUA program on real hardware
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(readout_time_opt)
        
        # Live monitoring of progress
        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            n = results.fetch_all()[0]
            # Total iterations = n_avg x num_integration_times
            progress_counter(n, n_avg, start_time=results.start_time)

# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    if node.parameters.load_data_id is None:
        # Fetch the data from the OPX
        # We get arrays of all individual measurements
        ds = fetch_results_as_xarray(
            job.result_handles, 
            qubits, 
            {"shot": np.arange(n_avg)}  # Dimension for individual shots
        )
        
        # Convert IQ data from arbitrary units to volts
        ds = convert_IQ_to_V(ds, qubits, ["I_g", "Q_g", "I_e", "Q_e"])
        
        # Calculate averages and standard deviations manually
        # Average over shots dimension
        I_g_avg = ds.I_g.mean(dim="shot")
        Q_g_avg = ds.Q_g.mean(dim="shot")
        I_e_avg = ds.I_e.mean(dim="shot")
        Q_e_avg = ds.Q_e.mean(dim="shot")
        
        # Standard deviation over shots dimension
        I_g_std = ds.I_g.std(dim="shot")
        Q_g_std = ds.Q_g.std(dim="shot")
        I_e_std = ds.I_e.std(dim="shot")
        Q_e_std = ds.Q_e.std(dim="shot")
        
        # Create new dataset with averaged values and stds
        ds = ds.assign({
            "I_g": I_g_avg,
            "Q_g": Q_g_avg,
            "I_e": I_e_avg,
            "Q_e": Q_e_avg,
            "I_g_std": I_g_std,
            "Q_g_std": Q_g_std,
            "I_e_std": I_e_std,
            "Q_e_std": Q_e_std,
        })
        
        # Remove the shot dimension since we've averaged
        ds = ds.drop_dims("shot")
        
        # =========================================================================
        # SIMULATE DIFFERENT INTEGRATION TIMES
        # =========================================================================
        # Since the integrated value scales with integration time for constant weights,
        # we can simulate shorter integration times by scaling the measured values
        #
        # For integration time t: I(t) = I(t_max) * (t / t_max)
        # And noise scales as: sigma(t) = sigma(t_max) * sqrt(t / t_max)
        
        # Create integration_time dimension with proper xarray DataArray
        integration_time_scale = integration_times_ns / max_integration_time_ns
        
        # Expand dataset to include integration_time dimension
        ds = ds.expand_dims(integration_time=integration_times_ns)
        
        # Create xarray DataArray for scaling with proper dimension alignment
        # This ensures broadcasting works correctly
        scale_array = ds.integration_time / max_integration_time_ns
        
        # Scale I/Q values proportionally to integration time
        for var in ["I_g", "Q_g", "I_e", "Q_e"]:
            ds[var] = ds[var] * scale_array
        
        # Scale standard deviations by sqrt(integration_time_scale)
        for var in ["I_g_std", "Q_g_std", "I_e_std", "Q_e_std"]:
            ds[var] = ds[var] * np.sqrt(scale_array)
        
        ds.integration_time.attrs = {"long_name": "Integration Time", "units": "ns"}
        
    else:
        # Load previously saved data
        node = node.load_from_id(node.parameters.load_data_id)
        ds = node.results["ds"]
    
    # Add the dataset to the node
    node.results = {"ds": ds}
    
    # %% {Data_analysis}
    # =========================================================================
    # COMPUTE SNR AND FIDELITY FOR EACH INTEGRATION TIME
    # =========================================================================
    
    # Calculate pointer separation delta(alpha) = sqrt((I_g - I_e)^2 + (Q_g - Q_e)^2)
    # This is the magnitude of the separation between the two coherent states
    # Equation (1) from the abstract: delta(alpha)(t) = (alpha)_e(t) - (alpha)_g(t)
    ds = ds.assign(
        {
            "delta_alpha": np.sqrt((ds.I_g - ds.I_e) ** 2 + (ds.Q_g - ds.Q_e) ** 2),
        }
    )
    
    # Calculate noise standard deviation sigma
    # Average the noise from both I and Q quadratures, and both |g> and |e> states
    # This represents the width of the Gaussian clouds in the I/Q plane
    ds = ds.assign(
        {
            "sigma": (ds.I_g_std + ds.Q_g_std + ds.I_e_std + ds.Q_e_std) / 4,
        }
    )
    
    # Calculate SNR using Equation (2) from abstract: SNR(t) approximately (|Delta_alpha_ss| / sigma) * sqrt(t)
    # This shows how SNR improves with integration time due to noise averaging
    # Note: integration_time is now in nanoseconds after coordinate assignment
    snr_baseline = ds.delta_alpha / ds.sigma  # This is the ratio at each time point

    
    ds = ds.assign({
        "SNR": ds.delta_alpha / ds.sigma 
        })
    
    # Calculate assignment fidelity from SNR using Equation (3) from abstract
    # F(t) approximately (1/2) * [1 + erf(SNR(t) / (2*sqrt(2)))]
    # This is the fidelity based purely on Gaussian noise separation
    ds = ds.assign(
        {
            "F_assignment": 0.5 * (1 + erf(ds.SNR / (2 * np.sqrt(2)))),
        }
    )
    
    # Calculate measurement fidelity including T1 decay using Equation (5) from abstract
    # F_measure(t) approximately F(t) * exp(-t/T1)
    # This combines noise-limited separation with lifetime-limited relaxation
    
    # Get T1 for each qubit from the machine state (in seconds)
    T1_values_s = np.array([q.T1 for q in qubits])
    print(f"T1 values: {T1_values_s * 1e6} μs")
    
    # Convert T1 to nanoseconds to match integration_time units
    T1_values_ns = T1_values_s * 1e9
    print(f"T1 in ns: {T1_values_ns}")
    # Create xarray DataArray for T1 with proper dimensions
    # This allows proper broadcasting with the dataset
    import xarray as xr
    T1_array = xr.DataArray(
        T1_values_ns,
        dims=["qubit"],
        coords={"qubit": ds.qubit.values}
    )
    
    ds = ds.assign(
        {
            "F_measure": ds.F_assignment * np.exp(-ds.integration_time / T1_array),
        }
    )
    
    # =========================================================================
    # FIND OPTIMAL INTEGRATION TIME
    # =========================================================================
    
    # Find the integration time that maximizes F_measure for each qubit
    # This is t_optimal = argmax F_measure(t) 
    optimal_times = ds.F_measure.idxmax(dim="integration_time")
    
    # Save fitting results 
    fit_results = {}
    for q in qubits:
        opt_time = float(optimal_times.loc[q.name].values)
        
        # Select the data for this qubit first (exact match on string)
        # Then select integration time with method='nearest' (numeric match)
        fidelity_at_opt = ds.F_measure.sel(qubit=q.name).sel(
            integration_time=opt_time, 
            method='nearest'
        )
        snr_at_opt = ds.SNR.sel(qubit=q.name).sel(
            integration_time=opt_time,
            method='nearest'
        )
        
        fit_results[q.name] = {
            "optimal_integration_time_ns": opt_time,
            "max_fidelity": float(fidelity_at_opt.values),
            "snr_at_optimal": float(snr_at_opt.values),
        }
    
    node.results["fit_results"] = fit_results
    
    # Print results to console for immediate feedback
    for q in qubits:
        print(f"\n{q.name}:")
        print(f"  Optimal integration time = {fit_results[q.name]['optimal_integration_time_ns']:.0f} ns")
        print(f"  Maximum fidelity = {fit_results[q.name]['max_fidelity']:.4f}")
        print(f"  SNR at optimal = {fit_results[q.name]['snr_at_optimal']:.2f}")
    
    # %% {Plotting}
    # =========================================================================
    # PLOT 1: SNR vs Integration Time
    # =========================================================================
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        qubit_name = qubit["qubit"]
        
        # Plot SNR scaling - should follow sqrt(t) behavior
        ds.SNR.loc[qubit].plot(ax=ax, x="integration_time", label="SNR", linewidth=2)
        
        # Mark optimal time with vertical line
        ax.axvline(
            fit_results[qubit_name]["optimal_integration_time_ns"],
            color="red",
            linestyle="--",
            label=f"Optimal: {fit_results[qubit_name]['optimal_integration_time_ns']:.0f} ns",
        )
        
        ax.set_xlabel("Integration Time [ns]")
        ax.set_ylabel("Signal-to-Noise Ratio")
        ax.set_title(f"{qubit_name}")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
    
    grid.fig.suptitle("SNR vs Integration Time (proportional to sqrt(t) scaling)")
    plt.tight_layout()
    plt.show()
    node.results["figure_snr"] = grid.fig
    
    # =========================================================================
    # PLOT 2: Fidelity vs Integration Time (Main Result)
    # =========================================================================
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        qubit_name = qubit["qubit"]
        
        # Plot assignment fidelity (without T1 decay)
        ds.F_assignment.loc[qubit].plot(
            ax=ax, x="integration_time", label="F_assignment (no T1)", 
            linewidth=2, linestyle="--", alpha=0.7, color="blue"
        )
        
        # Plot total measurement fidelity (with T1 decay) - the key result
        ds.F_measure.loc[qubit].plot(
            ax=ax, x="integration_time", label="F_measure (with T1)", 
            linewidth=3, color="green"
        )
        
        # Mark optimal time with vertical line
        ax.axvline(
            fit_results[qubit_name]["optimal_integration_time_ns"],
            color="red",
            linestyle="--",
            label=f"Optimal: {fit_results[qubit_name]['optimal_integration_time_ns']:.0f} ns",
        )
        
        # Mark maximum fidelity point with a red dot
        ax.plot(
            fit_results[qubit_name]["optimal_integration_time_ns"],
            fit_results[qubit_name]["max_fidelity"],
            'ro',
            markersize=10,
            label=f"Max F = {fit_results[qubit_name]['max_fidelity']:.4f}",
            zorder=10  # Ensure dot is on top
        )
        
        ax.set_xlabel("Integration Time [ns]")
        ax.set_ylabel("Fidelity")
        ax.set_title(f"{qubit_name}")
        ax.set_ylim([0, 1.05])
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
    
    grid.fig.suptitle("Readout Fidelity Optimization")
    plt.tight_layout()
    plt.show()
    node.results["figure_fidelity"] = grid.fig
    
    # =========================================================================
    # PLOT 3: IQ Blob Separation and Noise
    # =========================================================================
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        qubit_name = qubit["qubit"]
        
        # Create twin axis for plotting both separation and noise
        ax2 = ax.twinx()
        
        # Plot pointer separation |delta(alpha)| (left axis)
        line1 = ax.plot(
            ds.integration_time, 
            1e3 * ds.delta_alpha.loc[qubit],
            label="|delta(alpha)| separation", 
            linewidth=2, 
            color="blue"
        )
        
        # Plot noise sigma (right axis)
        line2 = ax2.plot(
            ds.integration_time,
            1e3 * ds.sigma.loc[qubit],
            label="Noise sigma",
            linewidth=2,
            color="orange"
        )
        
        # Mark optimal time
        ax.axvline(
            fit_results[qubit_name]["optimal_integration_time_ns"],
            color="red",
            linestyle="--",
            alpha=0.5,
        )
        
        ax.set_xlabel("Integration Time [ns]")
        ax.set_ylabel("Pointer Separation |delta(alpha)| [mV]", color="blue")
        ax2.set_ylabel("Noise sigma [mV]", color="orange")
        ax.set_title(f"{qubit_name}")
        ax.tick_params(axis='y', labelcolor="blue")
        ax2.tick_params(axis='y', labelcolor="orange")
        
       
        ax.grid(True, alpha=0.3)
    
    grid.fig.suptitle("IQ Blob Separation and Noise vs Integration Time")
    plt.tight_layout()
    plt.show()
    node.results["figure_separation"] = grid.fig
    
    # %% {Update_state}
    if node.parameters.load_data_id is None:
        # Update the integration time in the machine state
        with node.record_state_updates():
            for q in qubits:
                # Convert optimal time from ns to clock cycles (must be integer)
                optimal_time_cycles = int(fit_results[q.name]["optimal_integration_time_ns"] // 4)
                # Update the readout operation's length parameter
                q.resonator.operations["readout"].length = optimal_time_cycles
        
        # %% {Save_results}
        node.outcomes = {q.name: "successful" for q in qubits}
        node.results["initial_parameters"] = node.parameters.model_dump()
        node.machine = machine
        node.save()