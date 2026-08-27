#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <variant>

#include "plasma_leakage.hpp"

namespace py = pybind11;

// Python bindings for PlasmaState
void bind_plasma_state(py::module& m) {
    py::class_<PlasmaState>(m, "PlasmaState")
        .def(py::init<>())
        .def_readwrite("volume_leak", &PlasmaState::volume_leak)
        .def_readwrite("permeability", &PlasmaState::permeability)
        .def_readwrite("pressure", &PlasmaState::pressure)
        .def_readwrite("albumin", &PlasmaState::albumin)
        .def_readwrite("hematocrit", &PlasmaState::hematocrit)
        .def_readwrite("time", &PlasmaState::time)
        .def_readwrite("leakage_index", &PlasmaState::leakage_index)
        .def_readwrite("risk_score", &PlasmaState::risk_score)
        .def("compute_derived", &PlasmaState::compute_derived)
        .def("compute_continuous_risk", &PlasmaState::compute_continuous_risk);
}

// Python bindings for Patient Data
void bind_patient_data(py::module& m) {
    py::class_<PatientData>(m, "PatientData")
        .def(py::init<>())
        .def_readwrite("patient_id", &PatientData::patient_id)
        .def_readwrite("smallData", &PatientData::smallData)
        .def_readwrite("mean", &PatientData::mean)
        .def_readwrite("sd", &PatientData::sd)
        .def_readwrite("baseline_value", &PatientData::baseline_value)
        .def_readwrite("albumin", &PatientData::albumin)
        .def_readwrite("hematocrit", &PatientData::hematocrit)
        .def_readwrite("blood_pressure_systolic", &PatientData::blood_pressure_systolic)
        .def_readwrite("blood_pressure_diastolic", &PatientData::blood_pressure_diastolic)
        .def_readwrite("pulse_rate", &PatientData::pulse_rate)
        .def_readwrite("clinical_features", &PatientData::clinical_features);
}

// Python bindings for Simulation Parameters
void bind_simulation_parameters(py::module& m) {
    py::class_<SimulationParameters>(m, "SimulationParameters")
        .def(py::init<>())
        .def_readwrite("monte_carlo_iterations", &SimulationParameters::monte_carlo_iterations)
        .def_readwrite("time_steps", &SimulationParameters::time_steps)
        .def_readwrite("dt", &SimulationParameters::dt)
        .def_readwrite("noise_std", &SimulationParameters::noise_std)
        .def_readwrite("use_parallel", &SimulationParameters::use_parallel)
        .def_readwrite("num_threads", &SimulationParameters::num_threads)
        .def_readwrite("use_onnx", &SimulationParameters::use_onnx)
        .def_readwrite("onnx_model_path", &SimulationParameters::onnx_model_path)
        .def_readwrite("use_adaptive_sampling", &SimulationParameters::use_adaptive_sampling)
        .def_readwrite("convergence_threshold", &SimulationParameters::convergence_threshold);
}

// Python bindings for Simulation Result
void bind_simulation_result(py::module& m) {
    py::class_<SimulationResult>(m, "SimulationResult")
        .def(py::init<>())
        .def_readwrite("leakage_values", &SimulationResult::leakage_values)
        .def_readwrite("risk_scores", &SimulationResult::risk_scores)
        .def_readwrite("pressures", &SimulationResult::pressures)
        .def_readwrite("permeabilities", &SimulationResult::permeabilities)
        .def_readwrite("leakage_indices", &SimulationResult::leakage_indices)
        .def_readwrite("expected_leakage", &SimulationResult::expected_leakage)
        .def_readwrite("variance_leakage", &SimulationResult::variance_leakage)
        .def_readwrite("std_leakage", &SimulationResult::std_leakage)
        .def_readwrite("percentile_5", &SimulationResult::percentile_5)
        .def_readwrite("percentile_25", &SimulationResult::percentile_25)
        .def_readwrite("percentile_50", &SimulationResult::percentile_50)
        .def_readwrite("percentile_75", &SimulationResult::percentile_75)
        .def_readwrite("percentile_95", &SimulationResult::percentile_95)
        .def_readwrite("min_value", &SimulationResult::min_value)
        .def_readwrite("max_value", &SimulationResult::max_value)
        .def_readwrite("skewness", &SimulationResult::skewness)
        .def_readwrite("kurtosis", &SimulationResult::kurtosis)
        .def_readwrite("overall_risk_score", &SimulationResult::overall_risk_score)
        .def_readwrite("probability_critical", &SimulationResult::probability_critical)
        .def_readwrite("probability_high", &SimulationResult::probability_high)
        .def_readwrite("probability_moderate", &SimulationResult::probability_moderate)
        .def_readwrite("probability_low", &SimulationResult::probability_low)
        .def_readwrite("risk_level", &SimulationResult::risk_level)
        .def_readwrite("processing_time_ms", &SimulationResult::processing_time_ms)
        .def_readwrite("total_iterations", &SimulationResult::total_iterations)
        .def_readwrite("effective_sample_size", &SimulationResult::effective_sample_size);
}

// Python bindings for Risk Analysis Result
void bind_risk_analysis_result(py::module& m) {
    py::class_<RiskAnalysisResult>(m, "RiskAnalysisResult")
        .def(py::init<>())
        .def_readwrite("overall_risk_score", &RiskAnalysisResult::overall_risk_score)
        .def_readwrite("expected_leakage", &RiskAnalysisResult::expected_leakage)
        .def_readwrite("variance_leakage", &RiskAnalysisResult::variance_leakage)
        .def_readwrite("percentile_5", &RiskAnalysisResult::percentile_5)
        .def_readwrite("percentile_95", &RiskAnalysisResult::percentile_95)
        .def_readwrite("probability_density", &RiskAnalysisResult::probability_density)
        .def_readwrite("density_bins", &RiskAnalysisResult::density_bins)
        .def_readwrite("risk_factors", &RiskAnalysisResult::risk_factors)
        .def_readwrite("risk_level", &RiskAnalysisResult::risk_level)
        .def_readwrite("risk_level_str", &RiskAnalysisResult::risk_level_str)
        .def_readwrite("ci_lower", &RiskAnalysisResult::ci_lower)
        .def_readwrite("ci_upper", &RiskAnalysisResult::ci_upper)
        .def_readwrite("confidence_level", &RiskAnalysisResult::confidence_level)
        .def_readwrite("risk_curve", &RiskAnalysisResult::risk_curve);
}

// Python bindings for Risk Level enum
void bind_risk_level(py::module& m) {
    py::enum_<RiskLevel>(m, "RiskLevel")
        .value("LOW", RiskLevel::LOW)
        .value("MODERATE", RiskLevel::MODERATE)
        .value("HIGH", RiskLevel::HIGH)
        .value("CRITICAL", RiskLevel::CRITICAL)
        .export_values();
}

// Python bindings for Physics Model
void bind_physics_model(py::module& m) {
    py::class_<PhysicsModel>(m, "PhysicsModel")
        .def(py::init<>())
        .def("initialize_from_patient_data", &PhysicsModel::initialize_from_patient_data)
        .def("set_parameters", &PhysicsModel::set_parameters)
        .def("simulate_step", &PhysicsModel::simulate_step)
        .def("simulate_trajectory", &PhysicsModel::simulate_trajectory)
        .def("add_stochastic_noise", &PhysicsModel::add_stochastic_noise)
        .def("compute_filtration_rate", &PhysicsModel::compute_filtration_rate)
        .def("update_state", &PhysicsModel::update_state)
        .def("compute_risk_from_physics", &PhysicsModel::compute_risk_from_physics)
        .def("compute_leakage_index", &PhysicsModel::compute_leakage_index)
        .def("set_seed", &PhysicsModel::set_seed);
}

// Python bindings for Plasma Leakage Simulator
void bind_plasma_simulator(py::module& m) {
    py::class_<PlasmaLeakageSimulator>(m, "PlasmaLeakageSimulator")
        .def(py::init<>())
        .def("set_parameters", &PlasmaLeakageSimulator::set_parameters)
        .def("get_parameters", &PlasmaLeakageSimulator::get_parameters)
        .def("run_simulation", &PlasmaLeakageSimulator::run_simulation)
        .def("run_parallel_simulation", &PlasmaLeakageSimulator::run_parallel_simulation)
        .def("simulate_single_trajectory", &PlasmaLeakageSimulator::simulate_single_trajectory)
        .def("analyze_risk", &PlasmaLeakageSimulator::analyze_risk)
        .def("run_adaptive_monte_carlo", &PlasmaLeakageSimulator::run_adaptive_monte_carlo);
}

// Python bindings for Monte Carlo Engine
void bind_monte_carlo_engine(py::module& m) {
    py::class_<MonteCarloEngine>(m, "MonteCarloEngine")
        .def(py::init<>())
        .def("set_parameters", &MonteCarloEngine::set_parameters)
        .def("get_parameters", &MonteCarloEngine::get_parameters)
        .def("run_monte_carlo", &MonteCarloEngine::run_monte_carlo)
        .def("run_parallel_monte_carlo", &MonteCarloEngine::run_parallel_monte_carlo)
        .def("run_single_trajectory", &MonteCarloEngine::run_single_trajectory)
        .def("compute_statistics", &MonteCarloEngine::compute_statistics)
        .def("analyze_risk", &MonteCarloEngine::analyze_risk)
        .def("get_completed_iterations", &MonteCarloEngine::get_completed_iterations)
        .def("reset_progress", &MonteCarloEngine::reset_progress)
        .def("get_simulator", &MonteCarloEngine::get_simulator, py::return_value_policy::reference_internal);
}

// Python bindings for Plasma Engine (Main API)
void bind_plasma_engine(py::module& m) {
    py::class_<PlasmaEngine>(m, "PlasmaEngine")
        .def(py::init<>())
        
        // Main processing methods
        .def("process", &PlasmaEngine::process)
        .def("process_patient_data", &PlasmaEngine::process_patient_data)
        .def("process_vector", &PlasmaEngine::process_vector)
        
        // Configuration
        .def("set_iterations", &PlasmaEngine::set_iterations)
        .def("set_time_steps", &PlasmaEngine::set_time_steps)
        .def("set_parallel", &PlasmaEngine::set_parallel)
        .def("set_onnx_model", &PlasmaEngine::set_onnx_model)
        .def("enable_onnx", &PlasmaEngine::enable_onnx)
        
        // Parameters
        .def("set_parameter", &PlasmaEngine::set_parameter)
        .def("get_parameter", &PlasmaEngine::get_parameter)
        .def("get_all_parameters", &PlasmaEngine::get_all_parameters)
        
        // Utility
        .def("validate_input", &PlasmaEngine::validate_input)
        .def_static("risk_score_to_level", &PlasmaEngine::risk_score_to_level)
        .def_static("compute_leakage_index", &PlasmaEngine::compute_leakage_index)
        .def_static("get_risk_interpretation", &PlasmaEngine::get_risk_interpretation)
        
        // Access to underlying engines
        .def("get_mc_engine", &PlasmaEngine::get_mc_engine, py::return_value_policy::reference_internal)
        .def("get_physics_model", &PlasmaEngine::get_physics_model, py::return_value_policy::reference_internal);
}

// Helper function to convert result to Python dict
py::dict simulation_result_to_dict(const SimulationResult& result) {
    py::dict output;
    
    // Statistics
    output["expected_leakage"] = result.expected_leakage;
    output["variance_leakage"] = result.variance_leakage;
    output["std_leakage"] = result.std_leakage;
    output["percentile_5"] = result.percentile_5;
    output["percentile_25"] = result.percentile_25;
    output["percentile_50"] = result.percentile_50;
    output["percentile_75"] = result.percentile_75;
    output["percentile_95"] = result.percentile_95;
    output["min_value"] = result.min_value;
    output["max_value"] = result.max_value;
    output["skewness"] = result.skewness;
    output["kurtosis"] = result.kurtosis;
    
    // Risk
    output["overall_risk_score"] = result.overall_risk_score;
    output["probability_critical"] = result.probability_critical;
    output["probability_high"] = result.probability_high;
    output["probability_moderate"] = result.probability_moderate;
    output["probability_low"] = result.probability_low;
    output["risk_level"] = result.risk_level;
    
    // Metadata
    output["processing_time_ms"] = result.processing_time_ms;
    output["total_iterations"] = result.total_iterations;
    output["effective_sample_size"] = result.effective_sample_size;
    
    // Samples (limited)
    int sample_size = std::min(1000, static_cast<int>(result.leakage_values.size()));
    py::list leakage_samples, risk_samples;
    for (int i = 0; i < sample_size; ++i) {
        leakage_samples.append(result.leakage_values[i]);
        risk_samples.append(result.risk_scores[i]);
    }
    output["leakage_samples"] = leakage_samples;
    output["risk_samples"] = risk_samples;
    
    return output;
}

// Main module definition
PYBIND11_MODULE(plasma_leakage, m) {
    m.doc() = "Plasma Leakage Monte Carlo Engine for Dengue Hemorrhagic Fever Analysis\n\n"
              "Features:\n"
              "- Monte Carlo simulation: Small data (<200) -> 1 million conditions\n"
              "- Continuous probability based on dynamic plasma leakage physics\n"
              "- Markov chain simulation with stochastic dynamics\n"
              "- ONNX Runtime integration for inference\n"
              "- Parallel processing support";
    
    // Bind enums first
    bind_risk_level(m);
    
    // Bind data structures
    bind_plasma_state(m);
    bind_patient_data(m);
    bind_simulation_parameters(m);
    bind_simulation_result(m);
    bind_risk_analysis_result(m);
    
    // Bind core classes
    bind_physics_model(m);
    bind_plasma_simulator(m);
    bind_monte_carlo_engine(m);
    bind_plasma_engine(m);
    
    // Bind Leakage Index Calculator
    py::class_<LeakageIndexCalculator>(m, "LeakageIndexCalculator")
        .def_static("calculate", &LeakageIndexCalculator::calculate,
                   "Calculate leakage index from albumin and hematocrit\n\n"
                   "Args:\n"
                   "    albumin: Albumin concentration (g/dL)\n"
                   "    hematocrit: Hematocrit percentage (%)\n\n"
                   "Returns:\n"
                   "    Leakage index (Hct/Albumin) or NaN if invalid",
                   py::arg("albumin"), py::arg("hematocrit"))
        .def_static("validate_physiology", &LeakageIndexCalculator::validate_physiology,
                   "Validate physiological status based on leakage index\n\n"
                   "Args:\n"
                   "    albumin: Albumin concentration (g/dL)\n"
                   "    hematocrit: Hematocrit percentage (%)\n\n"
                   "Returns:\n"
                   "    Status string: 'Normal', 'Risiko Kebocoran', 'Kebocoran Plasma', or 'Invalid'",
                   py::arg("albumin"), py::arg("hematocrit"))
        .def_static("calculate_batch", &LeakageIndexCalculator::calculate_batch,
                   "Calculate leakage indices for multiple samples\n\n"
                   "Args:\n"
                   "    albumin_values: Vector of albumin values\n"
                   "    hematocrit_values: Vector of hematocrit values\n\n"
                   "Returns:\n"
                   "    Vector of leakage indices",
                   py::arg("albumin_values"), py::arg("hematocrit_values"))
        .def_static("assess_risk", &LeakageIndexCalculator::assess_risk,
                   "Assess risk level from leakage index\n\n"
                   "Args:\n"
                   "    leakage_index: Calculated leakage index\n\n"
                   "Returns:\n"
                   "    Risk level: 'LOW', 'MODERATE', 'HIGH', 'CRITICAL', or 'Unknown'",
                   py::arg("leakage_index"));
    
    // Bind Statistical Calculator
    py::class_<StatisticalCalculator>(m, "StatisticalCalculator")
        .def_static("shapiro_wilk_test", &StatisticalCalculator::shapiro_wilk_test,
                   "Perform Shapiro-Wilk normality test\n\n"
                   "Args:\n"
                   "    data: Vector of numeric values\n\n"
                   "Returns:\n"
                   "    Tuple of (test_name, statistic)",
                   py::arg("data"))
        .def_static("kolmogorov_smirnov_test", &StatisticalCalculator::kolmogorov_smirnov_test,
                   "Perform Kolmogorov-Smirnov normality test\n\n"
                   "Args:\n"
                   "    data: Vector of numeric values\n\n"
                   "Returns:\n"
                   "    Tuple of (test_name, statistic)",
                   py::arg("data"))
        .def_static("pearson_correlation", &StatisticalCalculator::pearson_correlation,
                   "Calculate Pearson correlation coefficient\n\n"
                   "Args:\n"
                   "    x: First variable values\n"
                   "    y: Second variable values\n\n"
                   "Returns:\n"
                   "    Tuple of (correlation, p_value, method)",
                   py::arg("x"), py::arg("y"))
        .def_static("spearman_correlation", &StatisticalCalculator::spearman_correlation,
                   "Calculate Spearman correlation coefficient\n\n"
                   "Args:\n"
                   "    x: First variable values\n"
                   "    y: Second variable values\n\n"
                   "Returns:\n"
                   "    Tuple of (correlation, p_value, method)",
                   py::arg("x"), py::arg("y"))
        .def_static("detect_outliers_iqr", &StatisticalCalculator::detect_outliers_iqr,
                   "Detect outliers using IQR method\n\n"
                   "Args:\n"
                   "    data: Vector of numeric values\n"
                   "    multiplier: IQR multiplier (default: 1.5)\n\n"
                   "Returns:\n"
                   "    Vector of boolean values indicating outliers",
                   py::arg("data"), py::arg("multiplier") = 1.5)
        .def_static("compute_summary_stats", &StatisticalCalculator::compute_summary_stats,
                   "Compute summary statistics\n\n"
                   "Args:\n"
                   "    data: Vector of numeric values\n\n"
                   "Returns:\n"
                   "    Dictionary with statistical measures",
                   py::arg("data"))
        .def_static("compute_batch_stats", &StatisticalCalculator::compute_batch_stats,
                   "Compute statistics for multiple datasets\n\n"
                   "Args:\n"
                   "    datasets: Vector of data vectors\n\n"
                   "Returns:\n"
                   "    Vector of statistics dictionaries",
                   py::arg("datasets"));
    
    // Standalone functions
    m.def("run_plasma_simulation", &run_plasma_simulation,
          "Run plasma leakage Monte Carlo simulation\n\n"
          "Args:\n"
          "    patient_data: Vector of [baseline, permeability, pressure, albumin, hematocrit]\n"
          "    iterations: Number of Monte Carlo iterations (default: 1,000,000)\n"
          "    time_steps: Number of time steps per simulation\n"
          "    use_parallel: Use parallel processing\n"
          "    num_threads: Number of threads for parallel processing\n\n"
          "Returns:\n"
          "    Dictionary with simulation results and risk analysis",
          py::arg("patient_data"),
          py::arg("iterations") = 1000000,
          py::arg("time_steps") = 10,
          py::arg("use_parallel") = true,
          py::arg("num_threads") = 4);
    
    m.def("create_patient_data", &create_patient_data,
          "Create PatientData from Python dictionary",
          py::arg("clinical_data"),
          py::arg("patient_id") = -1);
    
    m.def("convert_to_py_output", &convert_to_py_output,
          "Convert SimulationResult to PyOutputData",
          py::arg("result"));
}