#ifndef PLASMA_STATE_HPP
#ifndef DATA_MODELS_HPP
#ifndef PHYSICS_MODEL_HPP
#ifndef MONTE_CARLO_ENGINE_HPP
#ifndef PLASMA_ENGINE_HPP

#define PLASMA_STATE_HPP
#define DATA_MODELS_HPP
#define PHYSICS_MODEL_HPP
#define MONTE_CARLO_ENGINE_HPP
#define PLASMA_ENGINE_HPP

#include <vector>
#include <cmath>
#include <random>
#include <map>
#include <string>
#include <variant>
#include <optional>
#include <memory>
#include "plasma_state.hpp"
#include "data_models.hpp"
#include <functional>
#include "physics_model.hpp"
#include <future>
#include <atomic>
#include "monte_carlo_engine.hpp"

// Plasma State
struct PlasmaState {
    // Physical parameters
    double volume_leak;      // Volume of plasma leakage
    double permeability;     // Vascular permeability (0-1)
    double pressure;         // Hydrostatic pressure (mmHg)
    double albumin;          // Serum albumin (g/dL)
    double hematocrit;       // Hematocrit (%)
    double time;             // Current simulation time
    
    // Derived continuous risk metrics
    double leakage_index;    // Hct/Albumin ratio
    double risk_score;       // Continuous risk [0, 1]
    double capillary_pressure; // Capillary hydrostatic pressure
    double oncotic_pressure;   // Plasma oncotic pressure
    
    PlasmaState() 
        : volume_leak(0.0), permeability(0.0), pressure(100.0),
          albumin(3.5), hematocrit(45.0), time(0.0),
          leakage_index(0.0), risk_score(0.0),
          capillary_pressure(35.0), oncotic_pressure(25.0) {}
    
    // Compute derived quantities
    void compute_derived() {
        // Leakage index = Hematocrit / Albumin
        if (albumin > 0.0) {
            leakage_index = hematocrit / albumin;
        }
        
        // Starling forces for capillary exchange
        // Jv = Lp × [(Pc - Pi) - σ(πc - πi)]
        // Simplified: net filtration = permeability × (hydrostatic - oncotic)
        double net_filtration = permeability * (capillary_pressure - oncotic_pressure);
        volume_leak += net_filtration * 0.1;
        volume_leak = std::max(0.0, volume_leak);
        
        // Risk score based on continuous physiology
        // Higher leakage index + low albumin + high permeability = higher risk
        risk_score = compute_continuous_risk();
    }
    
    // Compute continuous risk score (not discrete classes)
    double compute_continuous_risk() const {
        // Combined risk from multiple continuous factors
        double risk = 0.0;
        
        // Risk from leakage index
        if (leakage_index > 0.0) {
            risk += std::min(1.0, leakage_index / 3.0) * 0.4;
        }
        
        // Risk from permeability
        risk += permeability * 0.25;
        
        // Risk from albumin (low albumin = high risk)
        if (albumin < 3.0) {
            risk += (3.0 - albumin) / 3.0 * 0.2;
        }
        
        // Risk from hematocrit (high = hemoconcentration = plasma leak)
        if (hematocrit > 50.0) {
            risk += (hematocrit - 50.0) / 50.0 * 0.15;
        }
        
        return std::min(1.0, std::max(0.0, risk));
    }
};

// Leakage Index Calculator - Standalone function for fast computation
class LeakageIndexCalculator {
public:
    // Calculate leakage index from albumin and hematocrit
    static double calculate(double albumin, double hematocrit) {
        if (albumin <= 0.0 || std::isnan(albumin) || std::isnan(hematokrit)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return hematocrit / albumin;
    }
    
    // Validate physiology status based on leakage index
    static std::string validate_physiology(double albumin, double hematocrit) {
        double idx = calculate(albumin, hematocrit);
        if (std::isnan(idx)) {
            return "Invalid";
        }
        if (idx < 1.5) {
            return "Normal";
        } else if (idx < 2.0) {
            return "Risiko Kebocoran";
        } else {
            return "Kebocoran Plasma";
        }
    }
    
    // Batch calculation for multiple samples
    static std::vector<double> calculate_batch(const std::vector<double>& albumin_values, 
                                             const std::vector<double>& hematocrit_values) {
        size_t n = std::min(albumin_values.size(), hematocrit_values.size());
        std::vector<double> results(n);
        
        for (size_t i = 0; i < n; ++i) {
            results[i] = calculate(albumin_values[i], hematocrit_values[i]);
        }
        
        return results;
    }
    
    // Risk assessment based on leakage index
    static std::string assess_risk(double leakage_index) {
        if (std::isnan(leakage_index)) {
            return "Unknown";
        }
        if (leakage_index < 1.5) {
            return "LOW";
        } else if (leakage_index < 2.0) {
            return "MODERATE";
        } else if (leakage_index < 2.5) {
            return "HIGH";
        } else {
            return "CRITICAL";
        }
    }
};

// Statistical Calculator - Fast statistical computations
class StatisticalCalculator {
public:
    // Normality tests
    static std::pair<std::string, double> shapiro_wilk_test(const std::vector<double>& data);
    static std::pair<std::string, double> kolmogorov_smirnov_test(const std::vector<double>& data);
    
    // Correlation analysis
    static std::tuple<double, double, std::string> pearson_correlation(
        const std::vector<double>& x, const std::vector<double>& y);
    static std::tuple<double, double, std::string> spearman_correlation(
        const std::vector<double>& x, const std::vector<double>& y);
    
    // Outlier detection
    static std::vector<bool> detect_outliers_iqr(const std::vector<double>& data, double multiplier = 1.5);
    
    // Summary statistics
    static std::map<std::string, double> compute_summary_stats(const std::vector<double>& data);
    
    // Batch processing for multiple variables
    static std::vector<std::map<std::string, double>> compute_batch_stats(
        const std::vector<std::vector<double>>& datasets);
};

    // State Transition
    // For Markov chain transitions between continuous states
    struct StateTransition {
        PlasmaState from_state;
        PlasmaState to_state;
        double transition_probability;
        double time_delta;
            
        StateTransition() : transition_probability(0.0), time_delta(0.0) {}
    };

    // Patient Trajectory
    // Full trajectory of a patient's plasma leakage evolution
    struct PatientTrajectory {
        std::vector<PlasmaState> states;
        std::vector<double> risk_history;
        std::vector<double> leakage_history;
        int patient_id;
        double initial_risk;
        double final_risk;
        double max_risk;
        
        PatientTrajectory() : patient_id(-1), initial_risk(0.0), final_risk(0.0), max_risk(0.0) {}
    };

#endif // Plasma State HPP

// Data Models
// Risk Level (for output compatibility)
enum class RiskLevel {
    LOW = 0,
    MODERATE = 1,
    HIGH = 2,
    CRITICAL = 3
};

    // Patient Data
    // Small patient data (< 200 points) from clinical records
    struct PatientData {
        int patient_id;
        std::vector<double> smallData;      // Original clinical measurements
        double mean;
        double sd;
        double baseline_value;
        double albumin;
        double hematocrit;
        double blood_pressure_systolic;
        double blood_pressure_diastolic;
        double pulse_rate;
        std::map<std::string, double> clinical_features;
        
        PatientData() 
            : patient_id(-1), mean(0.0), sd(0.0), baseline_value(0.0),
            albumin(3.5), hematocrit(45.0), 
            blood_pressure_systolic(120.0), blood_pressure_diastolic(80.0),
            pulse_rate(80.0) {}
    };

    // Simulated Data
    // Large simulated dataset (1 million points)
    struct SimulatedData {
        std::vector<double> simulated_leakage;
        std::vector<double> simulated_pressure;
        std::vector<double> simulated_permeability;
        std::vector<double> simulated_risk;
        std::vector<double> leakage_index_history;
        int source_patient_id;
        
        SimulatedData() : source_patient_id(-1) {}
    };

    // Simulation Parameters
    // Configuration for Monte Carlo simulation
    struct SimulationParameters {
        int monte_carlo_iterations;
        int time_steps;
        double dt;                    // Time step in hours
        double noise_std;
        bool use_parallel;
        int num_threads;
        bool use_onnx;
        std::string onnx_model_path;
        bool use_adaptive_sampling;    // Adaptive Monte Carlo
        double convergence_threshold;
        
        SimulationParameters() : 
            monte_carlo_iterations(1000000),  // 1 million as requested
            time_steps(10),
            dt(1.0),
            noise_std(0.1),
            use_parallel(true),
            num_threads(4),
            use_onnx(false),
            onnx_model_path("models/plasma_model.onnx"),
            use_adaptive_sampling(false),
            convergence_threshold(0.01) {}
    };

    // Simulation Result
    // Results from a single Monte Carlo run
    struct SimulationResult {
        // All values are continuous (not discrete classes)
        std::vector<double> leakage_values;
        std::vector<double> risk_scores;
        std::vector<double> pressures;
        std::vector<double> permeabilities;
        std::vector<double> leakage_indices;
        
        // Statistics
        double expected_leakage;
        double variance_leakage;
        double std_leakage;
        double percentile_5;
        double percentile_25;
        double percentile_50;
        double percentile_75;
        double percentile_95;
        double min_value;
        double max_value;
        double skewness;
        double kurtosis;
        
        // Risk analysis (continuous probabilities)
        double overall_risk_score;
        double probability_critical;
        double probability_high;
        double probability_moderate;
        double probability_low;
        std::string risk_level;
        
        // Metadata
        double processing_time_ms;
        int total_iterations;
        int effective_sample_size;
        
        SimulationResult() :
            expected_leakage(0.0), variance_leakage(0.0), std_leakage(0.0),
            percentile_5(0.0), percentile_25(0.0), percentile_50(0.0),
            percentile_75(0.0), percentile_95(0.0), min_value(0.0), max_value(0.0),
            skewness(0.0), kurtosis(0.0), overall_risk_score(0.0),
            probability_critical(0.0), probability_high(0.0),
            probability_moderate(0.0), probability_low(0.0),
            risk_level("UNKNOWN"), processing_time_ms(0.0),
            total_iterations(0), effective_sample_size(0) {}
    };

    // Risk Analysis Result
    // Comprehensive risk analysis output
    struct RiskAnalysisResult {
        // Continuous risk metrics
        double overall_risk_score;
        double expected_leakage;
        double variance_leakage;
        double percentile_5;
        double percentile_95;
        
        // Probability density estimation
        std::vector<double> probability_density;
        std::vector<double> density_bins;
        
        // Risk factors (continuous)
        std::map<std::string, double> risk_factors;
        
        // Risk level (categorical for compatibility)
        RiskLevel risk_level;
        std::string risk_level_str;
        
        // Confidence interval
        double ci_lower;
        double ci_upper;
        double confidence_level;
        
        // Risk curve (threshold vs probability)
        std::vector<std::pair<double, double>> risk_curve;
        
        RiskAnalysisResult() :
            overall_risk_score(0.0), expected_leakage(0.0), variance_leakage(0.0),
            percentile_5(0.0), percentile_95(0.0),
            risk_level(RiskLevel::LOW), risk_level_str("LOW"),
            ci_lower(0.0), ci_upper(0.0), confidence_level(0.95) {}
    };

    // Py Input Data
    // Input structure for Python binding
    struct PyInputData {
        std::vector<double> values;
        double mean;
        double sd;
        double baseline;
        int monte_carlo_iterations;
        bool use_onnx;
        std::string onnx_model_path;
        std::map<std::string, double> custom_params;
        
        PyInputData() : 
            mean(0.0), sd(0.0), baseline(0.0),
            monte_carlo_iterations(1000000),
            use_onnx(false),
            onnx_model_path("") {}
    };

    // Py Output Data  
    // Output structure for Python binding
    struct PyOutputData {
        double expected_leakage;
        double variance_leakage;
        double percentile_5;
        double percentile_25;
        double percentile_50;
        double percentile_75;
        double percentile_95;
        double min_value;
        double max_value;
        double skewness;
        double kurtosis;
        double overall_risk_score;
        double probability_critical;
        std::string risk_level;
        std::vector<double> leakage_samples;
        std::vector<double> risk_samples;
        std::map<std::string, double> risk_factors;
        std::string status;
        std::string message;
        double processing_time_ms;
        int total_iterations;
        
        PyOutputData() :
            expected_leakage(0.0), variance_leakage(0.0),
            percentile_5(0.0), percentile_25(0.0), percentile_50(0.0),
            percentile_75(0.0), percentile_95(0.0), min_value(0.0), max_value(0.0),
            skewness(0.0), kurtosis(0.0), overall_risk_score(0.0),
            probability_critical(0.0), risk_level("UNKNOWN"),
            status("SUCCESS"), message(""), processing_time_ms(0.0),
            total_iterations(0) {}
    };

#endif // Data Models HPP

// PHYSICS MODELS
constexpr double LEAK_RATE_COEFFICIENT = 0.1;
constexpr double HEALING_FACTOR_COEFFICIENT = 0.05;
constexpr double NOISE_STANDARD_DEVIATION = 0.01;
constexpr double CRITICAL_LEAKAGE_THRESHOLD = 2.5;
constexpr double HIGH_RISK_THRESHOLD = 1.5;
constexpr double MODERATE_RISK_THRESHOLD = 1.0;

    // PHYSICS MODEL CLASS
    // Simulates dynamic plasma leakage using physics-based approach
    // Continuous probability based on Starling forces and Markov chain
    class PhysicsModel {
    private:
        std::mt19937 rng;
        std::mutex rng_mutex;
        
        // Physics parameters
        double vascular_permeability_coef;  // Lp
        double reflection_coefficient;       // sigma
        double capillary_pressure;           // Pc
        double interstitial_pressure;        // Pi
        double plasma_oncotic_pressure;      // πc
        double interstitial_oncotic_pressure; // πi
        
        // Noise parameters for stochastic dynamics
        double pressure_noise_std;
        double permeability_noise_std;
        double volume_noise_std;
        
    public:
        PhysicsModel();
        ~PhysicsModel();
        
        // Initialize from patient data (small data < 200 points)
        void initialize_from_patient_data(const PatientData& patient);
        
        // Set custom physics parameters
        void set_parameters(double Lp, double sigma, double Pc, double Pi, 
                            double pi_c, double pi_i);
        
        // Simulate dynamics - Markov chain step
        // State(t) depends on State(t-1) with continuous probability
        PlasmaState simulate_step(const PlasmaState& current_state, double dt);
        
        // Full trajectory simulation (multiple time steps)
        std::vector<PlasmaState> simulate_trajectory(
            const PlasmaState& initial_state, 
            int num_steps, 
            double dt
        );
        
        // Compute transition probability for Markov chain
        // P(State(t+dt) | State(t)) - continuous probability
        double compute_transition_probability(
            const PlasmaState& from_state,
            const PlasmaState& to_state,
            double dt
        );
        
        // Apply Brownian noise to simulate stochastic dynamics
        PlasmaState add_stochastic_noise(const PlasmaState& state);
        
        // Compute net filtration using Starling equation
        // Jv = Lp × [(Pc - Pi) - σ(πc - πi)]
        double compute_filtration_rate(const PlasmaState& state) const;
        
        // Update state based on filtration and reabsorption
        PlasmaState update_state(const PlasmaState& state, double dt);
        
        // Risk score from physics model (continuous)
        double compute_risk_from_physics(const PlasmaState& state) const;
        
        // Leakage index computation
        double compute_leakage_index(double hematocrit, double albumin) const;
        
        // Set random seed for reproducibility
        void set_seed(unsigned int seed);
    };

    // MARKOV CHAIN MONTE CARLO (MCMC) FOR PLASMA STATE
    // Uses Metropolis-Hastings algorithm for sampling
    class MarkovChainMonteCarlo {
    private:
        PhysicsModel physics_model;
        std::mt19937 rng;
        
        // Proposal distribution parameters
        double proposal_std_leakage;
        double proposal_std_pressure;
        double proposal_std_permeability;
        
    public:
        MarkovChainMonteCarlo();
        ~MarkovChainMonteCarlo();
        
        // Initialize with patient data
        void initialize(const PatientData& patient);
        
        // Run MCMC sampling - generates continuous samples
        // NOT discrete classes like {1,2,3,4}
        std::vector<PlasmaState> sample(
            const PlasmaState& initial_state,
            int num_samples,
            int burn_in = 1000
        );
        
        // Metropolis-Hastings acceptance criterion
        double compute_acceptance_ratio(
            const PlasmaState& current,
            const PlasmaState& proposal
        );
        
        // Generate proposal state
        PlasmaState generate_proposal(const PlasmaState& current);
        
        // Compute likelihood
        double compute_likelihood(const PlasmaState& state);
        
        // Set proposal distribution parameters
        void set_proposal_std(double leakage, double pressure, double permeability);
    };

    // PLASMA LEAKAGE SIMULATOR
    // High-level interface for plasma leakage simulation
    class PlasmaLeakageSimulator {
    private:
        PhysicsModel physics;
        MarkovChainMonteCarlo mcmc;
        SimulationParameters params;
        
    public:
        PlasmaLeakageSimulator();
        ~PlasmaLeakageSimulator();
        
        void set_parameters(const SimulationParameters& p);
        const SimulationParameters& get_parameters() const;
        
        // Run simulation: small data -> 1 million conditions
        // Each patient trajectory is unique (not repeating patterns)
        SimulationResult run_simulation(const PatientData& patient_data);
        
        // Run parallel simulation
        SimulationResult run_parallel_simulation(const PatientData& patient_data);
        
        // Single trajectory with unique evolution
        PatientTrajectory simulate_single_trajectory(
            const PatientData& patient,
            int num_steps
        );
        
        // Analyze risk from simulation results
        RiskAnalysisResult analyze_risk(const SimulationResult& result);
        
        // Monte Carlo with adaptive sampling
        SimulationResult run_adaptive_monte_carlo(
            const PatientData& patient,
            double convergence_threshold = 0.01
        );
    };

#endif // Physich Model HPP

// MONTE CARLO ENGINE CLASS
    class MonteCarloEngine {
    private:
        SimulationParameters params;
        std::unique_ptr<PlasmaLeakageSimulator> simulator;
        
        // Thread management
        std::atomic<int> completed_iterations;
        std::mutex result_mutex;
        
    #ifdef USE_ONNX_RUNTIME
        // ONNX Runtime for inference
        void* ort_env;
        void* ort_session;
        bool onnx_loaded;
    #endif
        
    public:
        MonteCarloEngine();
        ~MonteCarloEngine();
        
        // Configuration
        void set_parameters(const SimulationParameters& p);
        const SimulationParameters& get_parameters() const;
        
        // Run Monte Carlo simulation
        SimulationResult run_monte_carlo(const PatientData& patient_data);
        
        // Parallel Monte Carlo with multiple threads
        SimulationResult run_parallel_monte_carlo(
            const PatientData& patient_data,
            int num_threads = 4
        );
        
        // Single simulation path (returns unique trajectory)
        PatientTrajectory run_single_trajectory(
            const PatientData& patient_data,
            int num_time_steps
        );
        
        // Statistics computation
        void compute_statistics(SimulationResult& result);
        
        // Risk analysis
        RiskAnalysisResult analyze_risk(const SimulationResult& result);
        
        // ONNX Model loading and inference
    #ifdef USE_ONNX_RUNTIME
        bool load_onnx_model(const std::string& model_path);
        double run_onnx_inference(const std::vector<float>& input_features);
    #endif
        
        // Progress tracking
        int get_completed_iterations() const;
        void reset_progress();
        
        // Get simulator
        PlasmaLeakageSimulator* get_simulator() const;
    };

    // PARALLEL MONTE CARLO WORKER
    // Function for parallel execution
    struct MCWorkerResult {
        std::vector<double> leakage_values;
        std::vector<double> risk_scores;
        std::vector<double> pressures;
        std::vector<double> permeabilities;
        int iterations_completed;
        double processing_time;
    };

    class MCWorker {
    private:
        SimulationParameters params;
        PatientData patient_data;
        int worker_id;
        
    public:
        MCWorker(const SimulationParameters& p, const PatientData& pd, int id)
            : params(p), patient_data(pd), worker_id(id) {}
        
        MCWorkerResult execute();
    };

    // DISTRIBUTION ESTIMATOR
    // Estimates continuous probability distribution from samples
    class DistributionEstimator {
    private:
        int num_bins;
        double bin_width;
        
    public:
        DistributionEstimator(int bins = 100);
        ~DistributionEstimator();
        
        // Kernel Density Estimation for continuous probability
        std::vector<double> kernel_density_estimation(
            const std::vector<double>& samples,
            double bandwidth
        ) const;
        
        // Histogram-based probability density
        std::pair<std::vector<double>, std::vector<double>> 
        compute_histogram(const std::vector<double>& samples) const;
        
        // Compute probability of exceeding threshold (continuous)
        double compute_exceedance_probability(
            const std::vector<double>& samples,
            double threshold
        ) const;
        
        // Compute confidence intervals
        std::pair<double, double> compute_confidence_interval(
            const std::vector<double>& samples,
            double confidence_level = 0.95
        ) const;
        
        // Effective sample size (for Monte Carlo convergence)
        double compute_ess(const std::vector<double>& samples) const;
    };

    // RESULT AGGREGATOR
    // Aggregates results from parallel workers
    class ResultAggregator {
    public:
        static SimulationResult aggregate(
            const std::vector<MCWorkerResult>& worker_results
        );
        
        static void compute_percentiles(
            const std::vector<double>& values,
            double& p5, double& p25, double& p50, 
            double& p75, double& p95
        );
        
        static void compute_higher_moments(
            const std::vector<double>& values,
            double mean,
            double& skewness,
            double& kurtosis
        );
    };

#endif // Monte Carlo Engine HPP

// PLASMA ENGINE CLASS
class PlasmaEngine {
private:
    MonteCarloEngine mc_engine;
    PhysicsModel physics_model;
    SimulationParameters params;
    
    // ONNX model path
    std::string onnx_model_path;
    bool use_onnx;
    
public:
    PlasmaEngine();
    ~PlasmaEngine();
    
    // Process patient data
    
    // Process from std::map (Python dict equivalent)
    std::map<std::string, std::variant<double, int, std::string, 
                                         std::vector<double>, 
                                         std::map<std::string, double>>>
    process(const std::map<std::string, double>& input_data);
    
    // Process from PatientData struct
    SimulationResult process_patient_data(const PatientData& patient_data);
    
    // Process from raw vectors (for Python numpy arrays)
    std::map<std::string, std::variant<double, int, std::string,
                                         std::vector<double>,
                                         std::map<std::string, double>>>
    process_vector(
        const std::vector<double>& values,
        int iterations = 1000000,
        bool use_parallel = true
    );
    
    // CONFIGURATION
    void set_iterations(int iterations);
    void set_time_steps(int steps);
    void set_parallel(bool use_parallel, int num_threads = 4);
    void set_onnx_model(const std::string& model_path);
    void enable_onnx(bool enable);
    
    // PARAMETER MANAGEMENT
    void set_parameter(const std::string& name, double value);
    double get_parameter(const std::string& name) const;
    std::map<std::string, double> get_all_parameters() const;
    
    // DIRECT MONTE CARLO ACCESS
    MonteCarloEngine* get_mc_engine() { return &mc_engine; }
    PhysicsModel* get_physics_model() { return &physics_model; }
    
    // UTILITY METHODS
    
    // Validate input data
    bool validate_input(const std::map<std::string, double>& data) const;
    
    // Convert risk score to risk level
    static RiskLevel risk_score_to_level(double risk_score);
    
    // Compute leakage index from clinical values
    static double compute_leakage_index(double hematocrit, double albumin);
    
    // Get risk interpretation
    static std::string get_risk_interpretation(RiskLevel level);
};

// STANDALONE FUNCTIONS

// Main entry point function (for Python binding)
std::map<std::string, std::variant<double, int, std::string,
                                     std::vector<double>,
                                     std::map<std::string, double>>>
run_plasma_simulation(
    const std::vector<double>& patient_data,
    int iterations = 1000000,
    int time_steps = 10,
    bool use_parallel = true,
    int num_threads = 4
);

// Convert result to Python-compatible format
PyOutputData convert_to_py_output(const SimulationResult& result);

// Create PatientData from map
PatientData create_patient_data(
    const std::map<std::string, double>& clinical_data,
    int patient_id = -1
);

#endif // Plasma Engine HPP