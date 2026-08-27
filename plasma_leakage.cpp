#include "plasma_state.hpp"
#include "data_models.hpp"
#include "physics_model.hpp"
#include "monte_carlo_engine.hpp"
#include "plasma_engine.hpp"
#include <cmath>
#include <algorithm>
#include <random>
#include <numeric>
#include <chrono>

// Main entry point for standalone compilation test
int main() {
    plasma::Plasma Engine engine;
    
    // Create sample patient data
    std::map<std::string, double> input;
    input["albumin"] = 3.0;
    input["hematocrit"] = 50.0;
    input["bp_systolic"] = 100.0;
    input["bp_diastolic"] = 70.0;
    input["baseline"] = 1.5;
    input["permeability"] = 0.2;
    
    // Validate input
    if (!engine.validate_input(input)) {
        std::cerr << "Invalid input data" << std::endl;
        return 1;
    }
    
    // Set parameters
    engine.set_iterations(1000);  // Test with smaller number
    engine.set_time_steps(10);
    engine.set_parallel(true, 4);
    
    // Run simulation
    auto result = engine.process(input);
    
    std::cout << "Simulation completed successfully!" << std::endl;
    std::cout << "Expected leakage: " << std::get<double>(result.at("expected_leakage")) << std::endl;
    std::cout << "Risk level: " << std::get<std::string>(result.at("risk_level")) << std::endl;
    
    return 0;
}

// PHYSICS MODEL IMPLEMENTATION
PhysicsModel::PhysicsModel() 
    : vascular_permeability_coef(0.01),
      reflection_coefficient(0.8),
      capillary_pressure(35.0),
      interstitial_pressure(0.0),
      plasma_oncotic_pressure(25.0),
      interstitial_oncotic_pressure(5.0),
      pressure_noise_std(2.0),
      permeability_noise_std(0.01),
      volume_noise_std(0.1)
{
    rng = std::mt19937(std::random_device{}());
}

PhysicsModel::~PhysicsModel() {}

void PhysicsModel::initialize_from_patient_data(const PatientData& patient) {
    // Set initial conditions from patient data
    capillary_pressure = patient.blood_pressure_systolic * 0.3; // Approx
    plasma_oncotic_pressure = patient.albumin * 7; // Approx: albumin × 7
    
    rng = std::mt19937(std::random_device{}());
}

void PhysicsModel::set_parameters(double Lp, double sigma, double Pc, double Pi, 
                                  double pi_c, double pi_i) {
    vascular_permeability_coef = Lp;
    reflection_coefficient = sigma;
    capillary_pressure = Pc;
    interstitial_pressure = Pi;
    plasma_oncotic_pressure = pi_c;
    interstitial_oncotic_pressure = pi_i;
}

PlasmaState PhysicsModel::simulate_step(const PlasmaState& current_state, double dt) {
    std::lock_guard<std::mutex> lock(rng_mutex);
    
    // Compute filtration using Starling equation
    double Jv = compute_filtration_rate(current_state);
    
    // Update state
    PlasmaState next_state = current_state;
    next_state.volume_leak = current_state.volume_leak + Jv * dt;
    next_state.volume_leak = std::max(0.0, next_state.volume_leak);
    
    // Add stochastic noise (Brownian motion)
    std::normal_distribution<> dis_pressure(0.0, pressure_noise_std);
    std::normal_distribution<> dis_perm(0.0, permeability_noise_std);
    std::normal_distribution<> dis_volume(0.0, volume_noise_std);
    
    next_state.pressure = current_state.pressure + dis_pressure(rng) * dt;
    next_state.permeability = current_state.permeability + dis_perm(rng) * dt;
    next_state.permeability = std::max(0.001, std::min(1.0, next_state.permeability));
    next_state.volume_leak += dis_volume(rng);
    next_state.volume_leak = std::max(0.0, next_state.volume_leak);
    
    // Compute derived quantities
    next_state.compute_derived();
    next_state.time = current_state.time + dt;
    
    return next_state;
}

std::vector<PlasmaState> PhysicsModel::simulate_trajectory(
    const PlasmaState& initial_state, 
    int num_steps, 
    double dt
) {
    std::vector<PlasmaState> trajectory;
    trajectory.reserve(num_steps + 1);
    
    PlasmaState current = initial_state;
    trajectory.push_back(current);
    
    for (int i = 0; i < num_steps; ++i) {
        current = simulate_step(current, dt);
        trajectory.push_back(current);
    }
    
    return trajectory;
}

double PhysicsModel::compute_transition_probability(
    const PlasmaState& from_state,
    const PlasmaState& to_state,
    double dt
) {
    // Continuous transition probability based on physics    
    double mean_leakage_change = compute_filtration_rate(from_state) * dt;
    double std_leakage_change = volume_noise_std * std::sqrt(dt);
    
    double leakage_diff = to_state.volume_leak - from_state.volume_leak;
    
    // Gaussian probability density
    double z = (leakage_diff - mean_leakage_change) / std_leakage_change;
    double probability = (1.0 / (std_leakage_change * std::sqrt(2.0 * M_PI))) 
                       * std::exp(-0.5 * z * z);
    
    return probability;
}

PlasmaState PhysicsModel::add_stochastic_noise(const PlasmaState& state) {
    std::lock_guard<std::mutex> lock(rng_mutex);
    
    std::normal_distribution<> dis_pressure(0.0, pressure_noise_std);
    std::normal_distribution<> dis_perm(0.0, permeability_noise_std);
    std::normal_distribution<> dis_volume(0.0, volume_noise_std);
    std::normal_distribution<> dis_albumin(0.0, 0.1);
    std::normal_distribution<> dis_hct(0.0, 1.0);
    
    PlasmaState noisy_state = state;
    noisy_state.pressure += dis_pressure(rng);
    noisy_state.permeability += dis_perm(rng);
    noisy_state.volume_leak += dis_volume(rng);
    noisy_state.albumin += dis_albumin(rng);
    noisy_state.hematocrit += dis_hct(rng);
    
    // Clamp values to physiological ranges
    noisy_state.albumin = std::max(0.5, std::min(6.0, noisy_state.albumin));
    noisy_state.hematocrit = std::max(10.0, std::min(70.0, noisy_state.hematocrit));
    noisy_state.permeability = std::max(0.001, std::min(1.0, noisy_state.permeability));
    noisy_state.pressure = std::max(40.0, std::min(200.0, noisy_state.pressure));
    noisy_state.volume_leak = std::max(0.0, noisy_state.volume_leak);
    
    noisy_state.compute_derived();
    
    return noisy_state;
}

double PhysicsModel::compute_filtration_rate(const PlasmaState& state) const {
    // Starling equation: Jv = Lp × [(Pc - Pi) - σ(πc - πi)]
    double effective_pressure = (capillary_pressure - interstitial_pressure) 
                               - reflection_coefficient 
                               * (plasma_oncotic_pressure - interstitial_oncotic_pressure);
    
    double Jv = vascular_permeability_coef * state.permeability * effective_pressure;
    
    return Jv;
}

PlasmaState PhysicsModel::update_state(const PlasmaState& state, double dt) {
    double Jv = compute_filtration_rate(state);
    
    PlasmaState new_state = state;
    new_state.volume_leak += Jv * dt;
    new_state.volume_leak = std::max(0.0, new_state.volume_leak);
    
    // Healing factor
    double healing = HEALING_FACTOR_COEFFICIENT * (1.0 - state.permeability);
    new_state.volume_leak -= healing * dt;
    new_state.volume_leak = std::max(0.0, new_state.volume_leak);
    
    new_state.compute_derived();
    new_state.time = state.time + dt;
    
    return new_state;
}

double PhysicsModel::compute_risk_from_physics(const PlasmaState& state) const {
    // Continuous risk based on physiology
    double risk = 0.0;
    
    // Risk from leakage index
    double li = compute_leakage_index(state.hematocrit, state.albumin);
    risk += std::min(1.0, li / 4.0) * 0.35;
    
    // Risk from permeability
    risk += state.permeability * 0.25;
    
    // Risk from filtration rate
    double Jv = compute_filtration_rate(state);
    risk += std::min(1.0, std::abs(Jv) / 10.0) * 0.2;
    
    // Risk from low albumin (hypoalbuminemia)
    if (state.albumin < 3.0) {
        risk += (3.0 - state.albumin) / 3.0 * 0.1;
    }
    
    // Risk from volume leakage
    risk += std::min(1.0, state.volume_leak / 5.0) * 0.1;
    
    return std::min(1.0, std::max(0.0, risk));
}

double PhysicsModel::compute_leakage_index(double hematocrit, double albumin) const {
    if (albumin <= 0.0) return 0.0;
    return hematocrit / albumin;
}

void PhysicsModel::set_seed(unsigned int seed) {
    rng = std::mt19937(seed);
}

// MARKOV CHAIN MONTE CARLO IMPLEMENTATION
MarkovChainMonteCarlo::MarkovChainMonteCarlo()
    : proposal_std_leakage(0.1),
      proposal_std_pressure(2.0),
      proposal_std_permeability(0.01)
{
    rng = std::mt19937(std::random_device{}());
}

MarkovChainMonteCarlo::~MarkovChainMonteCarlo() {}

void MarkovChainMonteCarlo::initialize(const PatientData& patient) {
    physics_model.initialize_from_patient_data(patient);
    rng = std::mt19937(std::random_device{}());
}

std::vector<PlasmaState> MarkovChainMonteCarlo::sample(
    const PlasmaState& initial_state,
    int num_samples,
    int burn_in
) {
    std::vector<PlasmaState> samples;
    samples.reserve(num_samples);
    
    PlasmaState current = initial_state;
    
    // Burn-in phase
    for (int i = 0; i < burn_in; ++i) {
        PlasmaState proposal = generate_proposal(current);
        double alpha = compute_acceptance_ratio(current, proposal);
        
        std::uniform_real_distribution<> dis(0.0, 1.0);
        if (dis(rng) < alpha) {
            current = proposal;
        }
    }
    
    // Sampling phase
    for (int i = 0; i < num_samples; ++i) {
        PlasmaState proposal = generate_proposal(current);
        double alpha = compute_acceptance_ratio(current, proposal);
        
        std::uniform_real_distribution<> dis(0.0, 1.0);
        if (dis(rng) < alpha) {
            current = proposal;
        }
        samples.push_back(current);
    }
    
    return samples;
}

double MarkovChainMonteCarlo::compute_acceptance_ratio(
    const PlasmaState& current,
    const PlasmaState& proposal
) {
    double likelihood_current = compute_likelihood(current);
    double likelihood_proposal = compute_likelihood(proposal);
    
    // Metropolis-Hastings ratio
    double ratio = likelihood_proposal / likelihood_current;
    
    // Prior ratio (uniform prior, so = 1)
    double prior_ratio = 1.0;
    
    // Proposal ratio (symmetric, so = 1)
    double proposal_ratio = 1.0;
    
    return std::min(1.0, ratio * prior_ratio * proposal_ratio);
}

PlasmaState MarkovChainMonteCarlo::generate_proposal(const PlasmaState& current) {
    std::normal_distribution<> dis_leakage(0.0, proposal_std_leakage);
    std::normal_distribution<> dis_pressure(0.0, proposal_std_pressure);
    std::normal_distribution<> dis_perm(0.0, proposal_std_permeability);
    
    PlasmaState proposal = current;
    proposal.volume_leak += dis_leakage(rng);
    proposal.pressure += dis_pressure(rng);
    proposal.permeability += dis_perm(rng);
    
    // Clamp to valid ranges
    proposal.volume_leak = std::max(0.0, proposal.volume_leak);
    proposal.permeability = std::max(0.001, std::min(1.0, proposal.permeability));
    proposal.pressure = std::max(40.0, std::min(200.0, proposal.pressure));
    proposal.albumin = std::max(0.5, std::min(6.0, proposal.albumin));
    proposal.hematocrit = std::max(10.0, std::min(70.0, proposal.hematocrit));
    
    proposal.compute_derived();
    
    return proposal;
}

double MarkovChainMonteCarlo::compute_likelihood(const PlasmaState& state) {
    // Likelihood based on physics model
    double risk = physics_model.compute_risk_from_physics(state);
    
    // Likelihood: lower risk = higher likelihood (healthy prior)
    double likelihood = 1.0 - risk;
    
    // Add constraint penalties
    if (state.permeability < 0.001 || state.permeability > 1.0) {
        likelihood *= 0.01;
    }
    if (state.pressure < 40.0 || state.pressure > 200.0) {
        likelihood *= 0.01;
    }
    
    return likelihood;
}

void MarkovChainMonteCarlo::set_proposal_std(
    double leakage, 
    double pressure, 
    double permeability
) {
    proposal_std_leakage = leakage;
    proposal_std_pressure = pressure;
    proposal_std_permeability = permeability;
}

// PLASMA LEAKAGE SIMULATOR IMPLEMENTATION
PlasmaLeakageSimulator::PlasmaLeakageSimulator() {
    rng = std::mt19937(std::random_device{}());
}

PlasmaLeakageSimulator::~PlasmaLeakageSimulator() {}

void PlasmaLeakageSimulator::set_parameters(const SimulationParameters& p) {
    params = p;
}

const SimulationParameters& PlasmaLeakageSimulator::get_parameters() const {
    return params;
}

SimulationResult PlasmaLeakageSimulator::run_simulation(
    const PatientData& patient_data
) {
    SimulationResult result;
    
    // Initialize physics model
    physics.initialize_from_patient_data(patient_data);
    
    // Prepare initial state
    PlasmaState initial_state;
    initial_state.volume_leak = patient_data.baseline_value;
    initial_state.permeability = 0.1;
    initial_state.pressure = patient_data.blood_pressure_systolic;
    initial_state.albumin = patient_data.albumin;
    initial_state.hematocrit = patient_data.hematocrit;
    initial_state.compute_derived();
    
    // Monte Carlo iterations - each path is unique
    for (int i = 0; i < params.monte_carlo_iterations; ++i) {
        // Generate unique initial conditions for each iteration
        PlasmaState state = initial_state;
        
        // Add patient-specific noise (variability)
        state = physics.add_stochastic_noise(state);
        
        // Simulate through time steps - unique trajectory
        for (int t = 0; t < params.time_steps; ++t) {
            state = physics.simulate_step(state, params.dt);
        }
        
        // Store results
        result.leakage_values.push_back(state.volume_leak);
        result.risk_scores.push_back(state.risk_score);
        result.pressures.push_back(state.pressure);
        result.permeabilities.push_back(state.permeability);
        result.leakage_indices.push_back(state.leakage_index);
    }
    
    result.total_iterations = params.monte_carlo_iterations;
    
    return result;
}

SimulationResult PlasmaLeakageSimulator::run_parallel_simulation(
    const PatientData& patient_data
) {
    // For parallel implementation, delegate to MonteCarloEngine
    return run_simulation(patient_data);
}

PatientTrajectory PlasmaLeakageSimulator::simulate_single_trajectory(
    const PatientData& patient,
    int num_steps
) {
    PatientTrajectory trajectory;
    trajectory.patient_id = patient.patient_id;
    
    physics.initialize_from_patient_data(patient);
    
    PlasmaState current;
    current.volume_leak = patient.baseline_value;
    current.permeability = 0.1;
    current.pressure = patient.blood_pressure_systolic;
    current.albumin = patient.albumin;
    current.hematocrit = patient.hematocrit;
    current.compute_derived();
    
    trajectory.initial_risk = current.risk_score;
    
    for (int t = 0; t < num_steps; ++t) {
        current = physics.simulate_step(current, params.dt);
        trajectory.states.push_back(current);
        trajectory.risk_history.push_back(current.risk_score);
        trajectory.leakage_history.push_back(current.volume_leak);
        
        trajectory.max_risk = std::max(trajectory.max_risk, current.risk_score);
    }
    
    trajectory.final_risk = current.risk_score;
    
    return trajectory;
}

RiskAnalysisResult PlasmaLeakageSimulator::analyze_risk(
    const SimulationResult& result
) {
    RiskAnalysisResult risk_result;
    
    if (result.leakage_values.empty()) return risk_result;
    
    const auto& values = result.leakage_values;
    const auto& risks = result.risk_scores;
    size_t n = values.size();
    
    // Compute statistics
    double sum = std::accumulate(values.begin(), values.end(), 0.0);
    risk_result.expected_leakage = sum / n;
    
    double sq_sum = 0.0;
    for (const auto& v : values) {
        sq_sum += (v - risk_result.expected_leakage) * (v - risk_result.expected_leakage);
    }
    risk_result.variance_leakage = sq_sum / n;
    
    // Percentiles
    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end());
    
    auto percentile = [&sorted, n](double p) -> double {
        double index = p / 100.0 * (n - 1);
        size_t lower = static_cast<size_t>(std::floor(index));
        size_t upper = static_cast<size_t>(std::ceil(index));
        if (lower == upper) return sorted[lower];
        return sorted[lower] * (upper - index) + sorted[upper] * (index - lower);
    };
    
    risk_result.percentile_5 = percentile(5.0);
    risk_result.percentile_95 = percentile(95.0);
    
    // Confidence interval
    double std = std::sqrt(risk_result.variance_leakage);
    double se = std / std::sqrt(n);
    risk_result.ci_lower = risk_result.expected_leakage - 1.96 * se;
    risk_result.ci_upper = risk_result.expected_leakage + 1.96 * se;
    
    // Risk analysis (continuous)
    double risk_sum = std::accumulate(risks.begin(), risks.end(), 0.0);
    risk_result.overall_risk_score = risk_sum / n;
    
    // Count continuous risk levels
    int count_critical = 0, count_high = 0, count_moderate = 0, count_low = 0;
    for (const auto& r : risks) {
        if (r >= 0.75) count_critical++;
        else if (r >= 0.5) count_high++;
        else if (r >= 0.25) count_moderate++;
        else count_low++;
    }
    
    risk_result.probability_critical = static_cast<double>(count_critical) / n;
    risk_result.probability_high = static_cast<double>(count_high) / n;
    risk_result.probability_moderate = static_cast<double>(count_moderate) / n;
    risk_result.probability_low = static_cast<double>(count_low) / n;
    
    // Determine risk level
    if (risk_result.probability_critical > 0.3) {
        risk_result.risk_level = RiskLevel::CRITICAL;
        risk_result.risk_level_str = "CRITICAL";
    } else if (risk_result.probability_high > 0.4) {
        risk_result.risk_level = RiskLevel::HIGH;
        risk_result.risk_level_str = "HIGH";
    } else if (risk_result.probability_moderate > 0.4) {
        risk_result.risk_level = RiskLevel::MODERATE;
        risk_result.risk_level_str = "MODERATE";
    } else {
        risk_result.risk_level = RiskLevel::LOW;
        risk_result.risk_level_str = "LOW";
    }
    
    // Risk factors
    risk_result.risk_factors["mean_leakage"] = risk_result.expected_leakage;
    risk_result.risk_factors["std_leakage"] = std;
    risk_result.risk_factors["mean_risk"] = risk_result.overall_risk_score;
    
    // Risk curve
    for (double threshold = 0.0; threshold <= 1.0; threshold += 0.1) {
        int count = 0;
        for (const auto& r : risks) {
            if (r >= threshold) count++;
        }
        risk_result.risk_curve.push_back({threshold, static_cast<double>(count) / n});
    }
    
    return risk_result;
}

SimulationResult PlasmaLeakageSimulator::run_adaptive_monte_carlo(
    const PatientData& patient,
    double convergence_threshold
) {
    SimulationResult result;
    int batch_size = 10000;
    int max_iterations = params.monte_carlo_iterations;
    
    // Initial simulation
    SimulationParameters batch_params = params;
    batch_params.monte_carlo_iterations = batch_size;
    
    SimulationResult batch_result;
    physics.initialize_from_patient_data(patient);
    
    PlasmaState initial_state;
    initial_state.volume_leak = patient.baseline_value;
    initial_state.permeability = 0.1;
    initial_state.pressure = patient.blood_pressure_systolic;
    initial_state.albumin = patient.albumin;
    initial_state.hematocrit = patient.hematocrit;
    initial_state.compute_derived();
    
    for (int i = 0; i < batch_size; ++i) {
        PlasmaState state = initial_state;
        state = physics.add_stochastic_noise(state);
        
        for (int t = 0; t < params.time_steps; ++t) {
            state = physics.simulate_step(state, params.dt);
        }
        
        batch_result.leakage_values.push_back(state.volume_leak);
        batch_result.risk_scores.push_back(state.risk_score);
    }
    
    result = batch_result;
    
    // Adaptive sampling
    double current_mean = result.expected_leakage;
    double current_var = result.variance_leakage;
    
    for (int total = batch_size; total < max_iterations; total += batch_size) {
        batch_result.leakage_values.clear();
        batch_result.risk_scores.clear();
        
        for (int i = 0; i < batch_size; ++i) {
            PlasmaState state = initial_state;
            state = physics.add_stochastic_noise(state);
            
            for (int t = 0; t < params.time_steps; ++t) {
                state = physics.simulate_step(state, params.dt);
            }
            
            batch_result.leakage_values.push_back(state.volume_leak);
            batch_result.risk_scores.push_back(state.risk_score);
        }
        
        // Merge results
        result.leakage_values.insert(result.leakage_values.end(),
                                     batch_result.leakage_values.begin(),
                                     batch_result.leakage_values.end());
        result.risk_scores.insert(result.risk_scores.end(),
                                  batch_result.risk_scores.begin(),
                                  batch_result.risk_scores.end());
        
        // Compute new statistics
        double new_sum = std::accumulate(result.leakage_values.begin(), 
                                         result.leakage_values.end(), 0.0);
        double new_mean = new_sum / result.leakage_values.size();
        
        double new_sq_sum = 0.0;
        for (const auto& v : result.leakage_values) {
            new_sq_sum += (v - new_mean) * (v - new_mean);
        }
        double new_var = new_sq_sum / result.leakage_values.size();
        
        // Check convergence
        double rel_change = std::abs(new_mean - current_mean) / (std::abs(current_mean) + 1e-10);
        
        if (rel_change < convergence_threshold && total > 50000) {
            std::cout << "Converged at " << total << " iterations" << std::endl;
            break;
        }
        
        current_mean = new_mean;
        current_var = new_var;
    }
    
    result.total_iterations = result.leakage_values.size();
    
    return result;
}

// End Physics Model CPP

// MONTE CARLO ENGINE IMPLEMENTATION
MonteCarloEngine::MonteCarloEngine() 
    : completed_iterations(0)
#ifdef USE_ONNX_RUNTIME
    , ort_env(nullptr), ort_session(nullptr), onnx_loaded(false)
#endif
{
    simulator = std::make_unique<PlasmaLeakageSimulator>();
}

MonteCarloEngine::~MonteCarloEngine() {}

void MonteCarloEngine::set_parameters(const SimulationParameters& p) {
    params = p;
    simulator->set_parameters(p);
}

const SimulationParameters& MonteCarloEngine::get_parameters() const {
    return params;
}

SimulationResult MonteCarloEngine::run_monte_carlo(const PatientData& patient_data) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Run simulation
    SimulationResult result = simulator->run_simulation(patient_data);
    
    // Compute statistics
    compute_statistics(result);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.processing_time_ms = std::chrono::duration<double, std::milli>(
        end_time - start_time
    ).count();
    
    return result;
}

SimulationResult MonteCarloEngine::run_parallel_monte_carlo(
    const PatientData& patient_data,
    int num_threads
) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    SimulationParameters local_params = params;
    local_params.use_parallel = true;
    local_params.num_threads = num_threads;
    
    int iterations_per_thread = local_params.monte_carlo_iterations / num_threads;
    int remainder = local_params.monte_carlo_iterations % num_threads;
    
    std::vector<std::future<MCWorkerResult>> futures;
    
    for (int t = 0; t < num_threads; ++t) {
        SimulationParameters thread_params = local_params;
        thread_params.monte_carlo_iterations = iterations_per_thread + (t < remainder ? 1 : 0);
        
        MCWorker worker(thread_params, patient_data, t);
        futures.push_back(std::async(std::launch::async, [&worker]() {
            return worker.execute();
        }));
    }
    
    // Collect results
    std::vector<MCWorkerResult> worker_results;
    for (auto& future : futures) {
        worker_results.push_back(future.get());
    }
    
    // Aggregate results
    SimulationResult result = ResultAggregator::aggregate(worker_results);
    
    // Compute statistics
    compute_statistics(result);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.processing_time_ms = std::chrono::duration<double, std::milli>(
        end_time - start_time
    ).count();
    
    return result;
}

PatientTrajectory MonteCarloEngine::run_single_trajectory(
    const PatientData& patient_data,
    int num_time_steps
) {
    return simulator->simulate_single_trajectory(patient_data, num_time_steps);
}

void MonteCarloEngine::compute_statistics(SimulationResult& result) {
    if (result.leakage_values.empty()) return;
    
    const auto& values = result.leakage_values;
    size_t n = values.size();
    
    // Basic statistics
    double sum = std::accumulate(values.begin(), values.end(), 0.0);
    result.expected_leakage = sum / n;
    
    double sq_sum = 0.0;
    for (const auto& v : values) {
        sq_sum += (v - result.expected_leakage) * (v - result.expected_leakage);
    }
    result.variance_leakage = sq_sum / n;
    result.std_leakage = std::sqrt(result.variance_leakage);
    
    // Min/Max
    result.min_value = *std::min_element(values.begin(), values.end());
    result.max_value = *std::max_element(values.begin(), values.end());
    
    // Percentiles
    ResultAggregator::compute_percentiles(
        values, 
        result.percentile_5, result.percentile_25, result.percentile_50,
        result.percentile_75, result.percentile_95
    );
    
    // Higher moments
    ResultAggregator::compute_higher_moments(
        values,
        result.expected_leakage,
        result.skewness,
        result.kurtosis
    );
    
    // Risk analysis
    if (!result.risk_scores.empty()) {
        const auto& risks = result.risk_scores;
        double risk_sum = std::accumulate(risks.begin(), risks.end(), 0.0);
        result.overall_risk_score = risk_sum / n;
        
        // Count risk levels
        int count_critical = 0, count_high = 0, count_moderate = 0, count_low = 0;
        for (const auto& r : risks) {
            if (r >= 0.75) count_critical++;
            else if (r >= 0.5) count_high++;
            else if (r >= 0.25) count_moderate++;
            else count_low++;
        }
        
        result.probability_critical = static_cast<double>(count_critical) / n;
        result.probability_high = static_cast<double>(count_high) / n;
        result.probability_moderate = static_cast<double>(count_moderate) / n;
        result.probability_low = static_cast<double>(count_low) / n;
        
        // Determine risk level
        if (result.probability_critical > 0.3) {
            result.risk_level = "CRITICAL";
        } else if (result.probability_high > 0.4) {
            result.risk_level = "HIGH";
        } else if (result.probability_moderate > 0.4) {
            result.risk_level = "MODERATE";
        } else {
            result.risk_level = "LOW";
        }
    }
    
    // Effective sample size
    DistributionEstimator estimator;
    result.effective_sample_size = static_cast<int>(estimator.compute_ess(values));
}

RiskAnalysisResult MonteCarloEngine::analyze_risk(const SimulationResult& result) {
    return simulator->analyze_risk(result);
}

int MonteCarloEngine::get_completed_iterations() const {
    return completed_iterations.load();
}

void MonteCarloEngine::reset_progress() {
    completed_iterations = 0;
}

PlasmaLeakageSimulator* MonteCarloEngine::get_simulator() const {
    return simulator.get();
}

// MC WORKER IMPLEMENTATION
MCWorkerResult MCWorker::execute() {
    MCWorkerResult result;
    
    PlasmaLeakageSimulator simulator;
    simulator.set_parameters(params);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Run simulation
    SimulationResult sim_result = simulator.run_simulation(patient_data);
    
    result.leakage_values = sim_result.leakage_values;
    result.risk_scores = sim_result.risk_scores;
    result.pressures = sim_result.pressures;
    result.permeabilities = sim_result.permeabilities;
    result.iterations_completed = params.monte_carlo_iterations;
    
    auto end = std::chrono::high_resolution_clock::now();
    result.processing_time = std::chrono::duration<double>(end - start).count();
    
    return result;
}

// End Monte Carlo Engine CPP

// STATISTIC
// DISTRIBUTION ESTIMATOR IMPLEMENTATION
DistributionEstimator::DistributionEstimator(int bins) 
    : num_bins(bins), bin_width(0.0) {}

DistributionEstimator::~DistributionEstimator() {}

std::vector<double> DistributionEstimator::kernel_density_estimation(
    const std::vector<double>& samples,
    double bandwidth
) const {
    if (samples.empty()) return {};
    
    // Find range
    double min_val = *std::min_element(samples.begin(), samples.end());
    double max_val = *std::max_element(samples.begin(), samples.end());
    double range = max_val - min_val;
    
    if (range < 1e-10) range = 1.0;
    
    // Create evaluation points
    std::vector<double> eval_points(num_bins);
    for (int i = 0; i < num_bins; ++i) {
        eval_points[i] = min_val + (i + 0.5) * range / num_bins;
    }
    
    // Compute KDE
    std::vector<double> density(num_bins, 0.0);
    double n = static_cast<double>(samples.size());
    
    for (int i = 0; i < num_bins; ++i) {
        for (const auto& sample : samples) {
            double z = (eval_points[i] - sample) / bandwidth;
            density[i] += std::exp(-0.5 * z * z);
        }
        density[i] /= (n * bandwidth * std::sqrt(2.0 * M_PI));
    }
    
    return density;
}

std::pair<std::vector<double>, std::vector<double>> 
DistributionEstimator::compute_histogram(const std::vector<double>& samples) const {
    if (samples.empty()) return {{}, {}};
    
    double min_val = *std::min_element(samples.begin(), samples.end());
    double max_val = *std::max_element(samples.begin(), samples.end());
    double range = max_val - min_val;
    
    if (range < 1e-10) {
        range = 1.0;
        min_val -= 0.5;
        max_val += 0.5;
    }
    
    bin_width = range / num_bins;
    
    std::vector<double> bins(num_bins, 0.0);
    std::vector<double> counts(num_bins, 0.0);
    
    for (int i = 0; i < num_bins; ++i) {
        bins[i] = min_val + (i + 0.5) * bin_width;
    }
    
    for (const auto& sample : samples) {
        int bin_idx = static_cast<int>((sample - min_val) / bin_width);
        if (bin_idx >= 0 && bin_idx < num_bins) {
            counts[bin_idx] += 1.0;
        }
    }
    
    // Normalize to probability density
    double n = static_cast<double>(samples.size());
    for (int i = 0; i < num_bins; ++i) {
        counts[i] /= (n * bin_width);
    }
    
    return {bins, counts};
}

double DistributionEstimator::compute_exceedance_probability(
    const std::vector<double>& samples,
    double threshold
) const {
    if (samples.empty()) return 0.0;
    
    int count = 0;
    for (const auto& sample : samples) {
        if (sample > threshold) count++;
    }
    
    return static_cast<double>(count) / samples.size();
}

std::pair<double, double> DistributionEstimator::compute_confidence_interval(
    const std::vector<double>& samples,
    double confidence_level
) const {
    if (samples.empty()) return {0.0, 0.0};
    
    std::vector<double> sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    
    size_t n = sorted.size();
    double alpha = 1.0 - confidence_level;
    
    size_t lower_idx = static_cast<size_t>(n * alpha / 2.0);
    size_t upper_idx = static_cast<size_t>(n * (1.0 - alpha / 2.0));
    
    lower_idx = std::min(lower_idx, n - 1);
    upper_idx = std::min(upper_idx, n - 1);
    
    return {sorted[lower_idx], sorted[upper_idx]};
}

double DistributionEstimator::compute_ess(const std::vector<double>& samples) const {
    if (samples.size() < 2) return 0.0;
    
    // Effective sample size using autocorrelation
    double mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
    
    double variance = 0.0;
    for (const auto& s : samples) {
        variance += (s - mean) * (s - mean);
    }
    variance /= samples.size();
    
    if (variance < 1e-10) return samples.size();
    
    // Compute autocorrelation at lag 1
    double autocov = 0.0;
    for (size_t i = 0; i < samples.size() - 1; ++i) {
        autocov += (samples[i] - mean) * (samples[i + 1] - mean);
    }
    autocov /= (samples.size() - 1);
    
    double rho = autocov / variance;
    
    // ESS formula
    double ess = samples.size() * (1.0 - rho) / (1.0 + rho);
    
    return std::max(1.0, ess);
}

// RESULT AGGREGATOR IMPLEMENTATION
SimulationResult ResultAggregator::aggregate(
    const std::vector<MCWorkerResult>& worker_results
) {
    SimulationResult merged;
    
    for (const auto& result : worker_results) {
        merged.leakage_values.insert(merged.leakage_values.end(),
                                    result.leakage_values.begin(),
                                    result.leakage_values.end());
        merged.risk_scores.insert(merged.risk_scores.end(),
                                  result.risk_scores.begin(),
                                  result.risk_scores.end());
        merged.pressures.insert(merged.pressures.end(),
                              result.pressures.begin(),
                              result.pressures.end());
        merged.permeabilities.insert(merged.permeabilities.end(),
                                    result.permeabilities.begin(),
                                    result.permeabilities.end());
    }
    
    merged.total_iterations = merged.leakage_values.size();
    
    return merged;
}

void ResultAggregator::compute_percentiles(
    const std::vector<double>& values,
    double& p5, double& p25, double& p50, 
    double& p75, double& p95
) {
    if (values.empty()) {
        p5 = p25 = p50 = p75 = p95 = 0.0;
        return;
    }
    
    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end());
    
    size_t n = sorted.size();
    
    auto get_percentile = [&sorted, n](double p) -> double {
        double index = p / 100.0 * (n - 1);
        size_t lower = static_cast<size_t>(std::floor(index));
        size_t upper = static_cast<size_t>(std::ceil(index));
        if (lower == upper) return sorted[lower];
        return sorted[lower] * (upper - index) + sorted[upper] * (index - lower);
    };
    
    p5 = get_percentile(5.0);
    p25 = get_percentile(25.0);
    p50 = get_percentile(50.0);
    p75 = get_percentile(75.0);
    p95 = get_percentile(95.0);
}

void ResultAggregator::compute_higher_moments(
    const std::vector<double>& values,
    double mean,
    double& skewness,
    double& kurtosis
) {
    if (values.empty()) {
        skewness = kurtosis = 0.0;
        return;
    }
    
    size_t n = values.size();
    
    double m2 = 0.0, m3 = 0.0, m4 = 0.0;
    for (const auto& v : values) {
        double diff = v - mean;
        double diff2 = diff * diff;
        m2 += diff2;
        m3 += diff2 * diff;
        m4 += diff2 * diff2;
    }
    m2 /= n;
    m3 /= n;
    m4 /= n;
    
    if (m2 > 0) {
        skewness = m3 / std::pow(m2, 1.5);
        kurtosis = m4 / (m2 * m2) - 3.0;
    } else {
        skewness = kurtosis = 0.0;
    }
}

// End Statistic CPP

// PLASMA ENGINE IMPLEMENTATION
PlasmaEngine::PlasmaEngine() 
    : onnx_model_path("models/plasma_model.onnx"), use_onnx(false)
{
    params = SimulationParameters();
}

PlasmaEngine::~PlasmaEngine() {}

std::map<std::string, std::variant<double, int, std::string, 
                                     std::vector<double>, 
                                     std::map<std::string, double>>>
PlasmaEngine::process(const std::map<std::string, double>& input_data) {
    // Convert map to PatientData
    PatientData patient = create_patient_data(input_data);
    
    // Run simulation
    SimulationResult result = process_patient_data(patient);
    
    // Analyze risk
    RiskAnalysisResult risk = mc_engine.analyze_risk(result);
    
    // Convert to output map
    std::map<std::string, std::variant<double, int, std::string,
                                         std::vector<double>,
                                         std::map<std::string, double>>> output;
    
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
    output["overall_risk_score"] = risk.overall_risk_score;
    output["probability_critical"] = risk.probability_critical;
    output["probability_high"] = risk.probability_high;
    output["probability_moderate"] = risk.probability_moderate;
    output["probability_low"] = risk.probability_low;
    output["risk_level"] = risk.risk_level_str;
    
    // Risk factors
    std::map<std::string, double> risk_factors;
    for (const auto& [key, value] : risk.risk_factors) {
        risk_factors[key] = value;
    }
    output["risk_factors"] = risk_factors;
    
    // Metadata
    output["processing_time_ms"] = result.processing_time_ms;
    output["total_iterations"] = result.total_iterations;
    output["effective_sample_size"] = result.effective_sample_size;
    
    // Sample results (first 1000 for visualization)
    std::vector<double> samples;
    int sample_size = std::min(1000, static_cast<int>(result.leakage_values.size()));
    for (int i = 0; i < sample_size; ++i) {
        samples.push_back(result.leakage_values[i]);
    }
    output["leakage_samples"] = samples;
    
    return output;
}

SimulationResult PlasmaEngine::process_patient_data(const PatientData& patient_data) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    SimulationResult result;
    
    if (params.use_parallel && params.num_threads > 1) {
        result = mc_engine.run_parallel_monte_carlo(patient_data, params.num_threads);
    } else {
        result = mc_engine.run_monte_carlo(patient_data);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.processing_time_ms = std::chrono::duration<double, std::milli>(
        end_time - start_time
    ).count();
    
    return result;
}

std::map<std::string, std::variant<double, int, std::string,
                                     std::vector<double>,
                                     std::map<std::string, double>>>
PlasmaEngine::process_vector(
    const std::vector<double>& values,
    int iterations,
    bool use_parallel
) {
    // Create patient data from vector
    PatientData patient;
    patient.patient_id = 0;
    patient.smallData = values;
    
    if (values.size() >= 5) {
        patient.baseline_value = values[0];
        patient.permeability = values[1];
        patient.blood_pressure_systolic = values[2];
        patient.albumin = values[3];
        patient.hematocrit = values[4];
    }
    
    // Set parameters
    params.monte_carlo_iterations = iterations;
    params.use_parallel = use_parallel;
    mc_engine.set_parameters(params);
    
    return process({{"patient_id", 0.0}}); // Simplified
}

void PlasmaEngine::set_iterations(int iterations) {
    params.monte_carlo_iterations = iterations;
    mc_engine.set_parameters(params);
}

void PlasmaEngine::set_time_steps(int steps) {
    params.time_steps = steps;
    mc_engine.set_parameters(params);
}

void PlasmaEngine::set_parallel(bool use_parallel, int num_threads) {
    params.use_parallel = use_parallel;
    params.num_threads = num_threads;
    mc_engine.set_parameters(params);
}

void PlasmaEngine::set_onnx_model(const std::string& model_path) {
    onnx_model_path = model_path;
}

void PlasmaEngine::enable_onnx(bool enable) {
    use_onnx = enable;
}

void PlasmaEngine::set_parameter(const std::string& name, double value) {
    if (name == "iterations") {
        params.monte_carlo_iterations = static_cast<int>(value);
    } else if (name == "time_steps") {
        params.time_steps = static_cast<int>(value);
    } else if (name == "dt") {
        params.dt = value;
    } else if (name == "noise_std") {
        params.noise_std = value;
    } else if (name == "num_threads") {
        params.num_threads = static_cast<int>(value);
    }
    mc_engine.set_parameters(params);
}

double PlasmaEngine::get_parameter(const std::string& name) const {
    if (name == "iterations") return params.monte_carlo_iterations;
    if (name == "time_steps") return params.time_steps;
    if (name == "dt") return params.dt;
    if (name == "noise_std") return params.noise_std;
    if (name == "num_threads") return params.num_threads;
    return 0.0;
}

std::map<std::string, double> PlasmaEngine::get_all_parameters() const {
    return {
        {"iterations", static_cast<double>(params.monte_carlo_iterations)},
        {"time_steps", static_cast<double>(params.time_steps)},
        {"dt", params.dt},
        {"noise_std", params.noise_std},
        {"num_threads", static_cast<double>(params.num_threads)},
        {"use_parallel", params.use_parallel ? 1.0 : 0.0},
        {"use_onnx", params.use_onnx ? 1.0 : 0.0}
    };
}

bool PlasmaEngine::validate_input(const std::map<std::string, double>& data) const {
    // Check required fields
    auto has_key = [&data](const std::string& key) {
        return data.find(key) != data.end();
    };
    
    // Basic validation
    if (!has_key("albumin") || !has_key("hematocrit")) {
        return false;
    }
    
    double albumin = data.at("albumin");
    double hct = data.at("hematocrit");
    
    // Physiological ranges
    if (albumin <= 0 || albumin > 10) return false;
    if (hct <= 0 || hct > 100) return false;
    
    return true;
}

RiskLevel PlasmaEngine::risk_score_to_level(double risk_score) {
    if (risk_score >= 0.75) return RiskLevel::CRITICAL;
    if (risk_score >= 0.5) return RiskLevel::HIGH;
    if (risk_score >= 0.25) return RiskLevel::MODERATE;
    return RiskLevel::LOW;
}

double PlasmaEngine::compute_leakage_index(double hematocrit, double albumin) {
    if (albumin <= 0) return 0.0;
    return hematocrit / albumin;
}

std::string PlasmaEngine::get_risk_interpretation(RiskLevel level) {
    switch (level) {
        case RiskLevel::LOW:
            return "Risiko rendah: Kondisi stabil, tidak ada tanda kebocoran plasma signifikan.";
        case RiskLevel::MODERATE:
            return "Risiko sedang: Terdapat tanda-tanda kebocoran plasma, perlu pemantauan.";
        case RiskLevel::HIGH:
            return "Risiko tinggi: Kebocoran plasma signifikan, intervention diperlukan.";
        case RiskLevel::CRITICAL:
            return "Risiko kritis: Kondisi gawat darurat, kebocoran plasma berat!";
        default:
            return "Level risiko tidak diketahui.";
    }
}

// STANDALONE FUNCTIONS
std::map<std::string, std::variant<double, int, std::string,
                                     std::vector<double>,
                                     std::map<std::string, double>>>
run_plasma_simulation(
    const std::vector<double>& patient_data,
    int iterations,
    int time_steps,
    bool use_parallel,
    int num_threads
) {
    PlasmaEngine engine;
    engine.set_iterations(iterations);
    engine.set_time_steps(time_steps);
    engine.set_parallel(use_parallel, num_threads);
    
    // Create input map
    std::map<std::string, double> input;
    input["baseline"] = patient_data.size() > 0 ? patient_data[0] : 0.0;
    input["permeability"] = patient_data.size() > 1 ? patient_data[1] : 0.1;
    input["pressure"] = patient_data.size() > 2 ? patient_data[2] : 120.0;
    input["albumin"] = patient_data.size() > 3 ? patient_data[3] : 3.5;
    input["hematocrit"] = patient_data.size() > 4 ? patient_data[4] : 45.0;
    
    return engine.process(input);
}

PyOutputData convert_to_py_output(const SimulationResult& result) {
    PyOutputData output;
    
    output.expected_leakage = result.expected_leakage;
    output.variance_leakage = result.variance_leakage;
    output.percentile_5 = result.percentile_5;
    output.percentile_25 = result.percentile_25;
    output.percentile_50 = result.percentile_50;
    output.percentile_75 = result.percentile_75;
    output.percentile_95 = result.percentile_95;
    output.min_value = result.min_value;
    output.max_value = result.max_value;
    output.skewness = result.skewness;
    output.kurtosis = result.kurtosis;
    output.overall_risk_score = result.overall_risk_score;
    output.probability_critical = result.probability_critical;
    output.risk_level = result.risk_level;
    output.processing_time_ms = result.processing_time_ms;
    output.total_iterations = result.total_iterations;
    
    // Sample results
    int sample_size = std::min(1000, static_cast<int>(result.leakage_values.size()));
    for (int i = 0; i < sample_size; ++i) {
        output.leakage_samples.push_back(result.leakage_values[i]);
        output.risk_samples.push_back(result.risk_scores[i]);
    }
    
    return output;
}

PatientData create_patient_data(
    const std::map<std::string, double>& clinical_data,
    int patient_id
) {
    PatientData patient;
    patient.patient_id = patient_id;
    
    // Extract values with defaults
    auto get_val = [&clinical_data](const std::string& key, double default_val) {
        auto it = clinical_data.find(key);
        return (it != clinical_data.end()) ? it->second : default_val;
    };
    
    patient.albumin = get_val("albumin", 3.5);
    patient.hematocrit = get_val("hematocrit", 45.0);
    patient.blood_pressure_systolic = get_val("bp_systolic", 120.0);
    patient.blood_pressure_diastolic = get_val("bp_diastolic", 80.0);
    patient.pulse_rate = get_val("pulse_rate", 80.0);
    patient.baseline_value = get_val("baseline", 0.0);
    patient.permeability = get_val("permeability", 0.1);
    
    // Compute derived values
    double sum = 0.0;
    for (const auto& [key, value] : clinical_data) {
        if (key.find("value") != std::string::npos) {
            patient.smallData.push_back(value);
            sum += value;
        }
    }
    patient.mean = patient.smallData.empty() ? 0.0 : sum / patient.smallData.size();
    
    // Compute SD
    if (!patient.smallData.empty()) {
        double sq_sum = 0.0;
        for (const auto& v : patient.smallData) {
            sq_sum += (v - patient.mean) * (v - patient.mean);
        }
        patient.sd = std::sqrt(sq_sum / patient.smallData.size());
    }
    
    return patient;
}

// End Plasma Engine CPP

// LEAKAGE INDEX CALCULATOR IMPLEMENTATION
double LeakageIndexCalculator::calculate(double albumin, double hematocrit) {
    if (albumin <= 0.0 || std::isnan(albumin) || std::isnan(hematokrit)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return hematocrit / albumin;
}

std::string LeakageIndexCalculator::validate_physiology(double albumin, double hematocrit) {
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

std::vector<double> LeakageIndexCalculator::calculate_batch(
    const std::vector<double>& albumin_values, 
    const std::vector<double>& hematocrit_values
) {
    size_t n = std::min(albumin_values.size(), hematocrit_values.size());
    std::vector<double> results(n);
    
    for (size_t i = 0; i < n; ++i) {
        results[i] = calculate(albumin_values[i], hematocrit_values[i]);
    }
    
    return results;
}

std::string LeakageIndexCalculator::assess_risk(double leakage_index) {
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

// STATISTICAL CALCULATOR IMPLEMENTATION
std::pair<std::string, double> StatisticalCalculator::shapiro_wilk_test(const std::vector<double>& data) {
    // Simplified Shapiro-Wilk test implementation
    // For full implementation, would need statistical library
    if (data.size() < 3) {
        return std::make_pair("Shapiro-Wilk", 1.0); // Cannot test
    }
    
    // Simplified implementation - sort and check distribution
    std::vector<double> sorted = data;
    std::sort(sorted.begin(), sorted.end());
    
    double mean = std::accumulate(sorted.begin(), sorted.end(), 0.0) / sorted.size();
    double variance = 0.0;
    for (const auto& val : sorted) {
        variance += (val - mean) * (val - mean);
    }
    variance /= sorted.size();
    
    // Simplified W statistic approximation
    double w = 1.0;
    if (variance > 0) {
        double sum_squared_deviations = 0.0;
        for (size_t i = 0; i < sorted.size(); ++i) {
            double expected = mean; // Simplified
            sum_squared_deviations += (sorted[i] - expected) * (sorted[i] - expected);
        }
        w = 1.0 - (sum_squared_deviations / (sorted.size() * variance));
        w = std::max(0.0, std::min(1.0, w));
    }
    
    return std::make_pair("Shapiro-Wilk", w);
}

std::pair<std::string, double> StatisticalCalculator::kolmogorov_smirnov_test(const std::vector<double>& data) {
    if (data.size() < 2) {
        return std::make_pair("Kolmogorov-Smirnov", 1.0);
    }
    
    std::vector<double> sorted = data;
    std::sort(sorted.begin(), sorted.end());
    
    double mean = std::accumulate(sorted.begin(), sorted.end(), 0.0) / sorted.size();
    double std = 0.0;
    for (const auto& val : sorted) {
        std += (val - mean) * (val - mean);
    }
    std = std::sqrt(std / sorted.size());
    
    if (std == 0) {
        return std::make_pair("Kolmogorov-Smirnov", 0.0); // Perfect normal
    }
    
    // Simplified KS test - maximum deviation from normal CDF
    double max_deviation = 0.0;
    for (size_t i = 0; i < sorted.size(); ++i) {
        double empirical = (i + 1.0) / sorted.size();
        double theoretical = 0.5 * (1.0 + std::erf((sorted[i] - mean) / (std * std::sqrt(2.0))));
        max_deviation = std::max(max_deviation, std::abs(empirical - theoretical));
    }
    
    return std::make_pair("Kolmogorov-Smirnov", max_deviation);
}

std::tuple<double, double, std::string> StatisticalCalculator::pearson_correlation(
    const std::vector<double>& x, const std::vector<double>& y) {
    
    size_t n = std::min(x.size(), y.size());
    if (n < 2) {
        return std::make_tuple(0.0, 1.0, "Insufficient data");
    }
    
    double sum_x = std::accumulate(x.begin(), x.begin() + n, 0.0);
    double sum_y = std::accumulate(y.begin(), y.begin() + n, 0.0);
    double sum_xy = 0.0;
    double sum_x2 = 0.0;
    double sum_y2 = 0.0;
    
    for (size_t i = 0; i < n; ++i) {
        sum_xy += x[i] * y[i];
        sum_x2 += x[i] * x[i];
        sum_y2 += y[i] * y[i];
    }
    
    double numerator = n * sum_xy - sum_x * sum_y;
    double denominator = std::sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y));
    
    double r = (denominator != 0.0) ? numerator / denominator : 0.0;
    r = std::max(-1.0, std::min(1.0, r));
    
    // Calculate p-value using t-distribution approximation
    double t = r * std::sqrt((n - 2) / (1 - r * r));
    double p = 2.0 * (1.0 - 0.5 * (1.0 + std::erf(t / std::sqrt(2.0)))); // Approximation
    
    return std::make_tuple(r, p, "Pearson");
}

std::tuple<double, double, std::string> StatisticalCalculator::spearman_correlation(
    const std::vector<double>& x, const std::vector<double>& y) {
    
    size_t n = std::min(x.size(), y.size());
    if (n < 2) {
        return std::make_tuple(0.0, 1.0, "Insufficient data");
    }
    
    // Create rank vectors
    std::vector<std::pair<double, size_t>> x_ranked, y_ranked;
    for (size_t i = 0; i < n; ++i) {
        x_ranked.emplace_back(x[i], i);
        y_ranked.emplace_back(y[i], i);
    }
    
    std::sort(x_ranked.begin(), x_ranked.end());
    std::sort(y_ranked.begin(), y_ranked.end());
    
    std::vector<double> x_ranks(n), y_ranks(n);
    for (size_t i = 0; i < n; ++i) {
        x_ranks[x_ranked[i].second] = i + 1;
        y_ranks[y_ranked[i].second] = i + 1;
    }
    
    // Use Pearson on ranks
    return pearson_correlation(x_ranks, y_ranks);
}

std::vector<bool> StatisticalCalculator::detect_outliers_iqr(
    const std::vector<double>& data, double multiplier) {
    
    if (data.size() < 4) {
        return std::vector<bool>(data.size(), false);
    }
    
    std::vector<double> sorted = data;
    std::sort(sorted.begin(), sorted.end());
    
    size_t n = sorted.size();
    size_t q1_idx = n / 4;
    size_t q3_idx = 3 * n / 4;
    
    double q1 = sorted[q1_idx];
    double q3 = sorted[q3_idx];
    double iqr = q3 - q1;
    
    double lower_bound = q1 - multiplier * iqr;
    double upper_bound = q3 + multiplier * iqr;
    
    std::vector<bool> outliers;
    for (const auto& val : data) {
        outliers.push_back(val < lower_bound || val > upper_bound);
    }
    
    return outliers;
}

std::map<std::string, double> StatisticalCalculator::compute_summary_stats(const std::vector<double>& data) {
    if (data.empty()) {
        return {{"count", 0}, {"mean", 0}, {"std", 0}, {"min", 0}, {"max", 0}};
    }
    
    double sum = std::accumulate(data.begin(), data.end(), 0.0);
    double mean = sum / data.size();
    
    double variance = 0.0;
    for (const auto& val : data) {
        variance += (val - mean) * (val - mean);
    }
    variance /= data.size();
    double std = std::sqrt(variance);
    
    double min_val = *std::min_element(data.begin(), data.end());
    double max_val = *std::max_element(data.begin(), data.end());
    
    return {
        {"count", static_cast<double>(data.size())},
        {"mean", mean},
        {"std", std},
        {"min", min_val},
        {"max", max_val}
    };
}

std::vector<std::map<std::string, double>> StatisticalCalculator::compute_batch_stats(
    const std::vector<std::vector<double>>& datasets) {
    
    std::vector<std::map<std::string, double>> results;
    for (const auto& dataset : datasets) {
        results.push_back(compute_summary_stats(dataset));
    }
    return results;
}