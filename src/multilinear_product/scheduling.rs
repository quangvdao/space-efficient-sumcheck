use std::collections::BTreeSet;

/// Scheduling parameters for different sum-check algorithms
#[derive(Debug, Clone)]
pub struct SchedulingParams {
    pub d: usize,              // degree (number of polynomials)
    pub num_variables: usize,  // ℓ in paper
    pub num_stages: usize,     // k in paper
}

/// Compute state computation set for CrossProduct algorithm
/// Uses η = d/(d-1) growth rate (coefficient-based)
pub fn compute_cross_product_schedule(params: &SchedulingParams) -> BTreeSet<usize> {
    debug_assert!(params.d >= 2, "CrossProduct requires D >= 2");
    
    let d = params.d;
    let num_variables = params.num_variables;
    let num_stages = params.num_stages;
    
    // Paper parameters: ℓ₀ = ⌊n/(d·k)⌋, time_phase_end = (d-1)·ℓ₀
    let ell = (num_variables as f64 / (d * num_stages) as f64).floor() as usize;
    let ell = ell.max(1); // ensure at least 1
    let time_phase_end = (d.saturating_sub(1)).saturating_mul(ell);
    
    // Hard stop: switch to time prover at n - floor(n/k) for optimal performance
    let hard_stop = num_variables.saturating_sub(num_variables / num_stages);
    let last_round_phase3 = hard_stop;
    
    let mut set: BTreeSet<usize> = BTreeSet::new();
    
    // Time-constrained phase: j' = ⌈η^t⌉ where η = d/(d-1)
    let eta = d as f64 / (d - 1) as f64;
    let mut j_start = 1.0f64;
    
    while (j_start as usize) <= time_phase_end && (j_start as usize) <= last_round_phase3 {
        set.insert(j_start.ceil() as usize);
        j_start *= eta;
        if j_start > num_variables as f64 { break; }
    }
    
    // Space-constrained phase: passes every ℓ₀ rounds
    if ell > 0 {
        let first_space_round = if time_phase_end == 0 { 
            ell 
        } else { 
            ((time_phase_end as f64 / ell as f64).ceil() as usize) * ell 
        };
        
        let mut j_space = first_space_round;
        while j_space <= last_round_phase3 {
            set.insert(j_space);
            if j_space > num_variables.saturating_sub(ell) { break; }
            j_space += ell;
        }
    }
    
    // Ensure hard stop is respected by removing any rounds that exceed it
    set.retain(|&round| round <= hard_stop);
    
    // Ensure we have at least one stage start
    if set.is_empty() { 
        set.insert(1); 
    }
    
    set
}

/// Compute state computation set for EvalProduct algorithm  
/// Uses α = δ/(δ-1) where δ = log₂(d+1) growth rate (evaluation-based)
pub fn compute_eval_product_schedule(params: &SchedulingParams) -> BTreeSet<usize> {
    debug_assert!(params.d >= 2, "EvalProduct requires D >= 2");
    
    let d = params.d;
    let num_variables = params.num_variables;
    let num_stages = params.num_stages;
    
    // EvalProduct uses δ = log₂(d+1) instead of d directly
    let delta = ((d + 1) as f64).log2();
    let alpha = delta / (delta - 1.0);
    
    // Space-constrained phase uses ℓ/(k·δ) instead of ℓ/(k·d)
    let ell_over_k_delta = num_variables as f64 / (num_stages as f64 * delta);
    let omega_space = ell_over_k_delta.floor() as usize;
    let omega_space = omega_space.max(1);
    
    // Time phase ends when window size would reach space phase target
    // Compute this more precisely by finding where α^t / (δ-1) ≈ ω_space
    let target_window_size = omega_space;
    let time_phase_end = ((target_window_size as f64 * (delta - 1.0)).log(alpha)).floor() as usize;
    
    // Hard stop: switch to time prover at n - floor(n/k) for optimal performance
    let hard_stop = num_variables.saturating_sub(num_variables / num_stages);
    let last_round_phase3 = hard_stop;
    
    let mut set: BTreeSet<usize> = BTreeSet::new();
    
    // Time-constrained phase: j' follows α^t growth
    let mut j_start = 1.0f64;
    
    while (j_start as usize) <= time_phase_end && (j_start as usize) <= last_round_phase3 {
        set.insert(j_start.ceil() as usize);
        j_start *= alpha;
        if j_start > num_variables as f64 { break; }
    }
    
    // Space-constrained phase: passes every ω_space rounds
    if omega_space > 0 {
        // Find the first space round that comes after the time-constrained phase
        let time_phase_actual_end = set.iter().max().copied().unwrap_or(1);
        let first_space_round = ((time_phase_actual_end as f64 / omega_space as f64).ceil() as usize) * omega_space;
        
        let mut j_space = first_space_round;
        while j_space <= last_round_phase3 {
            set.insert(j_space);
            if j_space > last_round_phase3.saturating_sub(omega_space) { break; }
            j_space += omega_space;
        }
    }
    
    // Ensure we have at least one stage start
    if set.is_empty() { 
        set.insert(1); 
    }
    
    set
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cross_product_schedule() {
        let params = SchedulingParams { d: 2, num_variables: 20, num_stages: 2 };
        let schedule = compute_cross_product_schedule(&params);
        println!("CrossProduct D=2, n=20, k=2: {:?}", schedule);
        
        let params = SchedulingParams { d: 3, num_variables: 15, num_stages: 2 };
        let schedule = compute_cross_product_schedule(&params);
        println!("CrossProduct D=3, n=15, k=2: {:?}", schedule);
    }
    
    #[test]
    fn test_eval_product_schedule() {
        let params = SchedulingParams { d: 2, num_variables: 20, num_stages: 2 };
        let schedule = compute_eval_product_schedule(&params);
        println!("EvalProduct D=2, n=20, k=2: {:?}", schedule);
        
        let params = SchedulingParams { d: 3, num_variables: 15, num_stages: 2 };
        let schedule = compute_eval_product_schedule(&params);
        println!("EvalProduct D=3, n=15, k=2: {:?}", schedule);
    }
    
    #[test]
    fn compare_schedules() {
        let params = SchedulingParams { d: 2, num_variables: 100, num_stages: 4 };
        
        let cross_schedule = compute_cross_product_schedule(&params);
        let eval_schedule = compute_eval_product_schedule(&params);
        
        println!("CrossProduct: {:?}", cross_schedule);
        println!("EvalProduct:  {:?}", eval_schedule);
        
        // EvalProduct should have fewer passes due to better growth rate
        assert!(eval_schedule.len() <= cross_schedule.len());
    }
}
