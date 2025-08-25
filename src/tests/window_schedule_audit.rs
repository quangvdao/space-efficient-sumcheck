#[cfg(test)]
use std::collections::BTreeSet;

/// Test function to audit window schedule initialization for eval product prover
#[cfg(test)]
fn test_eval_product_window_schedule(d: usize, num_variables: usize, num_stages: usize) -> Vec<usize> {
    let mut windows: Vec<usize> = Vec::new();
    let delta = ((d + 1) as f64).log2();
    let mut j_prime: usize = 0;
    
    println!("=== EvalProduct Window Schedule ===");
    println!("D={}, num_variables={}, num_stages={}, delta={:.3}", d, num_variables, num_stages, delta);
    
    // Time-constrained phase
    println!("Time-constrained phase:");
    while j_prime < num_variables {
        let omega = ((j_prime as f64) / (delta - 1.0)).floor() as usize;
        let omega = omega.max(1);
        if j_prime + omega > num_variables { 
            println!("  Breaking: j_prime={}, omega={}, would exceed num_variables", j_prime, omega);
            break; 
        }
        windows.push(omega);
        println!("  j_prime={}, omega={}", j_prime, omega);
        j_prime += omega;
        
        // stop when window size reaches space phase target
        let omega_space = (num_variables / (num_stages * (delta as usize).max(1))).max(1);
        if omega >= omega_space { 
            println!("  Switching to space phase: omega={} >= omega_space={}", omega, omega_space);
            break; 
        }
    }
    
    // Space-constrained phase
    let omega_space = (num_variables / (num_stages * (delta as usize).max(1))).max(1);
    println!("Space-constrained phase: omega_space={}", omega_space);
    while j_prime + omega_space <= num_variables {
        windows.push(omega_space);
        println!("  j_prime={}, omega_space={}", j_prime, omega_space);
        j_prime += omega_space;
    }
    
    // Fallback
    if windows.is_empty() {
        windows.push(1);
        println!("Fallback: added window size 1");
    }
    
    println!("Final windows: {:?}", windows);
    println!("Total rounds covered: {}", windows.iter().sum::<usize>());
    println!();
    
    windows
}

/// Test function to audit blendy prover's state computation set
#[cfg(test)]
fn test_blendy_state_comp_set(d: usize, num_variables: usize, num_stages: usize) -> BTreeSet<usize> {
    println!("=== Blendy State Computation Set ===");
    println!("D={}, num_variables={}, num_stages={}", d, num_variables, num_stages);
    
    // Paper-aligned parameters
    let ell: usize = std::cmp::max(1, num_variables / (d * num_stages));
    let time_phase_end: usize = (d.saturating_sub(1)).saturating_mul(ell);
    let tail_rounds: usize = std::cmp::max(1, ell / std::cmp::max(1, num_stages));
    let last_round_phase3: usize = num_variables.saturating_sub(tail_rounds);
    
    println!("ell={}, time_phase_end={}, tail_rounds={}, last_round_phase3={}", 
             ell, time_phase_end, tail_rounds, last_round_phase3);
    
    let mut set: BTreeSet<usize> = BTreeSet::new();
    
    if d >= 2 {
        // Time-constrained phase: j' = ceil(eta^t) where eta = d/(d-1)
        println!("Time-constrained phase:");
        let num: usize = d;
        let den: usize = d - 1;
        let mut j_start: usize = 1;
        while j_start <= time_phase_end && j_start <= last_round_phase3 {
            set.insert(j_start);
            println!("  j_start={}", j_start);
            let next = (num.saturating_mul(j_start) + (den - 1)) / den;
            if next <= j_start { 
                println!("  Breaking: next={} <= j_start={}", next, j_start);
                break; 
            }
            j_start = next;
        }
        
        // Space-constrained phase
        println!("Space-constrained phase:");
        if ell > 0 {
            let m = if time_phase_end == 0 { 1 } else { (time_phase_end + ell - 1) / ell };
            let mut j_space = m.saturating_mul(ell);
            if j_space == 0 { j_space = ell; }
            println!("  Starting j_space={} (m={})", j_space, m);
            while j_space <= last_round_phase3 {
                set.insert(j_space);
                println!("  j_space={}", j_space);
                let (next, overflow) = j_space.overflowing_add(ell);
                if overflow { break; }
                j_space = next;
            }
        }
        
        if set.is_empty() { 
            set.insert(1); 
            println!("Fallback: inserted 1");
        }
    }
    
    println!("Final state_comp_set: {:?}", set);
    println!();
    
    set
}

/// Compare paper formulas with implementations
#[cfg(test)]
fn compare_with_paper_formulas(d: usize, num_variables: usize, k: usize) {
    println!("=== Paper Formula Comparison ===");
    println!("D={}, ℓ={}, k={}", d, num_variables, k);
    
    let delta = ((d + 1) as f64).log2();
    let alpha = delta / (delta - 1.0);
    
    println!("δ = log₂(d+1) = {:.3}", delta);
    println!("α = δ/(δ-1) = {:.3}", alpha);
    
    // Time-constrained phase: ω_t = α^(t-1) / (δ-1)
    println!("Time-constrained phase windows (paper formula):");
    let mut t = 1;
    let mut total_rounds = 0;
    while total_rounds < num_variables {
        let omega_t = (alpha.powi(t - 1) / (delta - 1.0)).floor() as usize;
        let omega_t = omega_t.max(1);
        if total_rounds + omega_t > num_variables { break; }
        
        println!("  t={}, ω_t={}", t, omega_t);
        total_rounds += omega_t;
        t += 1;
        
        // Check if we should switch to space phase
        let omega_space = num_variables / (k * (delta as usize).max(1));
        if omega_t >= omega_space { 
            println!("  Should switch to space phase at ω_t={} >= {}", omega_t, omega_space);
            break; 
        }
    }
    
    // Space-constrained phase: ω = ℓ/(k·δ)
    let omega_space = num_variables / (k * (delta as usize).max(1));
    println!("Space-constrained phase: ω = ℓ/(k·δ) = {}/{} = {}", 
             num_variables, k * (delta as usize).max(1), omega_space);
    
    println!();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_various_parameters() {
        let test_cases = vec![
            (2, 10, 2),   // D=2, small instance
            (2, 20, 2),   // D=2, medium instance  
            (3, 15, 2),   // D=3, small instance
            (4, 24, 3),   // D=4, medium instance
            (2, 4, 2),    // D=2, very small (edge case)
            (2, 40, 4),  // D=2, huge instance
        ];
        
        for (d, num_vars, k) in test_cases {
            println!("########## Testing D={}, num_variables={}, k={} ##########", d, num_vars, k);
            
            compare_with_paper_formulas(d, num_vars, k);
            test_eval_product_window_schedule(d, num_vars, k);
            test_blendy_state_comp_set(d, num_vars, k);
            
            println!("{}", "=".repeat(60));
            println!();
        }
    }
}
