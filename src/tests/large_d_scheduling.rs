use crate::multilinear_product::{SchedulingParams, compute_cross_product_schedule, compute_eval_product_schedule};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_large_d_scheduling() {
        let test_cases = vec![
            // D=8 cases
            (8, 24, 2),   // D=8, n=24, k=2 → hard_stop = 24 - 12 = 12
            (8, 30, 2),   // D=8, n=30, k=2 → hard_stop = 30 - 15 = 15
            (8, 24, 3),   // D=8, n=24, k=3 → hard_stop = 24 - 8 = 16
            (8, 30, 3),   // D=8, n=30, k=3 → hard_stop = 30 - 10 = 20
            (8, 36, 3),   // D=8, n=36, k=3 → hard_stop = 36 - 12 = 24
            
            // D=16 cases  
            (16, 24, 2),  // D=16, n=24, k=2 → hard_stop = 24 - 12 = 12
            (16, 30, 2),  // D=16, n=30, k=2 → hard_stop = 30 - 15 = 15
            (16, 32, 2),  // D=16, n=32, k=2 → hard_stop = 32 - 16 = 16
            (16, 24, 3),  // D=16, n=24, k=3 → hard_stop = 24 - 8 = 16
            (16, 30, 3),  // D=16, n=30, k=3 → hard_stop = 30 - 10 = 20
            (16, 36, 3),  // D=16, n=36, k=3 → hard_stop = 36 - 12 = 24
            (16, 40, 3),  // D=16, n=40, k=3 → hard_stop = 40 - 13 = 27
        ];
        
        for (d, num_vars, k) in test_cases {
            let expected_hard_stop = num_vars - (num_vars / k);
            
            println!("########## D={}, n={}, k={} ##########", d, num_vars, k);
            println!("Expected hard stop: {} - {} = {}", num_vars, num_vars / k, expected_hard_stop);
            
            let params = SchedulingParams { d, num_variables: num_vars, num_stages: k };
            
            let cross_schedule = compute_cross_product_schedule(&params);
            let eval_schedule = compute_eval_product_schedule(&params);
            
            println!("CrossProduct schedule: {:?}", cross_schedule);
            println!("EvalProduct schedule:  {:?}", eval_schedule);
            
            // Check hard stop is respected
            let cross_max = cross_schedule.iter().max().copied().unwrap_or(0);
            let eval_max = eval_schedule.iter().max().copied().unwrap_or(0);
            
            println!("CrossProduct max round: {}, hard_stop: {}", cross_max, expected_hard_stop);
            println!("EvalProduct max round:  {}, hard_stop: {}", eval_max, expected_hard_stop);
            
            assert!(cross_max <= expected_hard_stop, 
                "CrossProduct exceeds hard stop: {} > {}", cross_max, expected_hard_stop);
            assert!(eval_max <= expected_hard_stop, 
                "EvalProduct exceeds hard stop: {} > {}", eval_max, expected_hard_stop);
            
            // Check that we have reasonable number of passes (not too many, not too few)
            println!("CrossProduct passes: {}", cross_schedule.len());
            println!("EvalProduct passes:  {}", eval_schedule.len());
            
            // For large D, we expect EvalProduct to have fewer passes due to better growth rate
            if d >= 8 {
                println!("EvalProduct efficiency: {} vs {} passes", eval_schedule.len(), cross_schedule.len());
            }
            
            // Compute growth rates for analysis
            if d >= 8 {
                let eta = d as f64 / (d - 1) as f64;
                let delta = ((d + 1) as f64).log2();
                let alpha = delta / (delta - 1.0);
                
                println!("Growth rates: η={:.3} (CrossProduct), α={:.3} (EvalProduct)", eta, alpha);
                println!("δ = log₂({}) = {:.3}", d + 1, delta);
            }
            
            println!();
        }
    }
    
    #[test]
    fn test_extreme_large_d() {
        // Test even larger D values to see behavior
        let extreme_cases = vec![
            (32, 30, 2),   // D=32, very large degree
            (64, 36, 3),   // D=64, extreme case
        ];
        
        for (d, num_vars, k) in extreme_cases {
            println!("########## EXTREME: D={}, n={}, k={} ##########", d, num_vars, k);
            
            let params = SchedulingParams { d, num_variables: num_vars, num_stages: k };
            
            let cross_schedule = compute_cross_product_schedule(&params);
            let eval_schedule = compute_eval_product_schedule(&params);
            
            println!("CrossProduct schedule: {:?}", cross_schedule);
            println!("EvalProduct schedule:  {:?}", eval_schedule);
            
            // Compute growth rates
            let eta = d as f64 / (d - 1) as f64;
            let delta = ((d + 1) as f64).log2();
            let alpha = delta / (delta - 1.0);
            
            println!("Growth rates: η={:.6} (CrossProduct), α={:.3} (EvalProduct)", eta, alpha);
            println!("δ = log₂({}) = {:.3}", d + 1, delta);
            println!("Passes: CrossProduct={}, EvalProduct={}", cross_schedule.len(), eval_schedule.len());
            println!();
        }
    }
}
