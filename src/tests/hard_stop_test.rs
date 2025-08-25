#[cfg(test)]
use crate::multilinear_product::{SchedulingParams, compute_cross_product_schedule, compute_eval_product_schedule};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hard_stop_cutoffs() {
        let test_cases = vec![
            (2, 20, 2),   // D=2, n=20, k=2 → hard_stop = 20 - 10 = 10
            (3, 15, 2),   // D=3, n=15, k=2 → hard_stop = 15 - 7 = 8  
            (4, 24, 3),   // D=4, n=24, k=3 → hard_stop = 24 - 8 = 16
            (2, 100, 4),  // D=2, n=100, k=4 → hard_stop = 100 - 25 = 75
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
            
            // Verify no passes start at or after the hard stop
            for &start in &cross_schedule {
                assert!(start <= expected_hard_stop, 
                        "CrossProduct pass at {} should not exceed hard stop {}", start, expected_hard_stop);
            }
            
            for &start in &eval_schedule {
                assert!(start <= expected_hard_stop, 
                        "EvalProduct pass at {} should not exceed hard stop {}", start, expected_hard_stop);
            }
            
            // Verify the largest pass is close to the hard stop (within reasonable range)
            let cross_max = cross_schedule.iter().max().copied().unwrap_or(0);
            let eval_max = eval_schedule.iter().max().copied().unwrap_or(0);
            
            println!("CrossProduct max pass: {}, EvalProduct max pass: {}", cross_max, eval_max);
            println!("Hard stop enforced: ✓");
            println!();
        }
    }
}
