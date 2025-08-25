use crate::multilinear_product::{SchedulingParams, compute_cross_product_schedule, compute_eval_product_schedule};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schedule_comparison() {
        let test_cases = vec![
            (2, 20, 2),   // D=2, medium instance
            (3, 15, 2),   // D=3, small instance
            (4, 24, 3),   // D=4, medium instance
            (2, 40, 4),  // D=2, huge instance
        ];
        
        for (d, num_vars, k) in test_cases {
            println!("########## D={}, num_variables={}, k={} ##########", d, num_vars, k);
            
            let params = SchedulingParams { d, num_variables: num_vars, num_stages: k };
            
            let cross_schedule = compute_cross_product_schedule(&params);
            let eval_schedule = compute_eval_product_schedule(&params);
            
            println!("CrossProduct schedule: {:?}", cross_schedule);
            println!("EvalProduct schedule:  {:?}", eval_schedule);
            
            // Verify basic properties
            assert!(!cross_schedule.is_empty(), "CrossProduct schedule should not be empty");
            assert!(!eval_schedule.is_empty(), "EvalProduct schedule should not be empty");
            
            // Both should start at round 1
            assert!(cross_schedule.contains(&1), "CrossProduct should start at round 1");
            assert!(eval_schedule.contains(&1), "EvalProduct should start at round 1");
            
            // EvalProduct should generally have fewer passes due to better growth rate
            println!("CrossProduct passes: {}, EvalProduct passes: {}", 
                     cross_schedule.len(), eval_schedule.len());
            
            println!();
        }
    }
}
