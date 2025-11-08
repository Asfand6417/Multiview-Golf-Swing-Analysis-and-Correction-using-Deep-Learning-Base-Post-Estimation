"""
MPJPE Evaluation and Comparison with Baselines

This module implements evaluation metrics for 3D pose estimation,
including Mean Per Joint Position Error (MPJPE) and comparison
with single-view and marker-based baselines.

Research Objectives:
    1. Accuracy of 3D reconstruction [[45], [46]]
    4. Comparison with single-view & marker-based baselines [[50]-[52]]
    
Technical Constraints:
    - Include asserts for reprojection error < 2 px for calibration sample
    - Include asserts for MPJPE < X mm on validation clip
    - Print F1-score for swing phase segmentation
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix


# Hyperparameters for evaluation
MPJPE_THRESHOLD = 50.0  # Maximum acceptable MPJPE in mm
REPROJECTION_ERROR_THRESHOLD = 2.0  # Maximum acceptable reprojection error in pixels
PHASE_F1_THRESHOLD = 0.7  # Minimum acceptable F1 score for phase segmentation


class PoseEvaluator:
    """
    Evaluation metrics for 3D pose estimation.
    
    Research Alignment:
        Supports Objective 1: Accuracy of 3D reconstruction [[45], [46]]
        by implementing MPJPE and other accuracy metrics.
        
        Supports Objective 4: Comparison with baselines [[50]-[52]]
        by enabling comparison with single-view and marker-based methods.
    """

    def __init__(self):
        """Initialize the pose evaluator."""
        self.metrics = {
            'mpjpe': [],
            'per_joint_error': {},
            'phase_accuracy': [],
            'reprojection_error': [],
            'baseline_comparison': {}
        }
        
        # Store results for different methods for comparison
        self.method_results = {}

    def calculate_mpjpe(self, pred_coords, gt_coords):
        """
        Calculate Mean Per Joint Position Error between predicted and ground truth coordinates.
        
        Args:
            pred_coords: Predicted landmark coordinates (n_joints, 3)
            gt_coords: Ground truth landmark coordinates (n_joints, 3)
            
        Returns:
            MPJPE value (mean Euclidean distance)
            
        Research Alignment:
            MPJPE is a standard metric used in [[45]] for evaluating 3D pose accuracy.
        """
        # Ensure arrays are the right shape
        pred = np.array(pred_coords)
        gt = np.array(gt_coords)
        
        # Make sure shapes match - truncate if necessary
        min_length = min(len(pred), len(gt))
        pred = pred[:min_length]
        gt = gt[:min_length]
        
        # Calculate Euclidean distance for each joint
        distances = np.linalg.norm(pred - gt, axis=1)
        
        # Store per-joint errors
        for i, dist in enumerate(distances):
            if i not in self.metrics['per_joint_error']:
                self.metrics['per_joint_error'][i] = []
            self.metrics['per_joint_error'][i].append(dist)
        
        # Calculate mean
        mpjpe = np.mean(distances)
        
        # Store for reporting
        self.metrics['mpjpe'].append(mpjpe)
        
        # Assert that MPJPE is below threshold
        assert mpjpe < MPJPE_THRESHOLD, f"MPJPE ({mpjpe:.2f} mm) exceeds threshold ({MPJPE_THRESHOLD} mm)"
        
        return mpjpe

    def calculate_reprojection_error(self, points_3d, points_2d, projection_matrix):
        """
        Calculate reprojection error for 3D points.
        
        Args:
            points_3d: 3D points (n_points, 3)
            points_2d: Corresponding 2D points (n_points, 2)
            projection_matrix: Camera projection matrix (3, 4)
            
        Returns:
            Mean reprojection error in pixels
            
        Research Alignment:
            Reprojection error is used in [[46]] to evaluate calibration accuracy.
        """
        # Convert 3D points to homogeneous coordinates
        points_3d_homogeneous = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
        
        # Project 3D points to 2D
        projected_points = projection_matrix @ points_3d_homogeneous.T
        projected_points = projected_points[:2] / projected_points[2]
        projected_points = projected_points.T
        
        # Calculate Euclidean distance between original and reprojected points
        errors = np.linalg.norm(points_2d - projected_points, axis=1)
        
        # Calculate mean error
        mean_error = np.mean(errors)
        
        # Store for reporting
        self.metrics['reprojection_error'].append(mean_error)
        
        # Assert that reprojection error is below threshold
        assert mean_error < REPROJECTION_ERROR_THRESHOLD, f"Reprojection error ({mean_error:.2f} px) exceeds threshold ({REPROJECTION_ERROR_THRESHOLD} px)"
        
        return mean_error

    def evaluate_phase_segmentation(self, predicted_phases, ground_truth_phases):
        """
        Evaluate the accuracy of swing phase segmentation.
        
        Args:
            predicted_phases: List of predicted phases
            ground_truth_phases: List of ground truth phases
            
        Returns:
            Dictionary with accuracy, recall, and F1 score
            
        Research Alignment:
            Phase segmentation evaluation follows metrics used in [[47]].
        """
        # Convert phases to numerical indices if they are strings
        if isinstance(predicted_phases[0], str):
            phase_to_idx = {phase: i for i, phase in enumerate(set(predicted_phases).union(set(ground_truth_phases)))}
            pred_idx = [phase_to_idx[phase] for phase in predicted_phases]
            gt_idx = [phase_to_idx[phase] for phase in ground_truth_phases]
        else:
            pred_idx = predicted_phases
            gt_idx = ground_truth_phases
        
        # Calculate metrics
        accuracy = accuracy_score(gt_idx, pred_idx)
        recall = recall_score(gt_idx, pred_idx, average='macro')
        f1 = f1_score(gt_idx, pred_idx, average='macro')
        conf_matrix = confusion_matrix(gt_idx, pred_idx)
        
        # Store metrics
        self.metrics['phase_accuracy'].append({
            'accuracy': accuracy,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': conf_matrix
        })
        
        # Print F1 score as required by technical constraints
        print(f"Swing-phase F1-score: {f1:.4f}")
        
        # Assert that F1 score is above threshold
        assert f1 > PHASE_F1_THRESHOLD, f"Phase segmentation F1 score ({f1:.2f}) below threshold ({PHASE_F1_THRESHOLD})"
        
        return {
            'accuracy': accuracy,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': conf_matrix
        }

    def compare_with_baseline(self, method_name, mpjpe_values, is_baseline=False):
        """
        Compare current method with baseline methods.
        
        Args:
            method_name: Name of the method
            mpjpe_values: List of MPJPE values for this method
            is_baseline: Whether this is a baseline method
            
        Research Alignment:
            Comparison methodology follows [[50]-[52]] for evaluating against baselines.
        """
        # Store results for this method
        self.method_results[method_name] = {
            'mpjpe': mpjpe_values,
            'mean_mpjpe': np.mean(mpjpe_values),
            'std_mpjpe': np.std(mpjpe_values),
            'is_baseline': is_baseline
        }
        
        # If this is not a baseline, compare with all baselines
        if not is_baseline:
            baselines = {name: data for name, data in self.method_results.items() if data['is_baseline']}
            
            if baselines:
                # Calculate improvement over each baseline
                improvements = {}
                for baseline_name, baseline_data in baselines.items():
                    baseline_mpjpe = baseline_data['mean_mpjpe']
                    current_mpjpe = self.method_results[method_name]['mean_mpjpe']
                    
                    # Calculate relative improvement
                    rel_improvement = (baseline_mpjpe - current_mpjpe) / baseline_mpjpe * 100
                    
                    improvements[baseline_name] = {
                        'absolute': baseline_mpjpe - current_mpjpe,
                        'relative': rel_improvement
                    }
                
                # Store comparison results
                self.metrics['baseline_comparison'][method_name] = improvements
        
        return self.method_results[method_name]

    def generate_report(self, output_path=None):
        """
        Generate a comprehensive evaluation report.
        
        Args:
            output_path: Path to save the report (if None, won't save)
            
        Returns:
            Dictionary with all evaluation metrics
        """
        report = {
            'mpjpe': {
                'mean': np.mean(self.metrics['mpjpe']),
                'std': np.std(self.metrics['mpjpe']),
                'min': np.min(self.metrics['mpjpe']) if self.metrics['mpjpe'] else None,
                'max': np.max(self.metrics['mpjpe']) if self.metrics['mpjpe'] else None
            },
            'per_joint_error': {
                joint: {
                    'mean': np.mean(errors),
                    'std': np.std(errors)
                } for joint, errors in self.metrics['per_joint_error'].items()
            },
            'phase_detection': {
                'accuracy': np.mean([m['accuracy'] for m in self.metrics['phase_accuracy']]) if self.metrics['phase_accuracy'] else None,
                'recall': np.mean([m['recall'] for m in self.metrics['phase_accuracy']]) if self.metrics['phase_accuracy'] else None,
                'f1': np.mean([m['f1'] for m in self.metrics['phase_accuracy']]) if self.metrics['phase_accuracy'] else None
            },
            'reprojection_error': {
                'mean': np.mean(self.metrics['reprojection_error']),
                'max': np.max(self.metrics['reprojection_error']) if self.metrics['reprojection_error'] else None
            },
            'baseline_comparison': self.metrics['baseline_comparison']
        }
        
        # Print summary
        print("\n===== Evaluation Report =====")
        print(f"Mean MPJPE: {report['mpjpe']['mean']:.2f} mm (±{report['mpjpe']['std']:.2f})")
        print(f"Mean Reprojection Error: {report['reprojection_error']['mean']:.2f} px")
        
        if report['phase_detection']['f1'] is not None:
            print(f"Phase Detection F1: {report['phase_detection']['f1']:.4f}")
        
        if self.metrics['baseline_comparison']:
            print("\nBaseline Comparisons:")
            for method, comparisons in self.metrics['baseline_comparison'].items():
                print(f"  {method}:")
                for baseline, improvement in comparisons.items():
                    print(f"    vs {baseline}: {improvement['absolute']:.2f} mm better ({improvement['relative']:.1f}% improvement)")
        
        # Save report if output path is provided
        if output_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save as CSV
            if output_path.endswith('.csv'):
                # Convert report to DataFrame
                df = pd.DataFrame({
                    'Metric': ['MPJPE (mm)', 'Reprojection Error (px)', 'Phase F1 Score'],
                    'Value': [
                        report['mpjpe']['mean'],
                        report['reprojection_error']['mean'],
                        report['phase_detection']['f1'] if report['phase_detection']['f1'] is not None else np.nan
                    ],
                    'Std': [
                        report['mpjpe']['std'],
                        np.std(self.metrics['reprojection_error']) if self.metrics['reprojection_error'] else np.nan,
                        np.std([m['f1'] for m in self.metrics['phase_accuracy']]) if self.metrics['phase_accuracy'] else np.nan
                    ]
                })
                
                df.to_csv(output_path, index=False)
                print(f"Report saved to {output_path}")
            
            # Save as JSON
            elif output_path.endswith('.json'):
                import json
                with open(output_path, 'w') as f:
                    json.dump(report, f, indent=2)
                print(f"Report saved to {output_path}")
        
        return report

    def plot_comparison(self, output_path=None):
        """
        Plot comparison between current method and baselines.
        
        Args:
            output_path: Path to save the plot (if None, won't save)
            
        Research Alignment:
            Visualization follows presentation format in [[52]] for comparing methods.
        """
        if not self.method_results:
            print("No methods to compare.")
            return
        
        # Prepare data for plotting
        methods = []
        means = []
        stds = []
        colors = []
        
        for name, data in self.method_results.items():
            methods.append(name)
            means.append(data['mean_mpjpe'])
            stds.append(data['std_mpjpe'])
            colors.append('lightblue' if data['is_baseline'] else 'green')
        
        # Sort by mean MPJPE (ascending)
        sorted_indices = np.argsort(means)
        methods = [methods[i] for i in sorted_indices]
        means = [means[i] for i in sorted_indices]
        stds = [stds[i] for i in sorted_indices]
        colors = [colors[i] for i in sorted_indices]
        
        # Create plot
        plt.figure(figsize=(10, 6))
        bars = plt.bar(methods, means, yerr=stds, capsize=10, color=colors)
        
        # Add labels and title
        plt.xlabel('Method')
        plt.ylabel('MPJPE (mm)')
        plt.title('Comparison of 3D Pose Estimation Methods')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Add value labels on top of bars
        for bar, mean in zip(bars, means):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{mean:.1f}', ha='center', va='bottom')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lightblue', label='Baseline'),
            Patch(facecolor='green', label='Our Method')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        # Save plot if output path is provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to {output_path}")
        
        plt.show()


def evaluate_reconstruction(pred_sequence, gt_sequence, projection_matrix=None):
    """
    Evaluate 3D reconstruction accuracy.
    
    Args:
        pred_sequence: Sequence of predicted 3D poses
        gt_sequence: Sequence of ground truth 3D poses
        projection_matrix: Optional camera projection matrix for reprojection error
        
    Returns:
        Dictionary with evaluation metrics
    """
    evaluator = PoseEvaluator()
    
    # Calculate MPJPE for each frame
    mpjpe_values = []
    for pred, gt in zip(pred_sequence, gt_sequence):
        mpjpe = evaluator.calculate_mpjpe(pred, gt)
        mpjpe_values.append(mpjpe)
    
    # Calculate reprojection error if projection matrix is provided
    if projection_matrix is not None:
        # We need 2D projections of ground truth for this
        # This is just a placeholder - in practice, you would use actual 2D detections
        gt_2d = np.array([point[:, :2] for point in gt_sequence])
        
        for points_3d, points_2d in zip(pred_sequence, gt_2d):
            evaluator.calculate_reprojection_error(points_3d, points_2d, projection_matrix)
    
    # Generate report
    report = evaluator.generate_report()
    
    return report


def compare_methods(method_results, baseline_results, output_path=None):
    """
    Compare current method with baseline methods.
    
    Args:
        method_results: Dictionary mapping method names to sequences of MPJPE values
        baseline_results: Dictionary mapping baseline names to sequences of MPJPE values
        output_path: Path to save the comparison plot
        
    Returns:
        PoseEvaluator instance with comparison results
    """
    evaluator = PoseEvaluator()
    
    # Add baseline methods
    for name, mpjpe_values in baseline_results.items():
        evaluator.compare_with_baseline(name, mpjpe_values, is_baseline=True)
    
    # Add current methods
    for name, mpjpe_values in method_results.items():
        evaluator.compare_with_baseline(name, mpjpe_values, is_baseline=False)
    
    # Generate report and plot
    evaluator.generate_report()
    evaluator.plot_comparison(output_path)
    
    return evaluator


if __name__ == "__main__":
    # Example usage
    print("Pose Evaluation module - Run through the main pipeline")