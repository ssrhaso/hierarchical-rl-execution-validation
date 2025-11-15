"""
GENERATE PROFESSIONAL VISUALIZATIONS FOR RL EXECUTION ANALYSIS

Creates 4 high-quality charts analyzing model performance:
1. Slippage Comparison (Box Plot)
2. Execution Efficiency (Scatter Plot)
3. Performance Heatmap (Per-Symbol Performance)
4. Statistical Summary (Bar Chart with Error Bars)
"""

import sys
sys.path.append('.')

import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/visualization.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# STYLING

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Professional color scheme
COLORS = {
    'dqn_guided': '#2E86AB',           # Deep Blue
    'ppo_strategic': '#A23B72',        # Purple
    'twap': '#C73E1D',                 # Red
    'vwap': '#6A994E',                 # Green
    'random': "#FFF700"                # Yellow
}

MODEL_NAMES = {
    'dqn_guided': 'DQN',
    'ppo_strategic': 'PPO',
    'twap': 'TWAP',
    'vwap': 'VWAP',
    'random': 'Random'
}

# CHART 1: SLIPPAGE COMPARISON (BOX PLOT)
def create_slippage_comparison(results_df, output_path):
    """Box plot of slippage distribution by model"""
    logger.info("Creating Chart 1: Slippage Comparison...")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Get models in order
    model_order = ['dqn_guided', 'ppo_strategic', 'vwap', 'twap', 'random']
    available_models = [m for m in model_order if m in results_df['model'].unique()]
    
    # Prepare data
    plot_data = []
    plot_models = []
    for model in available_models:
        data = results_df[results_df['model'] == model]['slippage_bps'].values
        plot_data.append(data)
        plot_models.append(MODEL_NAMES.get(model, model))
    
    # Create box plot
    bp = ax.boxplot(plot_data, labels=plot_models, patch_artist=True, widths=0.6,
                    boxprops=dict(linewidth=2),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5),
                    medianprops=dict(color='red', linewidth=2.5))
    
    # Color boxes
    colors_list = [COLORS.get(m, '#999999') for m in available_models]
    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Styling
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5, label='Break-even')
    ax.set_ylabel('Slippage (basis points)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Model', fontsize=13, fontweight='bold')
    ax.set_title('Execution Slippage Comparison: DQN vs PPO vs Baselines\n(Lower is Better)', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=11, loc='upper right')
    
    # Add value labels on medians
    for i, (model_name, data) in enumerate(zip(plot_models, plot_data)):
        median = np.median(data)
        ax.text(i+1, median, f'{median:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.xticks(fontsize=11, rotation=0)
    plt.yticks(fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved: {output_path}")
    plt.close()

# CHART 2: EXECUTION EFFICIENCY (SCATTER PLOT)
def create_execution_efficiency(results_df, output_path):
    """Scatter plot: Execution Time vs Slippage"""
    logger.info("Creating Chart 2: Execution Efficiency...")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    model_order = ['dqn_guided', 'ppo_strategic', 'vwap', 'twap', 'random']
    available_models = [m for m in model_order if m in results_df['model'].unique()]
    
    # Plot each model
    for model in available_models:
        model_data = results_df[results_df['model'] == model]
        ax.scatter(model_data['exec_time_steps'], model_data['slippage_bps'],
                  label=MODEL_NAMES.get(model, model),
                  color=COLORS.get(model, '#999999'),
                  s=120, alpha=0.7, edgecolors='black', linewidth=1.5)
    
    # Styling
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax.set_xlabel('Execution Time (steps)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Slippage (basis points)', fontsize=13, fontweight='bold')
    ax.set_title('Execution Efficiency: Speed vs Accuracy\n(Lower Right = Better)', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='best', framealpha=0.95)
    ax.grid(True, alpha=0.3)
    
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved: {output_path}")
    plt.close()

# CHART 3: PERFORMANCE HEATMAP (PER-SYMBOL)
def create_performance_heatmap(results_df, output_path):
    """Heatmap of average slippage by model and symbol"""
    logger.info("Creating Chart 3: Performance Heatmap...")
    
    # Create pivot table
    pivot_data = results_df.pivot_table(
        values='slippage_bps',
        index='model',
        columns='symbol',
        aggfunc='mean'
    )
    
    # Reorder for better visualization
    model_order = ['dqn_guided', 'ppo_strategic', 'vwap', 'twap', 'random']
    pivot_data = pivot_data.reindex([m for m in model_order if m in pivot_data.index])
    pivot_data.index = [MODEL_NAMES.get(m, m) for m in pivot_data.index]
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Create heatmap
    sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='RdYlGn_r', center=0,
                cbar_kws={'label': 'Slippage (bps)'}, ax=ax,
                linewidths=1, linecolor='white', vmin=-50, vmax=30)
    
    ax.set_title('Per-Symbol Performance: DQN vs PPO vs Baselines\n(Green = Better, Red = Worse)', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xlabel('Stock Symbol', fontsize=13, fontweight='bold')
    ax.set_ylabel('Model', fontsize=13, fontweight='bold')
    
    plt.xticks(fontsize=11, rotation=0)
    plt.yticks(fontsize=11, rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved: {output_path}")
    plt.close()


# CHART 4: STATISTICAL SUMMARY (BAR CHART)
def create_statistical_summary(results_df, output_path):
    """Bar chart with error bars showing mean ± std"""
    logger.info("Creating Chart 4: Statistical Summary...")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Calculate statistics
    stats = results_df.groupby('model')['slippage_bps'].agg(['mean', 'std']).reset_index()
    stats['model_name'] = stats['model'].map(MODEL_NAMES)
    stats = stats.sort_values('mean')
    
    # Create bar chart
    bars = ax.bar(range(len(stats)), stats['mean'], 
                  yerr=stats['std'],
                  color=[COLORS.get(m, '#999999') for m in stats['model']],
                  alpha=0.7, capsize=10, error_kw={'linewidth': 2, 'ecolor': 'black'},
                  edgecolor='black', linewidth=2)
    
    # Styling
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Break-even')
    ax.set_ylabel('Slippage (basis points)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Model', fontsize=13, fontweight='bold')
    ax.set_title('Algorithm Comparison: Mean ± Standard Deviation\n(Lower is Better = Higher Profitability)', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(range(len(stats)))
    ax.set_xticklabels(stats['model_name'], rotation=0)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=11)
    
    # Add value labels on bars
    for i, (mean, std) in enumerate(zip(stats['mean'], stats['std'])):
        ax.text(i, mean + std + 1, f'{mean:.2f}±{std:.2f}', 
               ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved: {output_path}")
    plt.close()

# MAIN
def main():
    logger.info("=" * 80)
    logger.info("GENERATING PROFESSIONAL VISUALIZATIONS")
    logger.info("=" * 80)
    logger.info("")
    
    # Load results
    results_path = Path("results/validation_results.csv")
    if not results_path.exists():
        logger.error(f"[FAILED] Results file not found: {results_path}")
        logger.error("Run validation first: python scripts/validate_models_on_real_data.py")
        return
    
    results_df = pd.read_csv(results_path)
    logger.info(f"[OK] Loaded {len(results_df)} validation episodes")
    logger.info(f"[OK] Models: {', '.join(results_df['model'].unique())}")
    logger.info("")
    
    # Create output directory
    viz_dir = Path("results/visualizations")
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate all charts
    logger.info("=" * 80)
    logger.info("GENERATING CHARTS")
    logger.info("=" * 80)
    logger.info("")
    
    create_slippage_comparison(results_df, viz_dir / "01_slippage_comparison.png")
    create_execution_efficiency(results_df, viz_dir / "02_execution_efficiency.png")
    create_performance_heatmap(results_df, viz_dir / "03_model_performance_heatmap.png")
    create_statistical_summary(results_df, viz_dir / "04_statistical_summary.png")
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("VISUALIZATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Output directory: {viz_dir}")
    logger.info("")
    logger.info("Generated files:")
    for i, file in enumerate(sorted(viz_dir.glob("*.png")), 1):
        logger.info(f"  {i}. {file.name}")
    logger.info("")

# ENTRY POINT
if __name__ == "__main__":
    Path("logs").mkdir(exist_ok=True)
    
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)