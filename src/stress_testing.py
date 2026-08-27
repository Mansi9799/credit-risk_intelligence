import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt

# Configuration
DATA_PATH = r"C:\Users\MY PC\OneDrive\ドキュメント\Rainmeter\Desktop\credit-risk-intelligence\processed_data\expected_loss_dataset.parquet"
OUTPUT_DIR = r"C:\Users\MY PC\OneDrive\ドキュメント\Rainmeter\Desktop\credit-risk-intelligence\stress_test_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define CCAR / Basel Macroeconomic Stress Scenarios
# These multipliers simulate how macroeconomic shocks (e.g., Unemployment spikes, GDP contraction)
# impact the Probability of Default (PD) and Loss Given Default (LGD).
SCENARIOS = {
    "Baseline": {
        "description": "Current economic conditions",
        "PD_Multiplier": 1.00,
        "LGD_Multiplier": 1.00
    },
    "Adverse": {
        "description": "Moderate recession, rising unemployment",
        "PD_Multiplier": 1.40,
        "LGD_Multiplier": 1.15
    },
    "Severely Adverse": {
        "description": "Severe global recession, housing market crash",
        "PD_Multiplier": 2.20,
        "LGD_Multiplier": 1.35
    }
}

def simulate_stress_test(df):
    print("Running CCAR / Basel Macroeconomic Stress Test...")
    results = {}
    
    total_ead = df['EAD'].sum()
    
    for scenario_name, params in SCENARIOS.items():
        # Apply shocks (ensuring probabilities/percentages don't exceed 1.0)
        stressed_pd = np.clip(df['PD'] * params['PD_Multiplier'], 0, 1.0)
        stressed_lgd = np.clip(df['LGD'] * params['LGD_Multiplier'], 0, 1.0)
        
        # Calculate Stressed Expected Loss: EL = PD * LGD * EAD
        stressed_el = stressed_pd * stressed_lgd * df['EAD']
        
        total_stressed_el = stressed_el.sum()
        portfolio_el_rate = total_stressed_el / total_ead
        
        results[scenario_name] = {
            "Total_EAD_Billions": float(round(total_ead / 1e9, 4)),
            "Total_Expected_Loss_Billions": float(round(total_stressed_el / 1e9, 4)),
            "Portfolio_Expected_Loss_Rate": float(round(portfolio_el_rate, 4)),
            "PD_Multiplier": float(params['PD_Multiplier']),
            "LGD_Multiplier": float(params['LGD_Multiplier'])
        }
        
    return results

def plot_stress_test(results):
    scenarios = list(results.keys())
    el_billions = [results[s]["Total_Expected_Loss_Billions"] for s in scenarios]
    el_rates = [results[s]["Portfolio_Expected_Loss_Rate"] * 100 for s in scenarios]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Bar chart for Expected Loss in Billions
    bars = ax1.bar(scenarios, el_billions, color=['#4CAF50', '#FF9800', '#F44336'], alpha=0.8)
    ax1.set_ylabel("Total Expected Loss ($ Billions)", fontsize=12, fontweight='bold')
    ax1.set_title("CCAR Macroeconomic Stress Test: Portfolio Expected Loss", fontsize=14, fontweight='bold')
    
    # Add value labels on bars
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f"${yval:.2f}B", ha='center', va='bottom', fontweight='bold')
        
    # Line chart for Portfolio EL Rate
    ax2 = ax1.twinx()
    ax2.plot(scenarios, el_rates, color='black', marker='o', linestyle='dashed', linewidth=2, markersize=8)
    ax2.set_ylabel("Portfolio EL Rate (%)", fontsize=12, fontweight='bold')
    ax2.set_ylim(0, max(el_rates) + 5)
    
    for i, rate in enumerate(el_rates):
        ax2.text(i, rate + 1, f"{rate:.1f}%", ha='center', va='bottom', color='black', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "01_stress_test_expected_loss.png"), dpi=300)
    plt.close()

def main():
    print("Loading Expected Loss Dataset...")
    if not os.path.exists(DATA_PATH):
        print(f"Error: Could not find {DATA_PATH}. Ensure Stage 7 was run.")
        return
        
    df = pd.read_parquet(DATA_PATH, engine='pyarrow')
    
    # Run the stress test
    stress_results = simulate_stress_test(df)
    
    # Save JSON report
    with open(os.path.join(OUTPUT_DIR, "ccar_stress_test_report.json"), "w") as f:
        json.dump(stress_results, f, indent=4)
        
    # Generate visualization
    plot_stress_test(stress_results)
    
    print("\n--- CCAR Stress Test Results ---")
    for scenario, metrics in stress_results.items():
        print(f"\n[{scenario} Scenario]")
        print(f"Total Portfolio EAD: ${metrics['Total_EAD_Billions']:.2f} Billion")
        print(f"Expected Loss: ${metrics['Total_Expected_Loss_Billions']:.2f} Billion")
        print(f"Portfolio EL Rate: {metrics['Portfolio_Expected_Loss_Rate']*100:.2f}%")
        
    print("\nStage 11: Macroeconomic Stress Testing complete.")

if __name__ == "__main__":
    main()
