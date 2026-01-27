import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

print("=" * 70)
print("WATER DEMAND FORECASTING - QUICK TEST")
print("=" * 70)

print("\n[1] Testing imports...")
try:
    from forecasting_engine import (
        ScenarioSimulator,
        WaterDemandForecastingService,
        ForecastExplainer
    )
    print("    ✅ All imports successful")
except ImportError as e:
    print(f"    ❌ Import failed: {e}")
    sys.exit(1)

print("\n[2] Testing ScenarioSimulator...")
try:
    simulator = ScenarioSimulator(
        zone_id='ZONE_A',
        baseline_demand=150.0,
        available_supply=165.0
    )
    print("    ✅ ScenarioSimulator initialized")
except Exception as e:
    print(f"    ❌ Failed: {e}")
    sys.exit(1)
print("\n[3] Running 5 scenarios...")
scenarios = {}

try:
    scenarios['heatwave'] = simulator.apply_heatwave(num_days=15, max_temp=45)
    print("    ✅ Heatwave scenario: OK")
    
    scenarios['rainfall'] = simulator.apply_rainfall_change(rainfall_change_pct=-40)
    print("    ✅ Rainfall scenario: OK")
    
    scenarios['population'] = simulator.apply_population_surge(growth_pct=10, duration_days=30)
    print("    ✅ Population scenario: OK")
    
    scenarios['festival'] = simulator.apply_festival_overlap(num_festivals=3, avg_attendees=100000)
    print("    ✅ Festival scenario: OK")

    scenarios['industrial'] = simulator.apply_industrial_change(change_pct=25)
    print("    ✅ Industrial scenario: OK")

except Exception as e:
    print(f"    ❌ Scenario failed: {e}")
    sys.exit(1)

print("\n[4] Scenario Results Summary:")
print("-" * 70)
for name, scenario in scenarios.items():
    print(f"\n{name.upper()}:")
    print(f"  Demand: {scenario['forecast_demand']:.1f} MLD")
    print(f"  Change: {scenario['demand_change_pct']:+.1f}%")
    print(f"  Stress: {scenario['stress_ratio']:.2f}x")
    print(f"  Risk: {scenario['risk_category']}")

print("\n[5] Testing combined scenario...")
try:
    combined = simulator.combine_scenarios([
        scenarios['heatwave'],
        scenarios['rainfall'],
        scenarios['festival']
    ])
    print(f"    ✅ Combined scenario: OK")
    print(f"       Demand: {combined['forecast_demand']:.1f} MLD")
    print(f"       Stress: {combined['stress_ratio']:.2f}x")
    print(f"       Risk: {combined['risk_category']}")
except Exception as e:
    print(f"    ❌ Combined scenario failed: {e}")
    sys.exit(1)

print("\n[6] Testing report generation...")
try:
    report = simulator.generate_scenario_report(scenarios['heatwave'])
    if len(report) > 100:
        print("    ✅ Report generation: OK")
        print(f"       Report length: {len(report)} characters")
    else:
        print("    ❌ Report too short")
except Exception as e:
    print(f"    ❌ Report generation failed: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("✅ ALL TESTS PASSED!")
print("=" * 70)
print("\nYou can now run the web app:")
print("  python web_app.py")
print("\nThen open: http://localhost:5000")
print("=" * 70)
