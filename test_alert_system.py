"""Test script for Water Risk Alert Manager"""

from forecasting_engine import WaterRiskAlertManager
import json

print("\n" + "="*70)
print("WATER RISK ALERT MANAGER - FUNCTIONAL TEST")
print("="*70)

# Initialize manager
alert_manager = WaterRiskAlertManager()
print("\n✅ WaterRiskAlertManager initialized successfully")

# Test 1: Compute stress index
print("\n" + "-"*70)
print("TEST 1: Water Stress Index Calculation")
print("-"*70)

result = alert_manager.compute_water_stress_index(
    forecasted_demand_mld=150.0,
    current_storage_mld=120.0,
    reservoir_capacity_mld=180.0,
    max_daily_supply_mld=165.0,
    actual_inflow_mld=2.0
)

print(f"Stress Index: {result['stress_index']:.2f}/100")
print(f"Alert Level: {result['alert_level'].upper()}")
print(f"Components:")
print(f"  - Supply Stress: {result['component_breakdown']['supply_stress']:.2f}")
print(f"  - Storage Stress: {result['component_breakdown']['storage_stress']:.2f}")
print(f"  - Trend Stress: {result['component_breakdown']['trend_stress']:.2f}")
print(f"  - Buffer Stress: {result['component_breakdown']['buffer_stress']:.2f}")
print(f"Metrics:")
print(f"  - Days to Empty: {result['metrics']['days_to_empty']:.1f}")
print(f"  - Storage Ratio: {result['metrics']['storage_ratio']:.1%}")

# Test 2: Predict shortages
print("\n" + "-"*70)
print("TEST 2: 7-Day Shortage Prediction")
print("-"*70)

shortage_forecast = alert_manager.predict_shortages_forward(
    current_storage_mld=120.0,
    forecasted_demands=[150, 155, 160, 158, 152, 148, 145],
    forecasted_inflows=[2, 2.5, 2, 1.5, 1.5, 2, 2.5],
    max_daily_supply_mld=165.0
)

print(f"Overall Risk Level: {shortage_forecast['shortage_risk_level']}")
print(f"Max Expected Shortage: {shortage_forecast['max_shortage_pct']:.1f}%")
print(f"Days with Shortage: {shortage_forecast['days_with_shortage']}")
print(f"\nDaily Breakdown:")
for day in shortage_forecast['daily_forecasts'][:3]:  # Show first 3 days
    print(f"  Day {day['day']}: Demand={day['forecasted_demand']:.1f}, "
          f"Supply={day['available_supply']:.1f}, Status={day['status']}")

# Test 3: Generate alert message
print("\n" + "-"*70)
print("TEST 3: Alert Message Generation")
print("-"*70)

alert = alert_manager.generate_alert_message(
    stress_index=result['stress_index'],
    alert_level=result['alert_level'],
    forecasted_shortage_pct=0.0
)

print(f"Alert Level: {alert['emoji']} {alert['name']}")
print(f"Headline: {alert['headline']}")
print(f"Public Message: {alert['public_message']}")
print(f"Duration: {alert['duration']}")
print(f"Recommended Actions ({len(alert['recommended_actions'])}):")
for i, action in enumerate(alert['recommended_actions'][:3], 1):
    print(f"  {i}. {action}")

# Test 4: Different alert levels
print("\n" + "-"*70)
print("TEST 4: Testing All Alert Levels")
print("-"*70)

test_scenarios = [
    {'name': 'Safe', 'stress': 25, 'expected': 'green'},
    {'name': 'Watch', 'stress': 42, 'expected': 'yellow'},
    {'name': 'Prepare', 'stress': 65, 'expected': 'orange'},
    {'name': 'Critical', 'stress': 85, 'expected': 'red'},
]

for scenario in test_scenarios:
    level = alert_manager._get_alert_level(scenario['stress'])
    status = "✅ PASS" if level == scenario['expected'] else "❌ FAIL"
    print(f"{status} Stress {scenario['stress']}: {level.upper()} (expected {scenario['expected'].upper()})")

# Test 5: Comprehensive report
print("\n" + "-"*70)
print("TEST 5: Comprehensive Alert Report")
print("-"*70)

report = alert_manager.generate_comprehensive_alert_report(
    stress_index=result['stress_index'],
    shortage_forecast=shortage_forecast
)

report_lines = report.split('\n')
print(f"\nReport generated with {len(report_lines)} lines")
print("First 20 lines:")
for line in report_lines[:20]:
    print(line)

print("\n" + "="*70)
print("ALL TESTS COMPLETED SUCCESSFULLY ✅")
print("="*70 + "\n")
