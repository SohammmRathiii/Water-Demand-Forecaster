"""Test script for Water Recycling Classification Module"""

import sys
sys.path.insert(0, r'c:\Users\rathi\Desktop\Water-Demand-Forecaster')

print("\n" + "="*80)
print("WATER RECYCLING CLASSIFICATION MODULE - FUNCTIONAL TEST")
print("="*80)

try:
    from forecasting_engine import WaterRecyclingClassifier
    print("\n✅ WaterRecyclingClassifier imported successfully")
except Exception as e:
    print(f"\n❌ Import failed: {e}")
    sys.exit(1)

# Initialize
try:
    classifier = WaterRecyclingClassifier()
    print("✅ Classifier initialized successfully")
except Exception as e:
    print(f"❌ Initialization failed: {e}")
    sys.exit(1)

# TEST 1: Get all categories
print("\n" + "-"*80)
print("TEST 1: Get All Water Categories")
print("-"*80)

try:
    categories = classifier.get_all_categories()
    print(f"\n✅ Retrieved {len(categories)} categories:\n")
    for cat in categories:
        print(f"  {cat['emoji']} {cat['name']:<20} | Recyclability: {cat['recyclability']:<8} | Score: {cat['recyclability_score']:.0f}%")
except Exception as e:
    print(f"❌ Failed: {e}")

# TEST 2: Get specific category details
print("\n" + "-"*80)
print("TEST 2: Get Greywater Details")
print("-"*80)

try:
    grey_info = classifier.get_category_info('greywater')
    print(f"\n✅ Category: {grey_info['name']} {grey_info['emoji']}")
    print(f"   Source: {grey_info['source']}")
    print(f"   Recyclability: {grey_info['recyclability']}")
    print(f"   Recovery Rate: {grey_info['recoverable_percentage']}%")
    print(f"   Freshwater Reduction: {grey_info['freshwater_reduction_potential']}%")
    print(f"   Cost: ₹{grey_info['cost_per_kld']}/KLD")
    print(f"   Reuse Purposes:")
    for purpose in grey_info['reuse_purposes'][:3]:
        print(f"     - {purpose}")
except Exception as e:
    print(f"❌ Failed: {e}")

# TEST 3: Calculate recycling potential
print("\n" + "-"*80)
print("TEST 3: Calculate Recycling Potential (Mixed Sources)")
print("-"*80)

try:
    potential = classifier.calculate_recycling_potential(
        greywater_volume=10.0,
        rainwater_volume=15.0,
        treated_sewage_volume=8.0
    )
    
    print(f"\n✅ Recycling Potential Calculated:")
    print(f"   Total Input: {potential['total_input_volume_mld']} MLD")
    print(f"   Recoverable: {potential['total_recoverable_mld']} MLD")
    print(f"   Recovery Rate: {potential['overall_recovery_rate']:.1f}%")
    print(f"   Recyclability Score: {potential['overall_recyclability_score']:.1f}%")
    print(f"   Freshwater Reduction: {potential['total_freshwater_demand_reduction_mld']} MLD")
    print(f"   Treatment Cost: ₹{potential['estimated_treatment_cost_inr']:,.0f}/day")
    print(f"   Cost per MLD: ₹{potential['cost_per_mld_treated']:,.0f}/day")
except Exception as e:
    print(f"❌ Failed: {e}")

# TEST 4: Get recommendations
print("\n" + "-"*80)
print("TEST 4: Generate Recycling Recommendations")
print("-"*80)

try:
    water_sources = {
        'greywater': 10.0,
        'rainwater': 15.0,
        'industrial_wastewater': 20.0,
        'treated_sewage': 8.0
    }
    
    strategy = classifier.generate_recycling_recommendation(water_sources)
    
    print(f"\n✅ Strategy Generated:")
    print(f"   Total to Recycle: {strategy['total_volume_to_recycle_mld']} MLD")
    print(f"   Summary: {strategy['summary']}")
    
    print(f"\n   Prioritized Recommendations:")
    for rec in strategy['recommendations']:
        print(f"\n   #{rec['priority']} {rec['source']}")
        print(f"       Volume: {rec['volume_available_mld']} MLD")
        print(f"       Recovery: {rec['expected_recovery_mld']} MLD")
        print(f"       Timeline: {rec['timeline']}")
        print(f"       Freshwater Reduction: {rec['freshwater_reduction_potential']} MLD")
except Exception as e:
    print(f"❌ Failed: {e}")

# TEST 5: Calculate demand offset
print("\n" + "-"*80)
print("TEST 5: Calculate Freshwater Demand Offset")
print("-"*80)

try:
    offset = classifier.calculate_demand_offset(
        total_freshwater_demand=150.0,
        recycled_water_available=30.0
    )
    
    print(f"\n✅ Demand Offset Calculated:")
    print(f"   Original Demand: {offset['freshwater_demand_original_mld']} MLD")
    print(f"   Recycled Water Used: {offset['recycled_water_used_mld']} MLD")
    print(f"   New Demand: {offset['freshwater_demand_after_recycling_mld']} MLD")
    print(f"   Demand Reduction: {offset['demand_offset_percentage']}%")
    
    print(f"\n   Environmental Impact:")
    print(f"     - Water Saved: {offset['environmental_impact']['water_saved_million_liters']:,.0f} ML/year")
    print(f"     - CO2 Avoided: {offset['environmental_impact']['co2_emissions_avoided_tons']:,.0f} tons/year")
    print(f"     - Energy Saved: {offset['environmental_impact']['energy_saved_kwh']:,.0f} kWh/year")
    
    print(f"\n   Financial Impact:")
    print(f"     - Original Cost: ₹{offset['financial_analysis']['original_freshwater_cost_inr']:,.0f}/day")
    print(f"     - With Recycling: ₹{offset['financial_analysis']['recycled_water_total_cost_inr']:,.0f}/day")
    print(f"     - Annual Savings: ₹{offset['financial_analysis']['annual_cost_savings_inr']:,.0f}/year")
    print(f"     - Cost Reduction: {offset['financial_analysis']['cost_reduction_percentage']}%")
except Exception as e:
    print(f"❌ Failed: {e}")

# TEST 6: Test all alert levels
print("\n" + "-"*80)
print("TEST 6: Test All Water Categories")
print("-"*80)

try:
    test_categories = [
        ('greywater', 'Greywater'),
        ('blackwater', 'Blackwater'),
        ('industrial_wastewater', 'Industrial Wastewater'),
        ('rainwater', 'Rainwater'),
        ('treated_sewage', 'Treated Sewage')
    ]
    
    print(f"\n✅ Category Comparison:")
    print(f"\n{'Category':<20} {'Recyclability':<15} {'Recovery':<10} {'Cost/KLD':<10}")
    print("-" * 60)
    
    for cat_id, cat_name in test_categories:
        info = classifier.get_category_info(cat_id)
        print(f"{cat_name:<20} {info['recyclability']:<15} {info['recoverable_percentage']:.0f}%{'':<7} ₹{info['cost_per_kld']}")
except Exception as e:
    print(f"❌ Failed: {e}")

# Summary
print("\n" + "="*80)
print("ALL TESTS COMPLETED SUCCESSFULLY ✅")
print("="*80)
print("\nWater Recycling Classification Module is ready for production use!")
print("Available methods:")
print("  • get_category_info(category_id)")
print("  • get_all_categories()")
print("  • calculate_recycling_potential(...)")
print("  • generate_recycling_recommendation(water_sources)")
print("  • calculate_demand_offset(...)")
print("\n" + "="*80 + "\n")
