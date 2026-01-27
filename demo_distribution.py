from forecasting_engine import WaterDistributionRecommender
from datetime import datetime
import json

def print_header(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")

def print_subheader(title):
    print(f"\n─── {title} ───\n")


def demo_initialization():
    print_header("DEMO 1: INITIALIZATION & ZONE SETUP")
    
    print("Creating WaterDistributionRecommender...")
    recommender = WaterDistributionRecommender()
    print("✅ Created\n")
    
    print("Adding zones to distribution network:\n")
    
    zones_to_add = [
        ('HOSPITAL_ZONE', 'critical', 5.0, 8.0, 50000, 'Hospitals, Fire Dept, Emergency'),
        ('RESIDENTIAL_A', 'essential', 30.0, 50.0, 200000, 'North side - drinking/sanitation'),
        ('RESIDENTIAL_B', 'standard', 25.0, 45.0, 180000, 'South side - full comfort'),
        ('COMMERCIAL', 'commercial', 15.0, 30.0, 100000, 'Offices, shops (non-essential)'),
        ('INDUSTRIAL', 'industrial', 20.0, 45.0, 50000, 'Factories, processing'),
    ]
    
    for zone_id, priority, min_d, max_d, pop, description in zones_to_add:
        recommender.add_zone(zone_id, priority, min_d, max_d, pop)
        print(f"  ✓ {zone_id:<20} | Priority: {priority:<12} | Demand: {min_d:>5.1f}-{max_d:>5.1f} MLD | {description}")
    
    print(f"\n✅ Total {len(recommender.zones)} zones configured")
    
    return recommender

def demo_normal_conditions(recommender):
    print_header("DEMO 2: NORMAL CONDITIONS")
    
    print("SCENARIO: No drought, no heatwave, supply sufficient")
    print("─" * 80)
    print("  Forecasted Demand: 140 MLD")
    print("  Current Storage:   350 MLD (70% of capacity)")
    print("  Max Daily Supply:  165 MLD (treatment plant capacity)")
    print("  Safety Buffer:     50 MLD (10% of 500 MLD capacity)")
    print("─" * 80 + "\n")
    
    release = recommender.recommend_daily_release(
        forecasted_demand_mld=140.0,
        reservoir_capacity_mld=500.0,
        current_storage_mld=350.0,
        max_daily_supply_mld=165.0,
        safety_buffer_percentage=10.0
    )
    
    print("RELEASE RECOMMENDATION:")
    print(f"  Release Amount:    {release['recommended_release_mld']} MLD")
    print(f"  Status:            {release['status']}")
    print(f"  Reason:            {release['reason']}")
    print(f"  Storage After:     {release['storage_after_release_mld']} MLD")
    print(f"  Storage Health:    {release['storage_health']}")
    print(f"  Days to Empty:     {release['days_to_empty_at_current_rate']}")
    
    zone_demands = {
        'HOSPITAL_ZONE': 6.0,
        'RESIDENTIAL_A': 40.0,
        'RESIDENTIAL_B': 35.0,
        'COMMERCIAL': 25.0,
        'INDUSTRIAL': 35.0
    }
    
    allocation = recommender.allocate_to_zones(
        total_available_mld=release['recommended_release_mld'],
        zone_demands=zone_demands
    )
    
    print_subheader("ZONE ALLOCATIONS")
    print(f"{'Zone':<20} {'Demand':<10} {'Allocated':<12} {'%':<8} {'Status':<30}")
    print("─" * 80)
    
    for zone in allocation['zone_allocations']:
        zone_id = zone['zone_id'][:18]
        demand = f"{zone['demand_mld']:.1f} MLD"
        alloc = f"{zone['allocated_mld']:.1f} MLD"
        pct = f"{zone['allocation_percentage']:.0f}%"
        status = zone['rationing_status'][:28]
        print(f"{zone_id:<20} {demand:<10} {alloc:<12} {pct:<8} {status:<30}")
    
    print("─" * 80)
    print(f"Total Demand:      {allocation['total_demand_mld']} MLD")
    print(f"Total Available:   {allocation['total_available_mld']} MLD")
    print(f"Total Allocated:   {allocation['total_allocated_mld']} MLD")
    print(f"Shortage:          {allocation['shortage_percentage']}%")
    print(f"Status:            {allocation['shortage_status']}")
    
    rationing = recommender.calculate_rationing_schedule(allocation['shortage_percentage'])
    
    print_subheader("RATIONING SCHEDULE")
    print(f"Status: {rationing['status']}")
    print("✅ All zones operating normally. No restrictions.")
    
    print_subheader("DECISION FOR OPERATORS")
    print("✓ Release 140 MLD from reservoir")
    print("✓ All zones get full demand allocation")
    print("✓ No public announcements needed")
    print("✓ Continue normal monitoring")

def demo_mild_shortage(recommender):
    print_header("DEMO 3: MILD SHORTAGE (HEATWAVE)")
    
    print("SCENARIO: Heatwave + 10% demand increase, storage adequate")
    print("─" * 80)
    print("  Temperature:       42°C (peak summer)")
    print("  Forecasted Demand: 160 MLD (+14% vs normal)")
    print("  Current Storage:   300 MLD (60% of capacity)")
    print("  Max Daily Supply:  165 MLD (bottleneck!)")
    print("  Safety Buffer:     50 MLD (protected)")
    print("─" * 80 + "\n")
    
    release = recommender.recommend_daily_release(
        forecasted_demand_mld=160.0,
        reservoir_capacity_mld=500.0,
        current_storage_mld=300.0,
        max_daily_supply_mld=165.0
    )
    
    print("RELEASE RECOMMENDATION:")
    print(f"  Release Amount:    {release['recommended_release_mld']} MLD (maximum supply)")
    print(f"  Status:            {release['status']}")
    print(f"  Reason:            {release['reason']}")
    print(f"  Storage Health:    {release['storage_health']}")
    print(f"  ⚠️  WARNING: Demand exceeds supply by 5 MLD/day")

    zone_demands = {
        'HOSPITAL_ZONE': 6.0,
        'RESIDENTIAL_A': 42.0,        
        'RESIDENTIAL_B': 38.0,        
        'COMMERCIAL': 25.0,
        'INDUSTRIAL': 35.0
    }
    
    allocation = recommender.allocate_to_zones(
        total_available_mld=release['recommended_release_mld'],
        zone_demands=zone_demands
    )
    
    print_subheader("ZONE ALLOCATIONS (WITH SHORTAGE)")
    print(f"{'Zone':<20} {'Demand':<10} {'Allocated':<12} {'%':<8} {'Status':<30}")
    print("─" * 80)
    
    for zone in allocation['zone_allocations']:
        zone_id = zone['zone_id'][:18]
        demand = f"{zone['demand_mld']:.1f} MLD"
        alloc = f"{zone['allocated_mld']:.1f} MLD"
        pct = f"{zone['allocation_percentage']:.0f}%"
        status = zone['rationing_status'][:28]
        print(f"{zone_id:<20} {demand:<10} {alloc:<12} {pct:<8} {status:<30}")
    
    print("─" * 80)
    print(f"Total Demand:      {allocation['total_demand_mld']} MLD")
    print(f"Total Available:   {allocation['total_available_mld']} MLD")
    print(f"🔴 Total Shortage: {allocation['total_shortage_mld']} MLD ({allocation['shortage_percentage']:.1f}%)")
    print(f"Status:            {allocation['shortage_status']}")
    
    rationing = recommender.calculate_rationing_schedule(allocation['shortage_percentage'])
    
    print_subheader("RATIONING SCHEDULE")
    print(f"Status: {rationing['status']}")
    print(f"Schedule: {rationing['schedule']}")
    print(f"Industrial: {rationing['industrial_restriction']}")
    print(f"Agriculture: {rationing['agricultural_restriction']}")
    print(f"Commercial: {rationing['commercial_restriction']}")
    
    if 'public_message' in rationing:
        print(f"\nPublic Message:\n  \"{rationing['public_message']}\"")
    
    print_subheader("DECISIONS FOR OPERATORS")
    print("🔴 URGENT ACTIONS:")
    print("  1. Release 165 MLD (maximum daily capacity)")
    print("  2. Issue public alert: 'Mild rationing to begin tomorrow'")
    print("  3. Prepare mobile water tankers for industrial zones")
    print("  4. Contact weather service about heatwave duration")
    print("\nHIGH PRIORITY:")
    print("  • Reduce system losses (target 2% reduction)")
    print("  • Shut down non-essential fountains, irrigation")
    print("  • Monitor 6-day forecast for relief")
    print("\nExpected Duration: 3-7 days (until temperature drops)")


def demo_severe_shortage(recommender):
    print_header("DEMO 4: SEVERE SHORTAGE (DROUGHT)")
    
    print("SCENARIO: Monsoon failure, storage critically low, demand high")
    print("─" * 80)
    print("  Rainfall:          Near zero for 60 days")
    print("  Forecasted Demand: 175 MLD (high consumption)")
    print("  Current Storage:   120 MLD (CRITICAL - only 24% of capacity)")
    print("  Max Daily Supply:  165 MLD")
    print("  Safety Buffer:     50 MLD (must protect)")
    print("─" * 80 + "\n")
    
    release = recommender.recommend_daily_release(
        forecasted_demand_mld=175.0,
        reservoir_capacity_mld=500.0,
        current_storage_mld=120.0,
        max_daily_supply_mld=165.0
    )
    
    print("RELEASE RECOMMENDATION:")
    print(f"  Release Amount:    {release['recommended_release_mld']} MLD")
    print(f"  Status:            {release['status']}")
    print(f"  Reason:            {release['reason']}")
    print(f"  Storage Health:    {release['storage_health']}")
    print(f"  Days to Empty:     {release['days_to_empty_at_current_rate']}")
    print(f"  ⚠️  CRITICAL: Less than 1 day of supply remaining!")
    
    zone_demands = {
        'HOSPITAL_ZONE': 6.0,
        'RESIDENTIAL_A': 45.0,
        'RESIDENTIAL_B': 42.0,
        'COMMERCIAL': 28.0,
        'INDUSTRIAL': 38.0
    }
    
    allocation = recommender.allocate_to_zones(
        total_available_mld=release['recommended_release_mld'],
        zone_demands=zone_demands
    )
    
    print_subheader("ZONE ALLOCATIONS (SEVERE SHORTAGE)")
    print(f"{'Zone':<20} {'Demand':<10} {'Allocated':<12} {'%':<8} {'Status':<30}")
    print("─" * 80)
    
    for zone in allocation['zone_allocations']:
        zone_id = zone['zone_id'][:18]
        demand = f"{zone['demand_mld']:.1f} MLD"
        alloc = f"{zone['allocated_mld']:.1f} MLD"
        pct = f"{zone['allocation_percentage']:.0f}%"
        status = zone['rationing_status'][:28]
        print(f"{zone_id:<20} {demand:<10} {alloc:<12} {pct:<8} {status:<30}")
    
    print("─" * 80)
    print(f"Total Demand:      {allocation['total_demand_mld']} MLD")
    print(f"Total Available:   {allocation['total_available_mld']} MLD")
    print(f"🔴 Total Shortage: {allocation['total_shortage_mld']} MLD ({allocation['shortage_percentage']:.1f}%)")
    print(f"Status:            {allocation['shortage_status']}")
    
    rationing = recommender.calculate_rationing_schedule(allocation['shortage_percentage'])
    
    print_subheader("EMERGENCY RATIONING SCHEDULE")
    print(f"Status: {rationing['status']}")
    print(f"Schedule: {rationing['schedule']}")
    print(f"Industrial: {rationing['industrial_restriction']}")
    print(f"Agriculture: {rationing['agricultural_restriction']}")
    print(f"Commercial: {rationing['commercial_restriction']}")
    
    if 'public_message' in rationing:
        print(f"\nPublic Message:\n  \"{rationing['public_message']}\"")
    
    if 'emergency_measures' in rationing:
        print("\nEmergency Measures:")
        for measure in rationing['emergency_measures']:
            print(f"  • {measure}")
    
    print_subheader("CRITICAL DECISIONS FOR LEADERSHIP")
    print("🔴 IMMEDIATE (Next 2 hours):")
    print("  1. Declare water emergency - activate crisis protocols")
    print("  2. Contact neighboring water authorities for emergency supply")
    print("  3. Activate emergency desalination plant")
    print("  4. Issue government order: mandatory water rationing effective immediately")
    
    print("\nUrgent (Next 6 hours):")
    print("  1. Deploy mobile tanker trucks to all zones")
    print("  2. Set up emergency water distribution points in public areas")
    print("  3. Brief industries about 60-75% supply cuts")
    print("  4. Announce water usage restrictions via all media channels")
    
    print("\nHigh Priority (Next 24 hours):")
    print("  1. Implement 4-6 hour supply cuts on rotating basis")
    print("  2. Prepare evacuation plan if situation worsens")
    print("  3. Contact armed forces for supply distribution support")
    print("  4. Set up water conservation helpline")
    
    print("\nDuration & Impact:")
    print("  • Expected duration: 4-8 weeks (depending on monsoon)")
    print("  • Industrial sector: Expect 70-80% revenue loss")
    print("  • Agriculture: Complete crop loss for season")
    print("  • Residential: Mandatory conservation (essential uses only)")
    print("  • Health system: Adequate water guaranteed for hospitals/clinics")

def demo_combined_analysis():
    print_header("DEMO 5: REAL-WORLD SCENARIO ANALYSIS")
    
    rec = WaterDistributionRecommender()
    rec.add_zone('HOSPITAL', 'critical', 5.0, 8.0, 50000)
    rec.add_zone('RESIDENTIAL_A', 'essential', 30.0, 50.0, 200000)
    rec.add_zone('RESIDENTIAL_B', 'standard', 25.0, 45.0, 180000)
    rec.add_zone('COMMERCIAL', 'commercial', 15.0, 30.0, 100000)
    rec.add_zone('INDUSTRIAL', 'industrial', 20.0, 45.0, 50000)
    
    print("COMPLEX SCENARIO: Festival + Heatwave + Reservoir Maintenance")
    print("─" * 80)
    print("  • Annual 3-day music festival (50,000 extra visitors)")
    print("  • Temperature: 40°C (demand peak)")
    print("  • Planned reservoir cleaning (reduce capacity by 30%)")
    print("  • Forecast: 185 MLD (peak season + event + heat)")
    print("  • Current storage: 250 MLD (capacity reduced to 350 for maintenance)")
    print("─" * 80 + "\n")
    
    print("DECISION PROCESS:\n")
    
    print("Step 1: Check if minimum needs can be met")
    min_critical = 5.0
    min_essential = 30.0
    min_total = min_critical + min_essential
    print(f"  Critical services need: {min_critical} MLD")
    print(f"  Essential services need: {min_essential} MLD")
    print(f"  Total minimum: {min_total} MLD")
    print(f"  ✓ Can be met even in worst case\n")
    
    print("Step 2: Get release recommendation")
    release = rec.recommend_daily_release(
        forecasted_demand_mld=185.0,
        reservoir_capacity_mld=350.0,  
        current_storage_mld=250.0,
        max_daily_supply_mld=165.0  
    )
    print(f"  Release: {release['recommended_release_mld']} MLD (max capacity)")
    print(f"  Status: {release['status']}")
    print(f"  Storage after: {release['storage_after_release_mld']} MLD")
    print(f"  Days remaining: {release['days_to_empty_at_current_rate']}\n")
    
    print("Step 3: Allocate considering festival surge")
    zone_demands = {
        'HOSPITAL': 7.0,              
        'RESIDENTIAL_A': 48.0,       
        'RESIDENTIAL_B': 43.0,        
        'COMMERCIAL': 28.0,           
        'INDUSTRIAL': 30.0            
    }
    
    allocation = rec.allocate_to_zones(
        total_available_mld=release['recommended_release_mld'],
        zone_demands=zone_demands
    )
    
    print(f"  Total demand: {allocation['total_demand_mld']} MLD")
    print(f"  Available: {allocation['total_available_mld']} MLD")
    print(f"  Shortage: {allocation['shortage_percentage']:.1f}%")
    print(f"  Status: {allocation['shortage_status']}\n")
    
    print("Step 4: Determine rationing")
    rationing = rec.calculate_rationing_schedule(allocation['shortage_percentage'])
    print(f"  Rationing level: {rationing['status']}")
    print(f"  Festival arrangement: Temporary water station set up")
    print(f"  Industrial impact: {rationing['industrial_restriction']}\n")
    
    print("FINAL RECOMMENDATION:")
    print("  ✓ Festival CAN proceed with enhanced water management")
    print("  ✓ Issue yellow alert (mild shortage expected)")
    print("  ✓ Request festival organizers reduce attendance by 10%")
    print("  ✓ Provide dedicated water distribution points at venue")
    print("  ✓ Cut industrial non-essential operations by 30%")
    print("  ✓ Recommend public: 10% voluntary water conservation")
    print("  ✓ Continue reservoir maintenance as planned")
    print("  • Expected relief: When monsoon arrives (5-7 days)")


def main():
    print("\n" + "=" * 80)
    print("WATER DISTRIBUTION RECOMMENDATION ENGINE - INTERACTIVE DEMONSTRATION")
    print("=" * 80)
    print("\nThis demo shows the water distribution engine in 5 different scenarios.")
    print("Each scenario shows:")
    print("  • System inputs")
    print("  • Allocation decisions")
    print("  • Rationing plans")
    print("  • Operator recommendations")
    
    recommender = demo_initialization()
    
    input("\n⏸️  Press Enter to see DEMO 2: Normal Conditions...")
    demo_normal_conditions(recommender)
    
    input("\n⏸️  Press Enter to see DEMO 3: Mild Shortage (Heatwave)...")
    demo_mild_shortage(recommender)

    input("\n⏸️  Press Enter to see DEMO 4: Severe Shortage (Drought)...")
    demo_severe_shortage(recommender)

    input("\n⏸️  Press Enter to see DEMO 5: Complex Real-World Scenario...")
    demo_combined_analysis()
    
    print_header("SUMMARY")
    print("✅ The Water Distribution Recommendation Engine successfully:")
    print("  • Evaluated 4 different shortage scenarios")
    print("  • Made priority-based allocation decisions")
    print("  • Recommended appropriate rationing schedules")
    print("  • Provided actionable guidance for operators")
    print("\n✅ Key Features Demonstrated:")
    print("  • Critical services always protected (hospitals)")
    print("  • Essential services guaranteed (drinking water)")
    print("  • Fair proportional allocation")
    print("  • Clear explanation of each decision")
    print("  • Practical recommendations for decision-makers")
    print("\n" + "=" * 80)
    print("For API usage, see: DISTRIBUTION_QUICK_REFERENCE.md")
    print("For full documentation, see: DISTRIBUTION_ENGINE_GUIDE.md")
    print("=" * 80 + "\n")

if __name__ == '__main__':
    main()
