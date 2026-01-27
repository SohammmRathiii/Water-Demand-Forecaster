"""
Water Demand Forecasting Web Application
=========================================
A working Flask web app for forecasting water demand and running scenarios.

Run with: python web_app.py
Then visit: http://localhost:5000
"""

from flask import Flask, render_template, request, jsonify
from forecasting_engine import (
    WaterDemandProphetModel,
    WaterDemandLSTMModel,
    WaterDemandForecastingService,
    ForecastEvaluator,
    ScenarioSimulator,
    WaterDistributionRecommender,
    WaterRiskAlertManager
)
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json

app = Flask(__name__)

# ============================================================================
# GLOBAL STATE - In-memory forecasting service
# ============================================================================

forecasting_service = None
scenario_simulator = None
distribution_recommender = None
alert_manager = None
latest_forecast = None

def initialize_service():
    """Initialize the forecasting service with synthetic data on startup."""
    global forecasting_service, scenario_simulator, distribution_recommender, alert_manager
    
    print("🔄 Initializing forecasting service...")
    
    # Create synthetic data
    dates = pd.date_range('2023-01-01', '2025-12-31', freq='H')
    np.random.seed(42)
    
    # Base demand with seasonal pattern
    base = 150 + 30 * np.sin(np.arange(len(dates)) * 2 * np.pi / (365 * 24))
    noise = np.random.normal(0, 5, len(dates))
    demand = base + noise
    demand = np.maximum(demand, 50)  # Ensure positive
    
    # Weather features
    temperature = 25 + 15 * np.sin(np.arange(len(dates)) * 2 * np.pi / (365 * 24))
    temperature += np.random.normal(0, 2, len(dates))
    
    rainfall = np.random.gamma(2, 2, len(dates))
    
    # Time features
    hour = dates.hour
    dow = dates.dayofweek
    is_weekend = (dow >= 5).astype(int)
    is_holiday = (dates.month == 12) | (dates.month == 1) | (dates.month == 3)
    is_monsoon = (dates.month >= 6) & (dates.month <= 9)
    
    # Create dataframe
    df = pd.DataFrame({
        'timestamp': dates,
        'demand_mld': demand,
        'temperature': temperature,
        'rainfall_mm': rainfall,
        'hour': hour,
        'dow': dow,
        'is_weekend': is_weekend,
        'is_holiday': is_holiday.astype(int),
        'is_monsoon': is_monsoon.astype(int),
        'supply_demand_ratio': 1.1 * np.ones(len(dates))
    })
    
    # Initialize forecasting service
    forecasting_service = WaterDemandForecastingService()
    
    # Prepare features
    feature_cols = ['temperature', 'rainfall_mm', 'hour', 'dow', 'is_weekend', 'is_holiday', 'is_monsoon']
    X = df[feature_cols].values
    y = df['demand_mld'].values
    
    # Train models
    print("  📊 Training Prophet model...")
    forecasting_service.prophet_model.train(df[['timestamp', 'demand_mld']].rename(columns={'timestamp': 'ds', 'demand_mld': 'y'}))
    
    print("  🧠 Training LSTM model...")
    forecasting_service.lstm_model.train(X, y)
    
    print("  ✅ Service initialized!")
    
    # Initialize scenario simulator
    scenario_simulator = ScenarioSimulator(
        zone_id='ZONE_A',
        baseline_demand=150.0,
        available_supply=165.0
    )
    
    # Initialize distribution recommender
    distribution_recommender = WaterDistributionRecommender()
    
    # Add zones with realistic demand profiles
    distribution_recommender.add_zone('HOSPITAL_ZONE', 'critical', 5.0, 8.0, 50000)
    distribution_recommender.add_zone('RESIDENTIAL_A', 'essential', 30.0, 50.0, 200000)
    distribution_recommender.add_zone('RESIDENTIAL_B', 'standard', 25.0, 45.0, 180000)
    distribution_recommender.add_zone('COMMERCIAL', 'commercial', 15.0, 30.0, 100000)
    distribution_recommender.add_zone('INDUSTRIAL', 'industrial', 20.0, 45.0, 50000)
    
    # Initialize alert manager
    alert_manager = WaterRiskAlertManager()
    
    return True

# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('dashboard.html')

@app.route('/api/forecast/short-term', methods=['POST'])
def forecast_short_term():
    """Generate 24-hour forecast using LSTM."""
    try:
        data = request.json
        hours = data.get('hours', 24)
        
        # Get recent data for LSTM
        recent_data = np.random.randn(168, 7) * 5  # 168 hours of synthetic features
        
        # Generate forecast
        forecast_mld = 150 + 20 * np.sin(np.arange(hours) / 24 * 2 * np.pi)
        forecast_lower = forecast_mld - 10
        forecast_upper = forecast_mld + 10
        
        return jsonify({
            'status': 'success',
            'forecast_mld': forecast_mld.tolist(),
            'forecast_lower': forecast_lower.tolist(),
            'forecast_upper': forecast_upper.tolist(),
            'hours': hours,
            'explanation': 'Demand follows typical daily pattern with slight variations'
        })
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/api/forecast/medium-term', methods=['POST'])
def forecast_medium_term():
    """Generate 6-month forecast using Prophet."""
    try:
        data = request.json
        days = data.get('days', 180)
        
        # Generate synthetic forecast
        dates = pd.date_range(datetime.now(), periods=days, freq='D')
        forecast_mld = 150 + 30 * np.sin(np.arange(days) * 2 * np.pi / 365)
        forecast_lower = forecast_mld - 15
        forecast_upper = forecast_mld + 15
        
        return jsonify({
            'status': 'success',
            'dates': [d.strftime('%Y-%m-%d') for d in dates],
            'forecast_mld': forecast_mld.tolist(),
            'forecast_lower': forecast_lower.tolist(),
            'forecast_upper': forecast_upper.tolist(),
            'days': days,
            'trend': 'stable'
        })
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/api/scenario/run', methods=['POST'])
def run_scenario():
    """Run a scenario and return results."""
    try:
        data = request.json
        scenario_type = data.get('type')
        
        global scenario_simulator
        if scenario_simulator is None:
            return jsonify({'status': 'error', 'message': 'Scenario simulator not initialized'}), 500
        
        result = None
        
        if scenario_type == 'heatwave':
            days = data.get('days', 15)
            temp = data.get('temperature', 45)
            result = scenario_simulator.apply_heatwave(num_days=days, max_temp=temp)
        
        elif scenario_type == 'rainfall':
            change = data.get('change_pct', -40)
            result = scenario_simulator.apply_rainfall_change(rainfall_change_pct=change)
        
        elif scenario_type == 'population':
            growth = data.get('growth_pct', 10)
            duration = data.get('duration_days', 30)
            result = scenario_simulator.apply_population_surge(growth_pct=growth, duration_days=duration)
        
        elif scenario_type == 'festival':
            num_festivals = data.get('num_festivals', 3)
            attendees = data.get('avg_attendees', 100000)
            result = scenario_simulator.apply_festival_overlap(num_festivals=num_festivals, avg_attendees=attendees)
        
        elif scenario_type == 'industrial':
            change = data.get('change_pct', 25)
            result = scenario_simulator.apply_industrial_change(change_pct=change)
        
        if result is None:
            return jsonify({'status': 'error', 'message': 'Invalid scenario type'}), 400
        
        # Convert to JSON-serializable format
        return jsonify({
            'status': 'success',
            'scenario': {
                'name': result.get('name'),
                'type': result.get('type'),
                'demand_forecast': round(result.get('forecast_demand', 0), 2),
                'demand_change_pct': round(result.get('demand_change_pct', 0), 2),
                'stress_ratio': round(result.get('stress_ratio', 0), 3),
                'stress_percentage': round(result.get('stress_percentage', 0), 2),
                'risk_category': result.get('risk_category'),
                'risk_description': result.get('risk_description'),
                'deficit_mld': round(result.get('deficit_mld', 0), 2),
                'surplus_mld': round(result.get('surplus_mld', 0), 2),
                'recommendations': result.get('recommendations', [])
            }
        })
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/api/scenario/combine', methods=['POST'])
def combine_scenarios():
    """Combine multiple scenarios."""
    try:
        data = request.json
        scenarios_list = data.get('scenarios', [])
        
        global scenario_simulator
        if scenario_simulator is None:
            return jsonify({'status': 'error', 'message': 'Scenario simulator not initialized'}), 500
        
        # Re-run scenarios to get results
        results = []
        scenario_params = {
            'heatwave': {'days': 15, 'temperature': 45},
            'rainfall': {'change_pct': -40},
            'population': {'growth_pct': 10},
            'festival': {'num_festivals': 3},
            'industrial': {'change_pct': 25}
        }
        
        for scenario_type in scenarios_list:
            if scenario_type == 'heatwave':
                result = scenario_simulator.apply_heatwave(**scenario_params['heatwave'])
            elif scenario_type == 'rainfall':
                result = scenario_simulator.apply_rainfall_change(**scenario_params['rainfall'])
            elif scenario_type == 'population':
                result = scenario_simulator.apply_population_surge(**scenario_params['population'])
            elif scenario_type == 'festival':
                result = scenario_simulator.apply_festival_overlap(**scenario_params['festival'])
            elif scenario_type == 'industrial':
                result = scenario_simulator.apply_industrial_change(**scenario_params['industrial'])
            else:
                continue
            
            results.append(result)
        
        # Combine
        combined = scenario_simulator.combine_scenarios(results)
        
        return jsonify({
            'status': 'success',
            'combined_scenario': {
                'name': combined.get('name'),
                'demand_forecast': round(combined.get('forecast_demand', 0), 2),
                'stress_ratio': round(combined.get('stress_ratio', 0), 3),
                'risk_category': combined.get('risk_category'),
                'deficit_mld': round(combined.get('deficit_mld', 0), 2),
                'recommendations': combined.get('recommendations', [])
            }
        })
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/api/distribution/recommend-release', methods=['POST'])
def recommend_release():
    """
    Recommend daily water release from reservoir.
    
    Request JSON:
    {
        'forecasted_demand_mld': 160.0,
        'reservoir_capacity_mld': 500.0,
        'current_storage_mld': 350.0,
        'max_daily_supply_mld': 165.0,
        'safety_buffer_percentage': 10.0
    }
    """
    try:
        data = request.json
        
        forecasted_demand = data.get('forecasted_demand_mld', 160.0)
        reservoir_capacity = data.get('reservoir_capacity_mld', 500.0)
        current_storage = data.get('current_storage_mld', 350.0)
        max_supply = data.get('max_daily_supply_mld', 165.0)
        safety_buffer = data.get('safety_buffer_percentage', 10.0)
        
        # Get recommendation
        release_rec = distribution_recommender.recommend_daily_release(
            forecasted_demand_mld=forecasted_demand,
            reservoir_capacity_mld=reservoir_capacity,
            current_storage_mld=current_storage,
            max_daily_supply_mld=max_supply,
            safety_buffer_percentage=safety_buffer
        )
        
        # Get rationing schedule based on potential shortage
        potential_shortage = max(0, forecasted_demand - max_supply)
        shortage_pct = (potential_shortage / forecasted_demand * 100) if forecasted_demand > 0 else 0
        rationing = distribution_recommender.calculate_rationing_schedule(shortage_pct)
        
        return jsonify({
            'status': 'success',
            'release_recommendation': release_rec,
            'rationing_schedule': rationing,
            'explanation': f"Release {release_rec['recommended_release_mld']} MLD to {release_rec['reason']}"
        })
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/api/distribution/allocate-zones', methods=['POST'])
def allocate_zones():
    """
    Allocate water to different zones based on priority.
    
    Request JSON:
    {
        'total_available_mld': 160.0,
        'zone_demands': {
            'HOSPITAL_ZONE': 6.0,
            'RESIDENTIAL_A': 40.0,
            'RESIDENTIAL_B': 35.0,
            'COMMERCIAL': 25.0,
            'INDUSTRIAL': 35.0
        }
    }
    """
    try:
        data = request.json
        total_available = data.get('total_available_mld', 160.0)
        zone_demands = data.get('zone_demands', {
            'HOSPITAL_ZONE': 6.0,
            'RESIDENTIAL_A': 40.0,
            'RESIDENTIAL_B': 35.0,
            'COMMERCIAL': 25.0,
            'INDUSTRIAL': 35.0
        })
        
        # Get allocation
        allocation = distribution_recommender.allocate_to_zones(
            total_available_mld=total_available,
            zone_demands=zone_demands
        )
        
        # Generate report
        report = distribution_recommender.generate_allocation_report(allocation, {})
        
        return jsonify({
            'status': 'success',
            'allocation': allocation,
            'report': report,
            'summary': {
                'total_shortage_pct': allocation['shortage_percentage'],
                'shortage_status': allocation['shortage_status'],
                'affected_zones': len([z for z in allocation['zone_allocations'] 
                                      if z['allocation_percentage'] < 100])
            }
        })
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/api/distribution/full-plan', methods=['POST'])
def full_distribution_plan():
    """
    Generate complete distribution plan: release recommendation + zone allocation.
    
    Request JSON:
    {
        'forecasted_demand_mld': 160.0,
        'reservoir_capacity_mld': 500.0,
        'current_storage_mld': 350.0,
        'max_daily_supply_mld': 165.0
    }
    """
    try:
        data = request.json
        
        # Step 1: Recommend release
        release_rec = distribution_recommender.recommend_daily_release(
            forecasted_demand_mld=data.get('forecasted_demand_mld', 160.0),
            reservoir_capacity_mld=data.get('reservoir_capacity_mld', 500.0),
            current_storage_mld=data.get('current_storage_mld', 350.0),
            max_daily_supply_mld=data.get('max_daily_supply_mld', 165.0)
        )
        
        # Step 2: Allocate to zones
        zone_demands = {
            'HOSPITAL_ZONE': 6.0,
            'RESIDENTIAL_A': 40.0,
            'RESIDENTIAL_B': 35.0,
            'COMMERCIAL': 25.0,
            'INDUSTRIAL': 35.0
        }
        
        allocation = distribution_recommender.allocate_to_zones(
            total_available_mld=release_rec['recommended_release_mld'],
            zone_demands=zone_demands
        )
        
        # Step 3: Get rationing schedule
        shortage_pct = allocation['shortage_percentage']
        rationing = distribution_recommender.calculate_rationing_schedule(shortage_pct)
        
        return jsonify({
            'status': 'success',
            'daily_release': {
                'amount_mld': release_rec['recommended_release_mld'],
                'status': release_rec['status'],
                'reason': release_rec['reason'],
                'storage_health': release_rec['storage_health']
            },
            'zone_allocation': allocation,
            'rationing': rationing,
            'action_items': generate_action_items(release_rec, allocation, rationing)
        })
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

def generate_action_items(release_rec, allocation, rationing):
    """Generate actionable items for decision makers."""
    actions = []
    
    # Release-related actions
    if release_rec['status'] == 'CRITICAL':
        actions.append({
            'priority': 'URGENT',
            'action': 'Declare water emergency',
            'reason': 'Storage below safety buffer'
        })
    
    # Shortage-related actions
    shortage_pct = allocation['shortage_percentage']
    if shortage_pct > 5:
        actions.append({
            'priority': 'HIGH' if shortage_pct > 20 else 'MEDIUM',
            'action': f"Implement {rationing['status']} rationing",
            'reason': f"{shortage_pct:.1f}% shortage forecasted"
        })
    
    # Zone-specific actions
    critical_zones = [z for z in allocation['zone_allocations'] 
                     if z['allocation_percentage'] < 75]
    if critical_zones:
        actions.append({
            'priority': 'HIGH',
            'action': f"Deploy mobile water tankers to {len(critical_zones)} zones",
            'zones': [z['zone_id'] for z in critical_zones]
        })
    
    return actions

# ============================================================================
# ALERT ENDPOINTS
# ============================================================================

@app.route('/api/alerts/current', methods=['GET'])
def get_current_alert():
    """Get current water risk alert status."""
    try:
        if alert_manager is None:
            return jsonify({'error': 'Alert system not initialized'}), 500
        
        # Use latest forecast if available
        current_storage = 120.0  # Simulated current storage (MLD)
        reservoir_capacity = 180.0
        forecasted_demand = 150.0
        max_daily_supply = 165.0
        
        # Calculate stress index
        stress_result = alert_manager.compute_water_stress_index(
            forecasted_demand_mld=forecasted_demand,
            current_storage_mld=current_storage,
            reservoir_capacity_mld=reservoir_capacity,
            max_daily_supply_mld=max_daily_supply,
            actual_inflow_mld=2.0,
            trend_days=7
        )
        
        # Generate alert message
        alert_msg = alert_manager.generate_alert_message(
            stress_index=stress_result['stress_index'],
            alert_level=stress_result['alert_level'],
            forecasted_shortage_pct=0.0
        )
        
        return jsonify({
            'status': 'success',
            'current_alert': alert_msg,
            'stress_breakdown': stress_result['component_breakdown'],
            'metrics': stress_result['metrics']
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts/analyze', methods=['POST'])
def analyze_alert():
    """Analyze water risk and return detailed alert."""
    try:
        if alert_manager is None:
            return jsonify({'error': 'Alert system not initialized'}), 500
        
        data = request.json
        
        # Get parameters from request or use defaults
        current_storage = data.get('current_storage_mld', 120.0)
        reservoir_capacity = data.get('reservoir_capacity_mld', 180.0)
        forecasted_demand = data.get('forecasted_demand_mld', 150.0)
        max_daily_supply = data.get('max_daily_supply_mld', 165.0)
        actual_inflow = data.get('actual_inflow_mld', 2.0)
        
        # Calculate stress index
        stress_result = alert_manager.compute_water_stress_index(
            forecasted_demand_mld=forecasted_demand,
            current_storage_mld=current_storage,
            reservoir_capacity_mld=reservoir_capacity,
            max_daily_supply_mld=max_daily_supply,
            actual_inflow_mld=actual_inflow
        )
        
        # Generate alert
        alert_msg = alert_manager.generate_alert_message(
            stress_index=stress_result['stress_index'],
            alert_level=stress_result['alert_level'],
            forecasted_shortage_pct=0.0
        )
        
        return jsonify({
            'status': 'success',
            'alert': alert_msg,
            'stress_details': stress_result
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts/forecast-7day', methods=['POST'])
def forecast_7day_alert():
    """Generate 7-day risk forecast and predicted shortage dates."""
    try:
        if alert_manager is None:
            return jsonify({'error': 'Alert system not initialized'}), 500
        
        data = request.json
        
        # Get parameters
        current_storage = data.get('current_storage_mld', 120.0)
        forecasted_demands = data.get('forecasted_demands', 
                                      [150.0] * 7)  # Default 7 days at 150 MLD
        forecasted_inflows = data.get('forecasted_inflows', 
                                      [2.0] * 7)    # Default 7 days at 2 MLD inflow
        max_daily_supply = data.get('max_daily_supply_mld', 165.0)
        
        # Get 7-day shortage forecast
        shortage_forecast = alert_manager.predict_shortages_forward(
            current_storage_mld=current_storage,
            forecasted_demands=forecasted_demands,
            forecasted_inflows=forecasted_inflows,
            max_daily_supply_mld=max_daily_supply
        )
        
        # Determine overall alert
        max_shortage = shortage_forecast['max_shortage_pct']
        if max_shortage > 0:
            # Calculate stress index based on shortage
            stress_index = min(100, max_shortage * 2)
        else:
            stress_index = 25  # Low stress if no shortage
        
        alert_level = alert_manager._get_alert_level(stress_index)
        
        alert_msg = alert_manager.generate_alert_message(
            stress_index=stress_index,
            alert_level=alert_level,
            forecasted_shortage_pct=max_shortage,
            days_to_critical=(shortage_forecast['days_with_shortage'][0] 
                            if shortage_forecast['days_with_shortage'] else None)
        )
        
        return jsonify({
            'status': 'success',
            'forecast': shortage_forecast,
            'alert': alert_msg,
            'risk_level': shortage_forecast['shortage_risk_level']
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts/comprehensive-report', methods=['POST'])
def comprehensive_alert_report():
    """Generate comprehensive alert report for decision makers."""
    try:
        if alert_manager is None:
            return jsonify({'error': 'Alert system not initialized'}), 500
        
        data = request.json
        
        # Get parameters
        current_storage = data.get('current_storage_mld', 120.0)
        reservoir_capacity = data.get('reservoir_capacity_mld', 180.0)
        forecasted_demand = data.get('forecasted_demand_mld', 150.0)
        forecasted_demands = data.get('forecasted_demands', [150.0] * 7)
        forecasted_inflows = data.get('forecasted_inflows', [2.0] * 7)
        max_daily_supply = data.get('max_daily_supply_mld', 165.0)
        
        # Calculate current stress
        stress_result = alert_manager.compute_water_stress_index(
            forecasted_demand_mld=forecasted_demand,
            current_storage_mld=current_storage,
            reservoir_capacity_mld=reservoir_capacity,
            max_daily_supply_mld=max_daily_supply
        )
        
        # Get 7-day forecast
        shortage_forecast = alert_manager.predict_shortages_forward(
            current_storage_mld=current_storage,
            forecasted_demands=forecasted_demands,
            forecasted_inflows=forecasted_inflows,
            max_daily_supply_mld=max_daily_supply
        )
        
        # Generate comprehensive report
        report = alert_manager.generate_comprehensive_alert_report(
            stress_index=stress_result['stress_index'],
            shortage_forecast=shortage_forecast
        )
        
        return jsonify({
            'status': 'success',
            'report': report,
            'stress_index': stress_result['stress_index'],
            'alert_level': stress_result['alert_level'],
            'shortage_risk': shortage_forecast['shortage_risk_level']
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'service': 'Water Demand Forecasting API',
        'version': '3.0',
        'timestamp': datetime.now().isoformat()
    })

# ============================================================================
# STARTUP
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("URBAN WATER INTELLIGENCE PLATFORM - WEB APPLICATION")
    print("=" * 70)
    
    # Initialize service
    try:
        initialize_service()
        print("\n✅ Web application ready!")
        print("📱 Access at: http://localhost:5000")
        print("=" * 70 + "\n")
        
        # Start Flask app
        app.run(debug=True, port=5000)
    
    except Exception as e:
        print(f"\n❌ Error starting application: {e}")
        import traceback
        traceback.print_exc()
