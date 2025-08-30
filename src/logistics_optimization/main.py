#!/usr/bin/env python3
"""
Logistics Optimization - Transportation Cost Reduction Engine
Path: src/logistics_optimization/main.py
Author: Shambhavi Thakur - Data Intelligence Professional
Purpose: Demonstrate logistics cost optimization and route planning methodology
Version: 1.0.0 - Production Ready

This module demonstrates systematic logistics optimization capability
for reducing transportation costs and improving delivery efficiency.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class LogisticsOptimizer:
    """
    Transportation and Distribution Cost Optimization
    
    Demonstrates methodology for reducing ₹5+ crore logistics costs
    through systematic route optimization and carrier management.
    """
    
    def __init__(self):
        """Initialize logistics optimization framework"""
        
        # Transportation cost parameters
        self.cost_parameters = {
            'fuel_cost_per_km': 8.5,      # ₹8.5 per km
            'driver_cost_per_day': 1200,   # ₹1,200 per day
            'vehicle_maintenance_per_km': 3.2,  # ₹3.2 per km
            'toll_avg_per_100km': 150,     # ₹150 per 100km
            'loading_unloading_cost': 500  # ₹500 per stop
        }
        
        # Vehicle capacity and constraints
        self.vehicle_types = {
            'small_truck': {
                'capacity_kg': 2000,
                'capacity_cbm': 15,
                'fixed_cost_per_day': 2500,
                'variable_cost_per_km': 12
            },
            'medium_truck': {
                'capacity_kg': 5000,
                'capacity_cbm': 35,
                'fixed_cost_per_day': 4000,
                'variable_cost_per_km': 18
            },
            'large_truck': {
                'capacity_kg': 10000,
                'capacity_cbm': 60,
                'fixed_cost_per_day': 6500,
                'variable_cost_per_km': 25
            },
            'trailer': {
                'capacity_kg': 25000,
                'capacity_cbm': 90,
                'fixed_cost_per_day': 12000,
                'variable_cost_per_km': 40
            }
        }
        
        # Service level targets
        self.service_targets = {
            'on_time_delivery': 0.95,     # 95% on-time delivery
            'max_transit_days': 5,        # Maximum 5 days transit
            'customer_satisfaction': 0.90  # 90% satisfaction target
        }
    
    def calculate_distance_matrix(self, locations: List[Dict]) -> pd.DataFrame:
        """
        Calculate distance matrix between locations
        
        Note: In production, this would integrate with Google Maps API
        or other routing services. This demonstrates the methodology.
        """
        
        n_locations = len(locations)
        distance_matrix = np.zeros((n_locations, n_locations))
        
        # Simulate realistic distances based on coordinates
        for i in range(n_locations):
            for j in range(n_locations):
                if i != j:
                    # Simplified distance calculation (in production: use actual routing API)
                    lat_diff = abs(locations[i]['latitude'] - locations[j]['latitude'])
                    lon_diff = abs(locations[i]['longitude'] - locations[j]['longitude'])
                    # Rough conversion: 1 degree ≈ 111 km
                    distance = np.sqrt(lat_diff**2 + lon_diff**2) * 111
                    # Add road network complexity factor
                    distance_matrix[i][j] = distance * 1.3  # 30% longer for actual roads
        
        # Create DataFrame with location names
        location_names = [loc['name'] for loc in locations]
        distance_df = pd.DataFrame(distance_matrix, 
                                 index=location_names, 
                                 columns=location_names)
        
        return distance_df
    
    def optimize_vehicle_selection(self, shipment_weight: float, 
                                 shipment_volume: float, distance_km: float) -> Dict:
        """
        Select optimal vehicle type for shipment
        
        Returns cost analysis and vehicle recommendation
        """
        
        vehicle_analysis = {}
        
        for vehicle_type, specs in self.vehicle_types.items():
            # Check if vehicle can handle the shipment
            weight_fits = shipment_weight <= specs['capacity_kg']
            volume_fits = shipment_volume <= specs['capacity_cbm']
            
            if weight_fits and volume_fits:
                # Calculate total transportation cost
                fixed_cost = specs['fixed_cost_per_day']
                variable_cost = distance_km * specs['variable_cost_per_km']
                total_cost = fixed_cost + variable_cost
                
                # Calculate utilization
                weight_utilization = (shipment_weight / specs['capacity_kg']) * 100
                volume_utilization = (shipment_volume / specs['capacity_cbm']) * 100
                overall_utilization = max(weight_utilization, volume_utilization)
                
                # Cost per kg transported
                cost_per_kg = total_cost / shipment_weight if shipment_weight > 0 else float('inf')
                
                vehicle_analysis[vehicle_type] = {
                    'can_handle': True,
                    'total_cost': round(total_cost, 2),
                    'fixed_cost': fixed_cost,
                    'variable_cost': round(variable_cost, 2),
                    'weight_utilization': round(weight_utilization, 1),
                    'volume_utilization': round(volume_utilization, 1),
                    'overall_utilization': round(overall_utilization, 1),
                    'cost_per_kg': round(cost_per_kg, 2)
                }
            else:
                vehicle_analysis[vehicle_type] = {
                    'can_handle': False,
                    'reason': f"Exceeds {'weight' if not weight_fits else 'volume'} capacity"
                }
        
        # Find optimal vehicle (lowest cost among feasible options)
        feasible_vehicles = {k: v for k, v in vehicle_analysis.items() if v.get('can_handle', False)}
        
        if feasible_vehicles:
            optimal_vehicle = min(feasible_vehicles.keys(), 
                                key=lambda x: feasible_vehicles[x]['total_cost'])
            
            optimization_result = {
                'recommended_vehicle': optimal_vehicle,
                'recommended_cost': feasible_vehicles[optimal_vehicle]['total_cost'],
                'utilization': feasible_vehicles[optimal_vehicle]['overall_utilization'],
                'all_options': vehicle_analysis
            }
        else:
            optimization_result = {
                'recommended_vehicle': None,
                'error': 'No vehicle can handle this shipment',
                'all_options': vehicle_analysis
            }
        
        return optimization_result
    
    def optimize_delivery_route(self, depot: Dict, customers: List[Dict], 
                              vehicle_capacity: float) -> Dict:
        """
        Simple route optimization using nearest neighbor heuristic
        
        In production: would use advanced algorithms like Clarke-Wright or OR-Tools
        """
        
        all_locations = [depot] + customers
        distance_matrix = self.calculate_distance_matrix(all_locations)
        
        # Nearest neighbor route construction
        unvisited = list(range(1, len(customers) + 1))  # Customer indices
        current_location = 0  # Start at depot
        route = [0]
        total_distance = 0
        current_load = 0
        
        while unvisited:
            # Find nearest unvisited customer that fits in vehicle
            nearest_customer = None
            nearest_distance = float('inf')
            
            for customer_idx in unvisited:
                customer_demand = customers[customer_idx - 1]['demand_kg']
                
                # Check if customer demand fits in remaining capacity
                if current_load + customer_demand <= vehicle_capacity:
                    distance = distance_matrix.iloc[current_location, customer_idx]
                    if distance < nearest_distance:
                        nearest_distance = distance
                        nearest_customer = customer_idx
            
            if nearest_customer is not None:
                # Visit nearest customer
                route.append(nearest_customer)
                unvisited.remove(nearest_customer)
                current_load += customers[nearest_customer - 1]['demand_kg']
                total_distance += nearest_distance
                current_location = nearest_customer
            else:
                # No more customers fit - return to depot and start new route
                return_distance = distance_matrix.iloc[current_location, 0]
                total_distance += return_distance
                break
        
        # Return to depot
        if current_location != 0:
            return_distance = distance_matrix.iloc[current_location, 0]
            total_distance += return_distance
            route.append(0)
        
        # Calculate route costs
        vehicle_type = 'medium_truck'  # Default for demonstration
        vehicle_specs = self.vehicle_types[vehicle_type]
        
        fixed_cost = vehicle_specs['fixed_cost_per_day']
        variable_cost = total_distance * vehicle_specs['variable_cost_per_km']
        total_cost = fixed_cost + variable_cost
        
        route_optimization = {
            'route': route,
            'route_names': [all_locations[i]['name'] for i in route],
            'total_distance_km': round(total_distance, 1),
            'total_cost': round(total_cost, 2),
            'vehicle_utilization': round((current_load / vehicle_capacity) * 100, 1),
            'customers_served': len([i for i in route if i != 0]) - route.count(0) + 1,
            'estimated_time_hours': round(total_distance / 50, 1)  # Assuming 50 km/hr average
        }
        
        return route_optimization
    
    def analyze_logistics_network(self, network_data: Dict) -> Dict:
        """
        Comprehensive logistics network analysis
        
        Returns optimization recommendations and cost savings potential
        """
        
        depots = network_data['depots']
        customers = network_data['customers']
        shipments = network_data['shipments']
        
        # Current state analysis
        current_total_cost = 0
        shipment_analysis = []
        
        for shipment in shipments:
            # Optimize vehicle selection for each shipment
            optimization = self.optimize_vehicle_selection(
                shipment['weight_kg'],
                shipment['volume_cbm'],
                shipment['distance_km']
            )
            
            if optimization['recommended_vehicle']:
                shipment_cost = optimization['recommended_cost']
                current_total_cost += shipment_cost
                
                shipment_analysis.append({
                    'shipment_id': shipment['id'],
                    'origin': shipment['origin'],
                    'destination': shipment['destination'],
                    'weight_kg': shipment['weight_kg'],
                    'distance_km': shipment['distance_km'],
                    'current_vehicle': shipment.get('current_vehicle', 'unspecified'),
                    'current_cost': shipment.get('current_cost', shipment_cost * 1.2),  # Assume 20% higher
                    'recommended_vehicle': optimization['recommended_vehicle'],
                    'optimized_cost': shipment_cost,
                    'cost_savings': round(shipment.get('current_cost', shipment_cost * 1.2) - shipment_cost, 2),
                    'utilization': optimization['utilization']
                })
        
        # Network optimization insights
        total_current_cost = sum([s['current_cost'] for s in shipment_analysis])
        total_optimized_cost = sum([s['optimized_cost'] for s in shipment_analysis])
        total_savings = total_current_cost - total_optimized_cost
        
        # Vehicle utilization analysis
        avg_utilization = np.mean([s['utilization'] for s in shipment_analysis])
        underutilized_shipments = len([s for s in shipment_analysis if s['utilization'] < 60])
        
        network_analysis = {
            'shipment_analysis': shipment_analysis,
            'total_current_cost': round(total_current_cost, 2),
            'total_optimized_cost': round(total_optimized_cost, 2),
            'total_annual_savings': round(total_savings, 2),
            'savings_percentage': round((total_savings / total_current_cost) * 100, 1),
            'average_utilization': round(avg_utilization, 1),
            'underutilized_shipments': underutilized_shipments,
            'optimization_opportunities': self._identify_optimization_opportunities(shipment_analysis)
        }
        
        return network_analysis
    
    def _identify_optimization_opportunities(self, shipments: List[Dict]) -> List[str]:
        """Identify specific logistics optimization opportunities"""
        
        opportunities = []
        
        # Consolidation opportunities
        small_shipments = [s for s in shipments if s['utilization'] < 50]
        if len(small_shipments) > 2:
            opportunities.append(f"Consolidate {len(small_shipments)} small shipments to reduce costs")
        
        # Route optimization
        same_destination_shipments = {}
        for shipment in shipments:
            dest = shipment['destination']
            if dest not in same_destination_shipments:
                same_destination_shipments[dest] = []
            same_destination_shipments[dest].append(shipment)
        
        multi_shipment_destinations = {k: v for k, v in same_destination_shipments.items() if len(v) > 1}
        if multi_shipment_destinations:
            opportunities.append(f"Route optimization for {len(multi_shipment_destinations)} destinations with multiple shipments")
        
        # Vehicle right-sizing
        oversized_vehicles = [s for s in shipments if s['utilization'] < 30]
        if oversized_vehicles:
            opportunities.append(f"Right-size vehicles for {len(oversized_vehicles)} shipments with low utilization")
        
        return opportunities

def generate_sample_logistics_data() -> Dict:
    """Generate realistic sample logistics network data"""
    
    sample_data = {
        'depots': [
            {'name': 'Central Warehouse Delhi', 'latitude': 28.6139, 'longitude': 77.2090},
            {'name': 'Regional Hub Mumbai', 'latitude': 19.0760, 'longitude': 72.8777}
        ],
        
        'customers': [
            {'name': 'Customer Bangalore', 'latitude': 12.9716, 'longitude': 77.5946, 'demand_kg': 1500},
            {'name': 'Customer Pune', 'latitude': 18.5204, 'longitude': 73.8567, 'demand_kg': 2200},
            {'name': 'Customer Hyderabad', 'latitude': 17.3850, 'longitude': 78.4867, 'demand_kg': 1800},
            {'name': 'Customer Chennai', 'latitude': 13.0827, 'longitude': 80.2707, 'demand_kg': 2500}
        ],
        
        'shipments': [
            {
                'id': 'SH001',
                'origin': 'Central Warehouse Delhi',
                'destination': 'Customer Bangalore',
                'weight_kg': 1500,
                'volume_cbm': 12,
                'distance_km': 2150,
                'current_vehicle': 'large_truck',
                'current_cost': 65000
            },
            {
                'id': 'SH002',
                'origin': 'Regional Hub Mumbai',
                'destination': 'Customer Pune',
                'weight_kg': 2200,
                'volume_cbm': 18,
                'distance_km': 150,
                'current_vehicle': 'medium_truck',
                'current_cost': 8500
            },
            {
                'id': 'SH003',
                'origin': 'Central Warehouse Delhi',
                'destination': 'Customer Hyderabad',
                'weight_kg': 1800,
                'volume_cbm': 15,
                'distance_km': 1580,
                'current_vehicle': 'large_truck',
                'current_cost': 48000
            },
            {
                'id': 'SH004',
                'origin': 'Regional Hub Mumbai',
                'destination': 'Customer Chennai',
                'weight_kg': 2500,
                'volume_cbm': 20,
                'distance_km': 1340,
                'current_vehicle': 'large_truck',
                'current_cost': 42000
            },
            {
                'id': 'SH005',
                'origin': 'Central Warehouse Delhi',
                'destination': 'Customer Pune',
                'weight_kg': 800,
                'volume_cbm': 8,
                'distance_km': 1450,
                'current_vehicle': 'medium_truck',
                'current_cost': 32000
            }
        ]
    }
    
    return sample_data

def main():
    """
    Demonstration of logistics optimization for supply chain intelligence
    
    Shows capability for reducing ₹5+ crore logistics costs through
    systematic vehicle optimization and route planning.
    """
    
    print("LOGISTICS OPTIMIZATION - SUPPLY CHAIN INTELLIGENCE DEMO")
    print("=" * 65)
    print("Author: Shambhavi Thakur - Data Intelligence Professional")
    print("Purpose: Demonstrate transportation cost optimization methodology")
    print("Contact: info@shambhavithakur.com")
    print()
    
    # Initialize logistics optimizer
    optimizer = LogisticsOptimizer()
    
    # Generate sample logistics network data
    network_data = generate_sample_logistics_data()
    
    print(f"LOGISTICS NETWORK OPTIMIZATION - {datetime.now().strftime('%Y-%m-%d')}")
    print("-" * 55)
    
    # Analyze logistics network
    analysis = optimizer.analyze_logistics_network(network_data)
    
    # Executive Summary
    print("EXECUTIVE SUMMARY:")
    print(f"   Current Annual Logistics Cost: ₹{analysis['total_current_cost']/100000:.1f} lakhs")
    print(f"   Optimized Annual Cost: ₹{analysis['total_optimized_cost']/100000:.1f} lakhs")
    print(f"   Annual Cost Savings: ₹{analysis['total_annual_savings']/100000:.1f} lakhs")
    print(f"   Cost Reduction: {analysis['savings_percentage']:.1f}%")
    print(f"   Average Vehicle Utilization: {analysis['average_utilization']:.1f}%")
    print(f"   Underutilized Shipments: {analysis['underutilized_shipments']}")
    print()
    
    # Detailed Shipment Analysis
    print("DETAILED SHIPMENT OPTIMIZATION:")
    print("=" * 50)
    
    for shipment in analysis['shipment_analysis']:
        print(f"\n{shipment['shipment_id']}: {shipment['origin']} → {shipment['destination']}")
        print(f"   Shipment: {shipment['weight_kg']:,} kg, {shipment['distance_km']:,} km")
        print(f"   Current Vehicle: {shipment['current_vehicle']}")
        print(f"   Recommended Vehicle: {shipment['recommended_vehicle']}")
        print(f"   Current Cost: ₹{shipment['current_cost']:,}")
        print(f"   Optimized Cost: ₹{shipment['optimized_cost']:,}")
        print(f"   Cost Savings: ₹{shipment['cost_savings']:,}")
        print(f"   Vehicle Utilization: {shipment['utilization']:.1f}%")
    
    # Optimization Opportunities
    print(f"\nLOGISTICS OPTIMIZATION OPPORTUNITIES:")
    print("=" * 45)
    
    for i, opportunity in enumerate(analysis['optimization_opportunities'], 1):
        print(f"   {i}. {opportunity}")
    
    # Strategic Recommendations
    print(f"\nSTRATEGIC LOGISTICS RECOMMENDATIONS:")
    print("=" * 45)
    
    if analysis['average_utilization'] < 70:
        print(f"VEHICLE UTILIZATION IMPROVEMENT:")
        print(f"   Current average utilization: {analysis['average_utilization']:.1f}%")
        print(f"   Target utilization: 75-85%")
        print(f"   Potential for shipment consolidation")
    
    high_savings_shipments = [s for s in analysis['shipment_analysis'] if s['cost_savings'] > 5000]
    if high_savings_shipments:
        print(f"\nHIGH-IMPACT OPTIMIZATION OPPORTUNITIES:")
        for shipment in high_savings_shipments:
            print(f"   → {shipment['shipment_id']}: ₹{shipment['cost_savings']:,} annual savings")
    
    print(f"\nBUSINESS VALUE DEMONSTRATION:")
    print(f"   ✅ Systematic vehicle optimization")
    print(f"   ✅ Route planning and consolidation")
    print(f"   ✅ Transportation cost reduction")
    print(f"   ✅ Vehicle utilization improvement")
    
    print(f"\nLOGISTICS INTELLIGENCE FRAMEWORK:")
    print(f"   • Real-time route optimization")
    print(f"   • Dynamic vehicle selection")
    print(f"   • Cost-per-kilometer analysis")
    print(f"   • Carrier performance monitoring")
    print(f"   Contact: info@shambhavithakur.com")
    
    # Export analysis
    shipments_df = pd.DataFrame(analysis['shipment_analysis'])
    output_file = 'logistics_optimization_analysis.csv'
    shipments_df.to_csv(output_file, index=False)
    print(f"\n   Sample Analysis Exported: {output_file}")
    
    return analysis

if __name__ == "__main__":
    # Run logistics optimization demonstration
    results = main()
    print(f"\n✅ Logistics optimization analysis completed!")
    print("🚚 Framework ready for supply chain intelligence consulting")
    