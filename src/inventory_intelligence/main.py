#!/usr/bin/env python3
"""
Inventory Intelligence - Stock Optimization Engine
Path: src/inventory_intelligence/main.py
Author: Shambhavi Thakur - Data Intelligence Professional
Purpose: Demonstrate inventory optimization and carrying cost analysis methodology
Version: 1.0.0 - Production Ready

This module demonstrates systematic inventory optimization capability
for reducing carrying costs while maintaining optimal service levels.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class InventoryOptimizer:
    """
    Inventory Intelligence and Stock Level Optimization
    
    Demonstrates methodology for reducing ₹8+ crore inventory carrying costs
    through systematic demand forecasting and stock optimization.
    """
    
    def __init__(self):
        """Initialize inventory optimization framework"""
        
        # Inventory cost parameters
        self.cost_parameters = {
            'holding_cost_rate': 0.25,  # 25% annual holding cost
            'ordering_cost': 5000,      # ₹5,000 per order
            'stockout_cost_multiplier': 3.0,  # 3x unit cost for stockouts
            'obsolescence_rate': 0.08    # 8% annual obsolescence
        }
        
        # Product categories with different optimization strategies
        self.product_categories = {
            'fast_moving': {
                'turnover_target': 12,  # 12x annual turnover
                'service_level': 0.98,  # 98% service level
                'safety_stock_days': 7
            },
            'medium_moving': {
                'turnover_target': 6,   # 6x annual turnover
                'service_level': 0.95,  # 95% service level
                'safety_stock_days': 14
            },
            'slow_moving': {
                'turnover_target': 3,   # 3x annual turnover
                'service_level': 0.90,  # 90% service level
                'safety_stock_days': 30
            },
            'critical_spare': {
                'turnover_target': 2,   # 2x annual turnover
                'service_level': 0.99,  # 99% service level
                'safety_stock_days': 60
            }
        }
        
        # ABC analysis thresholds
        self.abc_thresholds = {
            'A_items': 0.80,  # 80% of value
            'B_items': 0.95   # 95% of value (cumulative)
        }
    
    def calculate_economic_order_quantity(self, annual_demand: float, 
                                        unit_cost: float, ordering_cost: float = None) -> Dict:
        """
        Calculate optimal order quantity and related metrics
        
        Args:
            annual_demand: Annual demand quantity
            unit_cost: Cost per unit
            ordering_cost: Cost per order (optional)
            
        Returns:
            Dictionary with EOQ analysis results
        """
        
        if ordering_cost is None:
            ordering_cost = self.cost_parameters['ordering_cost']
        
        holding_cost = unit_cost * self.cost_parameters['holding_cost_rate']
        
        # Classic EOQ formula
        eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost)
        
        # Total annual costs
        ordering_cost_annual = (annual_demand / eoq) * ordering_cost
        holding_cost_annual = (eoq / 2) * holding_cost
        total_cost_annual = ordering_cost_annual + holding_cost_annual
        
        # Order frequency
        orders_per_year = annual_demand / eoq
        days_between_orders = 365 / orders_per_year
        
        analysis = {
            'optimal_order_quantity': round(eoq, 0),
            'total_annual_cost': round(total_cost_annual, 2),
            'ordering_cost_annual': round(ordering_cost_annual, 2),
            'holding_cost_annual': round(holding_cost_annual, 2),
            'orders_per_year': round(orders_per_year, 1),
            'days_between_orders': round(days_between_orders, 0),
            'unit_holding_cost': round(holding_cost, 2)
        }
        
        return analysis
    
    def calculate_safety_stock(self, daily_demand_avg: float, daily_demand_std: float,
                             lead_time_days: int, service_level: float) -> Dict:
        """
        Calculate safety stock requirements for target service level
        
        Returns safety stock analysis with cost implications
        """
        
        # Z-score for service level (normal distribution)
        z_scores = {0.90: 1.28, 0.95: 1.65, 0.98: 1.96, 0.99: 2.33}
        z_score = z_scores.get(service_level, 1.65)
        
        # Lead time demand variability
        lead_time_demand_std = daily_demand_std * np.sqrt(lead_time_days)
        
        # Safety stock calculation
        safety_stock = z_score * lead_time_demand_std
        
        # Reorder point
        reorder_point = (daily_demand_avg * lead_time_days) + safety_stock
        
        analysis = {
            'safety_stock_units': round(safety_stock, 0),
            'reorder_point': round(reorder_point, 0),
            'service_level': service_level,
            'lead_time_days': lead_time_days,
            'average_daily_demand': daily_demand_avg,
            'demand_variability': round(daily_demand_std, 2)
        }
        
        return analysis
    
    def perform_abc_analysis(self, inventory_data: List[Dict]) -> pd.DataFrame:
        """
        Perform ABC analysis for inventory prioritization
        
        Classifies inventory items by value contribution for optimization focus
        """
        
        # Calculate annual value for each item
        items_df = pd.DataFrame(inventory_data)
        items_df['annual_value'] = items_df['annual_demand'] * items_df['unit_cost']
        
        # Sort by annual value (descending)
        items_df = items_df.sort_values('annual_value', ascending=False).reset_index(drop=True)
        
        # Calculate cumulative percentage
        total_value = items_df['annual_value'].sum()
        items_df['cumulative_value'] = items_df['annual_value'].cumsum()
        items_df['cumulative_percentage'] = items_df['cumulative_value'] / total_value
        
        # Classify items
        def classify_item(cum_pct):
            if cum_pct <= self.abc_thresholds['A_items']:
                return 'A'
            elif cum_pct <= self.abc_thresholds['B_items']:
                return 'B'
            else:
                return 'C'
        
        items_df['abc_category'] = items_df['cumulative_percentage'].apply(classify_item)
        items_df['value_percentage'] = (items_df['annual_value'] / total_value) * 100
        
        return items_df
    
    def optimize_inventory_levels(self, item_data: Dict) -> Dict:
        """
        Comprehensive inventory optimization for a single item
        
        Returns optimized stock levels and cost analysis
        """
        
        # Extract item parameters
        annual_demand = item_data['annual_demand']
        unit_cost = item_data['unit_cost']
        lead_time_days = item_data.get('lead_time_days', 15)
        category = item_data.get('category', 'medium_moving')
        
        # Demand variability (estimated if not provided)
        daily_demand_avg = annual_demand / 365
        daily_demand_std = item_data.get('demand_std', daily_demand_avg * 0.3)
        
        # Category-specific parameters
        cat_params = self.product_categories.get(category, self.product_categories['medium_moving'])
        service_level = cat_params['service_level']
        
        # EOQ Analysis
        eoq_analysis = self.calculate_economic_order_quantity(annual_demand, unit_cost)
        
        # Safety Stock Analysis
        safety_analysis = self.calculate_safety_stock(
            daily_demand_avg, daily_demand_std, lead_time_days, service_level
        )
        
        # Current vs Optimized Comparison
        current_stock = item_data.get('current_stock_level', annual_demand / 6)  # 2-month default
        optimal_avg_stock = (eoq_analysis['optimal_order_quantity'] / 2) + safety_analysis['safety_stock_units']
        
        # Cost Analysis
        current_holding_cost = current_stock * unit_cost * self.cost_parameters['holding_cost_rate']
        optimal_holding_cost = optimal_avg_stock * unit_cost * self.cost_parameters['holding_cost_rate']
        annual_savings = current_holding_cost - optimal_holding_cost
        
        # Inventory turnover
        optimal_turnover = annual_demand / optimal_avg_stock
        current_turnover = annual_demand / current_stock
        
        optimization_results = {
            'item_name': item_data['item_name'],
            'category': category,
            'annual_demand': annual_demand,
            'unit_cost': unit_cost,
            'current_stock_level': round(current_stock, 0),
            'optimal_avg_stock': round(optimal_avg_stock, 0),
            'optimal_order_quantity': eoq_analysis['optimal_order_quantity'],
            'safety_stock': safety_analysis['safety_stock_units'],
            'reorder_point': safety_analysis['reorder_point'],
            'service_level': service_level,
            'current_turnover': round(current_turnover, 1),
            'optimal_turnover': round(optimal_turnover, 1),
            'current_holding_cost': round(current_holding_cost, 2),
            'optimal_holding_cost': round(optimal_holding_cost, 2),
            'annual_savings': round(annual_savings, 2),
            'days_between_orders': eoq_analysis['days_between_orders'],
            'stock_reduction_units': round(current_stock - optimal_avg_stock, 0),
            'stock_reduction_percentage': round(((current_stock - optimal_avg_stock) / current_stock) * 100, 1)
        }
        
        return optimization_results
    
    def generate_inventory_optimization_report(self, inventory_items: List[Dict]) -> pd.DataFrame:
        """
        Generate comprehensive inventory optimization report
        
        Returns DataFrame with optimization recommendations for all items
        """
        
        optimization_results = []
        
        for item in inventory_items:
            result = self.optimize_inventory_levels(item)
            optimization_results.append(result)
        
        # Convert to DataFrame and add summary metrics
        results_df = pd.DataFrame(optimization_results)
        
        # Add ABC analysis
        abc_data = []
        for item in inventory_items:
            abc_data.append({
                'item_name': item['item_name'],
                'annual_demand': item['annual_demand'],
                'unit_cost': item['unit_cost']
            })
        
        abc_analysis = self.perform_abc_analysis(abc_data)
        
        # Merge ABC classification
        results_df = results_df.merge(
            abc_analysis[['item_name', 'abc_category', 'value_percentage']], 
            on='item_name', 
            how='left'
        )
        
        # Sort by potential savings
        results_df = results_df.sort_values('annual_savings', ascending=False)
        
        return results_df

def generate_sample_inventory_data() -> List[Dict]:
    """Generate realistic sample inventory data for demonstration"""
    
    sample_inventory = [
        {
            'item_name': 'High-Value Component A',
            'category': 'fast_moving',
            'annual_demand': 12000,
            'unit_cost': 2500,
            'current_stock_level': 4000,
            'lead_time_days': 10,
            'demand_std': 35
        },
        {
            'item_name': 'Critical Spare Part B',
            'category': 'critical_spare',
            'annual_demand': 2400,
            'unit_cost': 15000,
            'current_stock_level': 1200,
            'lead_time_days': 45,
            'demand_std': 12
        },
        {
            'item_name': 'Standard Material C',
            'category': 'medium_moving',
            'annual_demand': 8000,
            'unit_cost': 800,
            'current_stock_level': 2500,
            'lead_time_days': 20,
            'demand_std': 25
        },
        {
            'item_name': 'Slow-Moving Item D',
            'category': 'slow_moving',
            'annual_demand': 1200,
            'unit_cost': 5000,
            'current_stock_level': 800,
            'lead_time_days': 30,
            'demand_std': 8
        },
        {
            'item_name': 'Bulk Raw Material E',
            'category': 'fast_moving',
            'annual_demand': 36000,
            'unit_cost': 150,
            'current_stock_level': 8000,
            'lead_time_days': 7,
            'demand_std': 120
        }
    ]
    
    return sample_inventory

def main():
    """
    Demonstration of inventory optimization for supply chain intelligence
    
    Shows capability for reducing ₹8+ crore inventory carrying costs through
    systematic stock level optimization and demand-driven planning.
    """
    
    print("INVENTORY INTELLIGENCE - SUPPLY CHAIN OPTIMIZATION DEMO")
    print("=" * 65)
    print("Author: Shambhavi Thakur - Data Intelligence Professional")
    print("Purpose: Demonstrate inventory optimization methodology")
    print("Contact: info@shambhavithakur.com")
    print()
    
    # Initialize inventory optimizer
    optimizer = InventoryOptimizer()
    
    # Generate sample inventory data
    inventory_items = generate_sample_inventory_data()
    
    print(f"INVENTORY OPTIMIZATION ANALYSIS - {datetime.now().strftime('%Y-%m-%d')}")
    print("-" * 55)
    
    # Generate comprehensive optimization report
    optimization_report = optimizer.generate_inventory_optimization_report(inventory_items)
    
    # Executive Summary
    total_current_value = (optimization_report['current_stock_level'] * optimization_report['unit_cost']).sum()
    total_optimal_value = (optimization_report['optimal_avg_stock'] * optimization_report['unit_cost']).sum()
    total_savings = optimization_report['annual_savings'].sum()
    total_stock_reduction = optimization_report['stock_reduction_units'].sum()
    
    print("EXECUTIVE SUMMARY:")
    print(f"   Current Inventory Value: ₹{total_current_value/100000:.1f} lakhs")
    print(f"   Optimized Inventory Value: ₹{total_optimal_value/100000:.1f} lakhs")
    print(f"   Annual Holding Cost Savings: ₹{total_savings/100000:.1f} lakhs")
    print(f"   Inventory Reduction: {total_stock_reduction:,.0f} units")
    print(f"   Value Release: ₹{(total_current_value - total_optimal_value)/100000:.1f} lakhs")
    print()
    
    # Detailed Item Analysis
    print("DETAILED INVENTORY OPTIMIZATION:")
    print("=" * 50)
    
    for _, item in optimization_report.iterrows():
        print(f"\n{item['item_name'].upper()}")
        print(f"   ABC Category: {item['abc_category']} (Value: {item['value_percentage']:.1f}%)")
        print(f"   Current Stock: {item['current_stock_level']:,.0f} units")
        print(f"   Optimal Stock: {item['optimal_avg_stock']:,.0f} units")
        print(f"   Recommended Order Qty: {item['optimal_order_quantity']:,.0f} units")
        print(f"   Reorder Point: {item['reorder_point']:,.0f} units")
        print(f"   Service Level: {item['service_level']*100:.0f}%")
        print(f"   Current Turnover: {item['current_turnover']:.1f}x annually")
        print(f"   Optimal Turnover: {item['optimal_turnover']:.1f}x annually")
        print(f"   Annual Savings: ₹{item['annual_savings']/100000:.1f} lakhs")
        if item['stock_reduction_units'] > 0:
            print(f"   Stock Reduction: {item['stock_reduction_units']:,.0f} units ({item['stock_reduction_percentage']:.1f}%)")
        print(f"   Order Frequency: Every {item['days_between_orders']:.0f} days")
    
    # Strategic Recommendations
    print(f"\nSTRATEGIC INVENTORY RECOMMENDATIONS:")
    print("=" * 45)
    
    # High-impact items
    high_savings_items = optimization_report[optimization_report['annual_savings'] > 100000]  # >₹1 lakh
    if len(high_savings_items) > 0:
        print(f"HIGH-IMPACT OPTIMIZATION OPPORTUNITIES:")
        for _, item in high_savings_items.iterrows():
            print(f"   → {item['item_name']}: ₹{item['annual_savings']/100000:.1f} lakhs annual savings")
    
    # ABC Analysis insights
    a_items = optimization_report[optimization_report['abc_category'] == 'A']
    b_items = optimization_report[optimization_report['abc_category'] == 'B']
    c_items = optimization_report[optimization_report['abc_category'] == 'C']
    
    print(f"\nINVENTORY CATEGORY INSIGHTS:")
    print(f"   A-Items ({len(a_items)}): Focus on service level optimization")
    print(f"   B-Items ({len(b_items)}): Balance cost and service")
    print(f"   C-Items ({len(c_items)}): Minimize carrying costs")
    
    print(f"\nBUSINESS VALUE DEMONSTRATION:")
    print(f"   ✅ Systematic inventory optimization")
    print(f"   ✅ ABC analysis for prioritization")
    print(f"   ✅ Service level maintenance")
    print(f"   ✅ Carrying cost reduction")
    
    print(f"\nINVENTORY INTELLIGENCE FRAMEWORK:")
    print(f"   • Demand forecasting integration")
    print(f"   • Dynamic safety stock calculation")
    print(f"   • Economic order quantity optimization")
    print(f"   • Real-time reorder point management")
    print(f"   Contact: info@shambhavithakur.com")
    
    # Export sample data
    output_file = 'inventory_optimization_analysis.csv'
    optimization_report.to_csv(output_file, index=False)
    print(f"\n   Sample Analysis Exported: {output_file}")
    
    return optimization_report

if __name__ == "__main__":
    # Run inventory optimization demonstration
    results = main()
    print(f"\n✅ Inventory optimization analysis completed!")
    print("📊 Framework ready for supply chain intelligence consulting")
    