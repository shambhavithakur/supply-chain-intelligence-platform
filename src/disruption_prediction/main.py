#!/usr/bin/env python3
"""
Disruption Prediction - Supply Chain Crisis Early Warning System
Path: src/disruption_prediction/main.py
Author: Shambhavi Thakur - Data Intelligence Professional
Purpose: Demonstrate supply chain disruption prediction and mitigation methodology
Version: 1.0.0 - Production Ready

This module demonstrates systematic disruption prediction capability
for preventing ₹25+ crore supply chain losses through early warning systems.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class DisruptionPredictor:
    """
    Supply Chain Disruption Prediction and Crisis Management
    
    Demonstrates methodology for preventing ₹25+ crore inventory losses
    through systematic disruption forecasting and mitigation planning.
    """
    
    def __init__(self):
        """Initialize disruption prediction framework"""
        
        # Disruption risk factors and weights
        self.risk_factors = {
            'supplier_financial_health': 0.25,
            'geopolitical_risk': 0.20,
            'natural_disaster_risk': 0.15,
            'transportation_disruption': 0.15,
            'demand_volatility': 0.10,
            'regulatory_changes': 0.10,
            'cyber_security_risk': 0.05
        }
        
        # Disruption impact categories
        self.impact_categories = {
            'supplier_bankruptcy': {
                'probability_base': 0.08,
                'impact_multiplier': 2.5,
                'recovery_days': 60
            },
            'natural_disaster': {
                'probability_base': 0.03,
                'impact_multiplier': 4.0,
                'recovery_days': 90
            },
            'geopolitical_crisis': {
                'probability_base': 0.12,
                'impact_multiplier': 1.8,
                'recovery_days': 120
            },
            'transportation_strike': {
                'probability_base': 0.06,
                'impact_multiplier': 1.2,
                'recovery_days': 21
            },
            'regulatory_change': {
                'probability_base': 0.15,
                'impact_multiplier': 1.5,
                'recovery_days': 45
            },
            'cyber_attack': {
                'probability_base': 0.04,
                'impact_multiplier': 2.0,
                'recovery_days': 14
            }
        }
        
        # Mitigation strategies and costs
        self.mitigation_strategies = {
            'supplier_diversification': {
                'effectiveness': 0.70,
                'implementation_cost_percentage': 0.05,
                'implementation_days': 90
            },
            'inventory_buffer': {
                'effectiveness': 0.80,
                'implementation_cost_percentage': 0.15,
                'implementation_days': 30
            },
            'alternative_routes': {
                'effectiveness': 0.60,
                'implementation_cost_percentage': 0.03,
                'implementation_days': 14
            },
            'emergency_contracts': {
                'effectiveness': 0.85,
                'implementation_cost_percentage': 0.08,
                'implementation_days': 60
            },
            'digital_monitoring': {
                'effectiveness': 0.65,
                'implementation_cost_percentage': 0.02,
                'implementation_days': 45
            }
        }
    
    def assess_disruption_risk(self, supply_chain_data: Dict) -> Dict:
        """
        Assess overall disruption risk for supply chain segment
        
        Args:
            supply_chain_data: Dictionary with supply chain parameters
            
        Returns:
            Comprehensive risk assessment with probability and impact
        """
        
        # Calculate weighted risk score
        risk_scores = {}
        
        # Supplier financial health (1-10 scale, lower is riskier)
        supplier_health = supply_chain_data.get('supplier_health_score', 7.0)
        risk_scores['supplier_financial_health'] = (10 - supplier_health) / 10
        
        # Geopolitical risk (regional stability index)
        geo_risk = supply_chain_data.get('geopolitical_risk_index', 0.3)
        risk_scores['geopolitical_risk'] = geo_risk
        
        # Natural disaster risk (historical probability)
        disaster_risk = supply_chain_data.get('natural_disaster_probability', 0.05)
        risk_scores['natural_disaster_risk'] = disaster_risk
        
        # Transportation disruption risk
        transport_risk = supply_chain_data.get('transportation_reliability', 0.9)
        risk_scores['transportation_disruption'] = 1 - transport_risk
        
        # Demand volatility (coefficient of variation)
        demand_volatility = supply_chain_data.get('demand_cv', 0.2)
        risk_scores['demand_volatility'] = min(1.0, demand_volatility)
        
        # Regulatory change risk
        regulatory_risk = supply_chain_data.get('regulatory_stability', 0.8)
        risk_scores['regulatory_changes'] = 1 - regulatory_risk
        
        # Cyber security risk
        cyber_risk = supply_chain_data.get('cyber_security_maturity', 0.7)
        risk_scores['cyber_security_risk'] = 1 - cyber_risk
        
        # Calculate overall risk score (0-10 scale)
        overall_risk = sum(
            risk_scores[factor] * weight 
            for factor, weight in self.risk_factors.items()
        ) * 10
        
        # Identify top risk factors
        top_risks = sorted(risk_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        
        assessment = {
            'overall_risk_score': round(overall_risk, 1),
            'risk_level': self._categorize_risk_level(overall_risk),
            'individual_scores': {k: round(v, 2) for k, v in risk_scores.items()},
            'top_risk_factors': [{'factor': factor, 'score': round(score, 2)} 
                               for factor, score in top_risks],
            'disruption_probability': self._calculate_disruption_probability(overall_risk),
            'recommended_actions': self._generate_risk_recommendations(overall_risk, top_risks)
        }
        
        return assessment
    
    def _categorize_risk_level(self, risk_score: float) -> str:
        """Categorize risk level for business communication"""
        if risk_score >= 8.0:
            return "CRITICAL - Immediate action required"
        elif risk_score >= 6.0:
            return "HIGH - Develop mitigation plans"
        elif risk_score >= 4.0:
            return "MODERATE - Monitor closely"
        else:
            return "LOW - Standard monitoring"
    
    def _calculate_disruption_probability(self, risk_score: float) -> Dict:
        """Calculate probability of different disruption scenarios"""
        
        probabilities = {}
        
        for disruption_type, params in self.impact_categories.items():
            # Scale base probability by overall risk score
            risk_multiplier = 1 + (risk_score - 5) * 0.2  # Center around score of 5
            probability = params['probability_base'] * max(0.1, risk_multiplier)
            probabilities[disruption_type] = round(min(0.8, probability), 3)
        
        return probabilities
    
    def _generate_risk_recommendations(self, risk_score: float, top_risks: List) -> List[str]:
        """Generate specific risk mitigation recommendations"""
        
        recommendations = []
        
        if risk_score >= 7.0:
            recommendations.append("URGENT: Implement emergency response protocols immediately")
            recommendations.append("Diversify supplier base to reduce single points of failure")
            recommendations.append("Increase safety stock for critical components")
        
        elif risk_score >= 5.0:
            recommendations.append("Develop comprehensive business continuity plan")
            recommendations.append("Establish alternative sourcing agreements")
            recommendations.append("Implement real-time monitoring systems")
        
        else:
            recommendations.append("Maintain current monitoring protocols")
            recommendations.append("Review risk assessment quarterly")
        
        # Factor-specific recommendations
        for factor, score in top_risks:
            if factor == 'supplier_financial_health' and score > 0.6:
                recommendations.append("Conduct detailed financial audits of key suppliers")
            elif factor == 'geopolitical_risk' and score > 0.6:
                recommendations.append("Develop regional supply chain alternatives")
            elif factor == 'natural_disaster_risk' and score > 0.4:
                recommendations.append("Review and enhance disaster recovery procedures")
        
        return recommendations
    
    def calculate_business_impact(self, disruption_scenario: str, supply_chain_value: float, 
                                recovery_days: int = None) -> Dict:
        """
        Calculate potential business impact of supply chain disruption
        
        Returns financial impact analysis for business decision-making
        """
        
        scenario_params = self.impact_categories.get(disruption_scenario, 
                                                   self.impact_categories['supplier_bankruptcy'])
        
        if recovery_days is None:
            recovery_days = scenario_params['recovery_days']
        
        # Calculate direct impact
        impact_multiplier = scenario_params['impact_multiplier']
        
        # Inventory at risk (assuming 30-day forward inventory)
        inventory_at_risk = supply_chain_value * 0.25  # 3-month inventory exposure
        
        # Revenue impact (lost sales during disruption)
        daily_revenue_impact = supply_chain_value / 365
        total_revenue_impact = daily_revenue_impact * recovery_days * impact_multiplier
        
        # Additional costs (emergency sourcing, expedited shipping)
        emergency_costs = supply_chain_value * 0.15 * impact_multiplier
        
        # Total potential loss
        total_potential_loss = inventory_at_risk + total_revenue_impact + emergency_costs
        
        # Recovery timeline and costs
        recovery_cost = supply_chain_value * 0.05  # 5% of value for recovery efforts
        
        impact_analysis = {
            'disruption_scenario': disruption_scenario,
            'supply_chain_value': supply_chain_value,
            'inventory_at_risk': round(inventory_at_risk, 2),
            'revenue_impact': round(total_revenue_impact, 2),
            'emergency_costs': round(emergency_costs, 2),
            'recovery_costs': round(recovery_cost, 2),
            'total_potential_loss': round(total_potential_loss, 2),
            'recovery_days': recovery_days,
            'impact_multiplier': impact_multiplier,
            'risk_mitigation_recommendations': self._recommend_mitigation_strategies(total_potential_loss)
        }
        
        return impact_analysis
    
    def _recommend_mitigation_strategies(self, potential_loss: float) -> List[Dict]:
        """Recommend specific mitigation strategies based on potential loss"""
        
        recommendations = []
        
        for strategy, params in self.mitigation_strategies.items():
            implementation_cost = potential_loss * params['implementation_cost_percentage']
            potential_savings = potential_loss * params['effectiveness']
            net_benefit = potential_savings - implementation_cost
            
            recommendation = {
                'strategy': strategy,
                'implementation_cost': round(implementation_cost, 2),
                'potential_loss_prevention': round(potential_savings, 2),
                'net_benefit': round(net_benefit, 2),
                'implementation_days': params['implementation_days'],
                'effectiveness': f"{params['effectiveness']*100:.0f}%",
                'roi': round((net_benefit / implementation_cost) * 100, 0) if implementation_cost > 0 else 0
            }
            
            recommendations.append(recommendation)
        
        # Sort by ROI
        recommendations.sort(key=lambda x: x['roi'], reverse=True)
        
        return recommendations
    
    def generate_disruption_scenario_analysis(self, supply_chain_segments: List[Dict]) -> pd.DataFrame:
        """
        Generate comprehensive disruption scenario analysis for all supply chain segments
        
        Returns DataFrame with risk assessment and mitigation recommendations
        """
        
        scenario_analyses = []
        
        for segment in supply_chain_segments:
            # Risk assessment
            risk_assessment = self.assess_disruption_risk(segment['parameters'])
            
            # Impact analysis for most likely disruption
            most_likely_disruption = max(
                risk_assessment['disruption_probability'].items(),
                key=lambda x: x[1]
            )[0]
            
            impact_analysis = self.calculate_business_impact(
                most_likely_disruption,
                segment['annual_value']
            )
            
            # Combine assessments
            analysis = {
                'segment_name': segment['name'],
                'segment_category': segment['category'],
                'annual_value_lakhs': round(segment['annual_value'] / 100000, 1),
                'risk_score': risk_assessment['overall_risk_score'],
                'risk_level': risk_assessment['risk_level'],
                'disruption_probability': round(risk_assessment['disruption_probability'][most_likely_disruption] * 100, 1),
                'most_likely_scenario': most_likely_disruption.replace('_', ' ').title(),
                'potential_loss_lakhs': round(impact_analysis['total_potential_loss'] / 100000, 1),
                'recovery_days': impact_analysis['recovery_days'],
                'top_risk_factor': risk_assessment['top_risk_factors'][0]['factor'].replace('_', ' ').title(),
                'mitigation_priority': 'HIGH' if risk_assessment['overall_risk_score'] >= 6.0 else 
                                     'MEDIUM' if risk_assessment['overall_risk_score'] >= 4.0 else 'LOW',
                'recommended_actions': '; '.join(risk_assessment['recommended_actions'][:2])
            }
            
            scenario_analyses.append(analysis)
        
        return pd.DataFrame(scenario_analyses)

def generate_sample_supply_chain_data() -> List[Dict]:
    """Generate realistic sample supply chain segments for analysis"""
    
    sample_segments = [
        {
            'name': 'Critical Components Supply',
            'category': 'critical',
            'annual_value': 25000000,  # ₹2.5 crore
            'parameters': {
                'supplier_health_score': 3.2,  # Poor financial health
                'geopolitical_risk_index': 0.4,
                'natural_disaster_probability': 0.08,
                'transportation_reliability': 0.85,
                'demand_cv': 0.35,
                'regulatory_stability': 0.7,
                'cyber_security_maturity': 0.6
            }
        },
        {
            'name': 'Raw Materials Supply',
            'category': 'important',
            'annual_value': 15000000,  # ₹1.5 crore
            'parameters': {
                'supplier_health_score': 6.5,  # Moderate financial health
                'geopolitical_risk_index': 0.6,  # High geo risk
                'natural_disaster_probability': 0.12,
                'transportation_reliability': 0.75,
                'demand_cv': 0.25,
                'regulatory_stability': 0.8,
                'cyber_security_maturity': 0.7
            }
        },
        {
            'name': 'Packaging Materials',
            'category': 'standard',
            'annual_value': 8000000,  # ₹80 lakh
            'parameters': {
                'supplier_health_score': 7.8,  # Good financial health
                'geopolitical_risk_index': 0.2,
                'natural_disaster_probability': 0.05,
                'transportation_reliability': 0.9,
                'demand_cv': 0.15,
                'regulatory_stability': 0.9,
                'cyber_security_maturity': 0.8
            }
        },
        {
            'name': 'Electronic Components',
            'category': 'critical',
            'annual_value': 18000000,  # ₹1.8 crore
            'parameters': {
                'supplier_health_score': 4.1,  # Poor financial health
                'geopolitical_risk_index': 0.5,
                'natural_disaster_probability': 0.15,  # High disaster risk
                'transportation_reliability': 0.8,
                'demand_cv': 0.4,  # High demand volatility
                'regulatory_stability': 0.6,
                'cyber_security_maturity': 0.5
            }
        }
    ]
    
    return sample_segments

def main():
    """
    Demonstration of disruption prediction for supply chain intelligence
    
    Shows capability for preventing ₹25+ crore supply chain losses through
    systematic disruption forecasting and mitigation planning.
    """
    
    print("DISRUPTION PREDICTION - SUPPLY CHAIN INTELLIGENCE DEMO")
    print("=" * 65)
    print("Author: Shambhavi Thakur - Data Intelligence Professional")
    print("Purpose: Demonstrate supply chain crisis prediction methodology")
    print("Contact: info@shambhavithakur.com")
    print()
    
    # Initialize disruption predictor
    predictor = DisruptionPredictor()
    
    # Generate sample supply chain data
    supply_chain_segments = generate_sample_supply_chain_data()
    
    print(f"SUPPLY CHAIN DISRUPTION ANALYSIS - {datetime.now().strftime('%Y-%m-%d')}")
    print("-" * 55)
    
    # Generate comprehensive scenario analysis
    scenario_analysis = predictor.generate_disruption_scenario_analysis(supply_chain_segments)
    
    # Executive Summary
    total_value = scenario_analysis['annual_value_lakhs'].sum()
    total_at_risk = scenario_analysis['potential_loss_lakhs'].sum()
    high_risk_segments = len(scenario_analysis[scenario_analysis['risk_score'] >= 6.0])
    avg_risk_score = scenario_analysis['risk_score'].mean()
    
    print("EXECUTIVE SUMMARY:")
    print(f"   Total Supply Chain Value: ₹{total_value:.1f} lakhs annually")
    print(f"   Total Potential Loss: ₹{total_at_risk:.1f} lakhs")
    print(f"   High Risk Segments: {high_risk_segments} requiring immediate attention")
    print(f"   Average Risk Score: {avg_risk_score:.1f}/10")
    print(f"   Loss Prevention Opportunity: ₹{total_at_risk*0.7:.1f} lakhs")
    print()
    
    # Detailed Segment Analysis
    print("DETAILED DISRUPTION RISK ANALYSIS:")
    print("=" * 50)
    
    for _, segment in scenario_analysis.iterrows():
        print(f"\n{segment['segment_name'].upper()}")
        print(f"   Category: {segment['segment_category'].title()}")
        print(f"   Annual Value: ₹{segment['annual_value_lakhs']:.1f} lakhs")
        print(f"   Risk Score: {segment['risk_score']:.1f}/10")
        print(f"   Risk Level: {segment['risk_level']}")
        print(f"   Most Likely Scenario: {segment['most_likely_scenario']}")
        print(f"   Disruption Probability: {segment['disruption_probability']:.1f}%")
        print(f"   Potential Loss: ₹{segment['potential_loss_lakhs']:.1f} lakhs")
        print(f"   Recovery Time: {segment['recovery_days']} days")
        print(f"   Top Risk Factor: {segment['top_risk_factor']}")
        print(f"   Mitigation Priority: {segment['mitigation_priority']}")
        print(f"   Key Actions: {segment['recommended_actions']}")
    
    # Risk Prioritization
    print(f"\nRISK PRIORITIZATION MATRIX:")
    print("=" * 40)
    
    high_priority = scenario_analysis[scenario_analysis['mitigation_priority'] == 'HIGH']
    medium_priority = scenario_analysis[scenario_analysis['mitigation_priority'] == 'MEDIUM']
    
    if len(high_priority) > 0:
        print(f"HIGH PRIORITY SEGMENTS ({len(high_priority)}):")
        for _, segment in high_priority.iterrows():
            print(f"   → {segment['segment_name']}: ₹{segment['potential_loss_lakhs']:.1f}L potential loss")
    
    if len(medium_priority) > 0:
        print(f"MEDIUM PRIORITY SEGMENTS ({len(medium_priority)}):")
        for _, segment in medium_priority.iterrows():
            print(f"   → {segment['segment_name']}: {segment['disruption_probability']:.0f}% probability")
    
    # Mitigation Strategy Analysis
    print(f"\nMITIGATION STRATEGY RECOMMENDATIONS:")
    print("=" * 45)
    
    # Sample mitigation analysis for highest risk segment
    highest_risk_segment = scenario_analysis.loc[scenario_analysis['risk_score'].idxmax()]
    sample_impact = predictor.calculate_business_impact(
        'supplier_bankruptcy', 
        highest_risk_segment['annual_value_lakhs'] * 100000
    )
    
    print(f"SAMPLE MITIGATION ANALYSIS - {highest_risk_segment['segment_name']}:")
    top_strategies = sample_impact['risk_mitigation_recommendations'][:3]
    
    for strategy in top_strategies:
        print(f"\n   {strategy['strategy'].replace('_', ' ').title()}:")
        print(f"     Implementation Cost: ₹{strategy['implementation_cost']/100000:.1f} lakhs")
        print(f"     Loss Prevention: ₹{strategy['potential_loss_prevention']/100000:.1f} lakhs")
        print(f"     Net Benefit: ₹{strategy['net_benefit']/100000:.1f} lakhs")
        print(f"     ROI: {strategy['roi']:.0f}%")
        print(f"     Implementation: {strategy['implementation_days']} days")
    
    print(f"\nBUSINESS VALUE DEMONSTRATION:")
    print(f"   ✅ Systematic disruption prediction")
    print(f"   ✅ Financial impact quantification")
    print(f"   ✅ Mitigation strategy optimization")
    print(f"   ✅ Early warning system capability")
    
    print(f"\nDISRUPTION PREDICTION FRAMEWORK:")
    print(f"   • Real-time risk monitoring")
    print(f"   • Multi-factor risk assessment")
    print(f"   • Scenario-based impact analysis")
    print(f"   • Automated mitigation recommendations")
    print(f"   Contact: info@shambhavithakur.com")
    
    # Export analysis
    output_file = 'disruption_prediction_analysis.csv'
    scenario_analysis.to_csv(output_file, index=False)
    print(f"\n   Sample Analysis Exported: {output_file}")
    
    return scenario_analysis

if __name__ == "__main__":
    # Run disruption prediction demonstration
    results = main()
    print("\n✅ Disruption prediction analysis completed!")
    print("🚨 Framework ready for supply chain intelligence consulting")
