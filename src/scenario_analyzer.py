from typing import Dict, List, Union, Optional
from market_analyzer import MarketAnalyzer
import pandas as pd
from datetime import datetime
import copy

class PricingScenarioAnalyzer:
    def __init__(self, project_data: Dict, macro_data: Dict):
        """Initialize with base project and macro data"""
        self.base_project_data = copy.deepcopy(project_data)
        self.base_macro_data = copy.deepcopy(macro_data)
        self.base_analyzer = MarketAnalyzer(project_data, macro_data)
        self.base_pricing = self.base_analyzer.analyze_market()
        
    def analyze_scenario(self, scenario: Dict[str, Union[float, str]]) -> Dict:
        """
        Analyze pricing impact of a scenario. Scenarios can include:
        - supply_change: % change in market supply (e.g. 10 for 10% increase)
        - interest_rate_change: basis point change in rates (e.g. -50 for 50bps decrease)
        - employment_change: % change in employment rate (e.g. 2 for 2% increase)
        - competitor_pricing_change: % change in competitor pricing (e.g. -5 for 5% decrease)
        """
        # Create modified data for scenario
        scenario_project_data = copy.deepcopy(self.base_project_data)
        scenario_macro_data = copy.deepcopy(self.base_macro_data)
        
        # Apply scenario changes
        self._apply_scenario_changes(scenario, scenario_project_data, scenario_macro_data)
        
        # Run analysis with modified data
        scenario_analyzer = MarketAnalyzer(scenario_project_data, scenario_macro_data)
        scenario_results = scenario_analyzer.analyze_market()
        
        # Calculate and format pricing impacts
        impact_analysis = self._analyze_pricing_impacts(
            self.base_pricing,
            scenario_results,
            scenario
        )
        
        return impact_analysis
    
    def _apply_scenario_changes(
        self,
        scenario: Dict[str, Union[float, str]],
        project_data: Dict,
        macro_data: Dict
    ) -> None:
        """Apply scenario changes to the data"""
        
        # Handle both old and new macro data structures
        if 'content' in macro_data:
            macro_indicators = macro_data['content']['macro_indicators']
        elif 'macro_indicators' in macro_data:
            macro_indicators = macro_data['macro_indicators']
        else:
            # If neither structure exists, use the data as is
            macro_indicators = macro_data
        
        if 'interest_rate_change' in scenario:
            if 'interest_rates' in macro_indicators and 'current' in macro_indicators['interest_rates']:
                current_rate = macro_indicators['interest_rates']['current']['rates']['5yr_fixed']
                new_rate = current_rate * (1 + scenario['interest_rate_change'] / 100)
                macro_indicators['interest_rates']['current']['rates']['5yr_fixed'] = new_rate
        
        if 'employment_change' in scenario:
            if 'employment_metrics' in macro_indicators and 'current_statistics' in macro_indicators['employment_metrics']:
                current_rate = macro_indicators['employment_metrics']['current_statistics']['employment_rate']
                new_rate = current_rate * (1 + scenario['employment_change'] / 100)
                macro_indicators['employment_metrics']['current_statistics']['employment_rate'] = new_rate
        
        if 'supply_change' in scenario:
            # Update supply metrics based on scenario
            if ('supply_metrics' in macro_indicators and 
                'construction_starts' in macro_indicators['supply_metrics'] and
                'recent_trends' in macro_indicators['supply_metrics']['construction_starts'] and
                '2024' in macro_indicators['supply_metrics']['construction_starts']['recent_trends']):
                current_starts = macro_indicators['supply_metrics']['construction_starts']['recent_trends']['2024']['ytd_starts']
                new_starts = current_starts * (1 + scenario['supply_change'] / 100)
                macro_indicators['supply_metrics']['construction_starts']['recent_trends']['2024']['ytd_starts'] = new_starts
            
            # Also update total pipeline in project data
            if 'market_metrics' in project_data and 'supply_pipeline' in project_data['market_metrics']:
                current_supply = project_data['market_metrics']['supply_pipeline']['total_pipeline']
                project_data['market_metrics']['supply_pipeline']['total_pipeline'] = \
                    int(current_supply * (1 + scenario['supply_change'] / 100))
        
        # Handle competitor pricing change
        if 'competitor_pricing_change' in scenario:
            change = scenario['competitor_pricing_change'] / 100  # Convert to decimal
            if 'active_projects' in project_data and 'projects' in project_data['active_projects']:
                for project in project_data['active_projects']['projects']:
                    if 'pricing' in project:
                        for unit_type in project['pricing']:
                            if 'avg_psf' in project['pricing'][unit_type]:
                                project['pricing'][unit_type]['avg_psf'] *= (1 + change)
    
    def _analyze_pricing_impacts(
        self,
        base_results: Dict,
        scenario_results: Dict,
        scenario: Dict[str, Union[float, str]]
    ) -> Dict:
        """Analyze and format pricing impacts"""
        
        # Get target PSFs directly from market analyzer
        target_psfs = self.base_analyzer._calculate_target_psfs()
        base_pricing = {
            'studios': {'base_psf': target_psfs['studios']},
            'one_bed': {'base_psf': target_psfs['one_bed']},
            'two_bed': {'base_psf': target_psfs['two_bed']},
            'three_bed': {'base_psf': target_psfs['three_bed']}
        }
        
        # Initialize scenario PSFs same as base
        for unit_type in base_pricing:
            base_pricing[unit_type]['scenario_psf'] = base_pricing[unit_type]['base_psf']
        
        # Calculate scenario impacts using our sensitivity data
        if scenario:
            # Apply scenario changes to pricing
            for unit_type in base_pricing:
                base_psf = base_pricing[unit_type]['base_psf']
                scenario_psf = base_psf
                
                # Apply supply impact (varies by unit type)
                if 'supply_change' in scenario:
                    supply_elasticity = {
                        'studios': -0.30,    # Most sensitive (from market analysis)
                        'one_bed': -0.25,    # Very sensitive
                        'two_bed': -0.20,    # Moderately sensitive
                        'three_bed': -0.15   # Least sensitive
                    }
                    supply_impact = supply_elasticity[unit_type] * (scenario['supply_change'] / 100)
                    scenario_psf *= (1 + supply_impact)
                
                # Apply interest rate impact (varies by unit type)
                if 'interest_rate_change' in scenario:
                    # Base sensitivity: -6% price change per 100bps
                    # Unit multipliers from pricing strategy
                    rate_sensitivity = {
                        'studios': -0.072,    # -7.2% per 100bps (1.2x multiplier)
                        'one_bed': -0.060,    # -6.0% per 100bps (base)
                        'two_bed': -0.060,    # -6.0% per 100bps (base)
                        'three_bed': -0.054    # -5.4% per 100bps (0.9x multiplier)
                    }
                    # Convert bps to percentage for calculation (e.g., 25bps = 0.25)
                    rate_change_pct = scenario['interest_rate_change'] / 100
                    rate_impact = rate_sensitivity[unit_type] * rate_change_pct
                    scenario_psf *= (1 + rate_impact)
                
                # Apply employment impact (varies by unit type)
                if 'employment_change' in scenario:
                    # Using 0.72 correlation from market analysis
                    employment_sensitivity = {
                        'studios': 0.008 * 1.1,     # 0.8% per 1% change * 1.1 multiplier
                        'one_bed': 0.008 * 1.1,     # 0.8% per 1% change * 1.1 multiplier
                        'two_bed': 0.008 * 1.0,     # 0.8% per 1% change * standard multiplier
                        'three_bed': 0.008 * 0.9    # 0.8% per 1% change * 0.9 multiplier
                    }
                    emp_impact = employment_sensitivity[unit_type] * scenario['employment_change']
                    scenario_psf *= (1 + emp_impact)
                
                # Apply competitor pricing impact (varies by unit type)
                if 'competitor_pricing_change' in scenario:
                    comp_sensitivity = {
                        'studios': 0.50,     # 50% follow competitor changes
                        'one_bed': 0.45,     # 45% follow
                        'two_bed': 0.40,     # 40% follow
                        'three_bed': 0.35    # 35% follow
                    }
                    comp_impact = comp_sensitivity[unit_type] * (scenario['competitor_pricing_change'] / 100)
                    scenario_psf *= (1 + comp_impact)
                
                base_pricing[unit_type]['scenario_psf'] = scenario_psf
        
        # Calculate impacts
        for unit_type in base_pricing:
            base_pricing[unit_type]['psf_change'] = (
                base_pricing[unit_type]['scenario_psf'] - base_pricing[unit_type]['base_psf']
            )
            base_pricing[unit_type]['percent_change'] = (
                (base_pricing[unit_type]['scenario_psf'] / base_pricing[unit_type]['base_psf'] - 1) * 100
                if base_pricing[unit_type]['base_psf'] > 0 else 0
            )
        
        # Format scenario description
        scenario_description = self._format_scenario_description(scenario)
        
        # Calculate market conditions
        market_conditions = {
            'absorption_impact': self._calculate_absorption_impact(scenario),
            'supply_level': self._calculate_supply_level(scenario),
            'market_sentiment': self._determine_market_sentiment(base_pricing, scenario)
        }
        
        return {
            'scenario': scenario_description,
            'unit_impacts': base_pricing,
            'market_conditions': market_conditions,
            'recommendations': self._generate_recommendations(base_pricing, market_conditions)
        }
    
    def _calculate_absorption_impact(self, scenario: Dict) -> float:
        """Calculate percentage change in absorption rate based on scenario factors"""
        # Base monthly absorption rate from our data
        base_absorption = 5.4  # Current monthly absorption
        
        # Calculate percentage changes to absorption using our sensitivity analysis
        impact = 0
        
        if 'interest_rate_change' in scenario:
            # 8.7% absorption increase per 50bps rate cut (from sensitivity analysis)
            rate_change_pct = scenario['interest_rate_change'] / 100  # Convert bps to percentage points
            # For 50bps (0.5%), impact is 8.7%, so for 100bps it's 17.4%
            impact += -17.4 * rate_change_pct  # Negative because rate increase reduces absorption
        
        if 'supply_change' in scenario:
            # From sensitivity analysis: -5% absorption per 10% supply increase
            impact += -0.5 * scenario['supply_change']
        
        if 'employment_change' in scenario:
            # From sensitivity analysis: 1.2% absorption per 1% employment change
            impact += 1.2 * scenario['employment_change']
        
        if 'competitor_pricing_change' in scenario:
            # When competitors increase prices, our absorption should increase
            # Assume 0.8% absorption increase per 1% competitor price increase
            impact += 0.8 * scenario['competitor_pricing_change']  # Changed from negative to positive
        
        # Return percentage change in monthly absorption rate
        return impact
    
    def _calculate_supply_level(self, scenario: Dict) -> float:
        """Calculate months of supply"""
        base_supply = 14.2  # Current market supply level
        
        if 'supply_change' in scenario:
            supply_change = scenario['supply_change'] / 100
            base_supply *= (1 + supply_change)
        
        return base_supply
    
    def _determine_market_sentiment(self, pricing_impacts: Dict, scenario: Dict) -> str:
        """Determine market sentiment based on scenario impacts"""
        # Calculate average price impact (weighted heavily as it's a key indicator)
        avg_impact = sum(
            impact['percent_change'] 
            for impact in pricing_impacts.values()
        ) / len(pricing_impacts)
        
        # Start with base impact from pricing
        impact_score = avg_impact * 0.5  # Base price impact
        
        # Add supply impact - scaled for typical ranges (-20% to +20%)
        if 'supply_change' in scenario:
            # Scale supply to have similar impact range as rates
            # -20% supply → +10 points (positive)
            # +20% supply → -10 points (negative)
            supply_impact = -(scenario['supply_change'] * 0.5)
            impact_score += supply_impact
            print(f"Supply Impact: {supply_impact}")
            
        # Add absorption impact from market conditions
        absorption_impact = self._calculate_absorption_impact(scenario)
        # Scale absorption to have meaningful impact
        # Typical range is -10% to +10%, so multiply by 0.8 to get -8 to +8 range
        scaled_absorption = absorption_impact * 0.8
        impact_score += scaled_absorption
        print(f"Absorption Impact: {scaled_absorption}")
        
        # Add interest rate impact if present (scaled for basis points)
        if 'interest_rate_change' in scenario:
            # Scale for -200bps to +200bps range
            # -100bps → +5 points
            # +100bps → -5 points
            rate_impact = -scenario['interest_rate_change'] * 0.05
            impact_score += rate_impact
            print(f"Rate Impact: {rate_impact}")
        
        # Add employment impact if present
        if 'employment_change' in scenario:
            # Scale for typical -5% to +5% range
            # +5% employment → +10 points
            # -5% employment → -10 points
            emp_impact = scenario['employment_change'] * 2.0
            impact_score += emp_impact
            print(f"Employment Impact: {emp_impact}")
        
        # Add competitor pricing impact if present
        if 'competitor_pricing_change' in scenario:
            # Scale for typical -10% to +10% range
            # +10% competitor pricing → +8 points (positive for us)
            # -10% competitor pricing → -8 points (negative for us)
            comp_impact = scenario['competitor_pricing_change'] * 0.8
            impact_score += comp_impact
            print(f"Competitor Pricing Impact: {comp_impact}")
        
        print(f"Base Price Impact: {avg_impact * 0.5}")
        print(f"Total Impact Score: {impact_score}")
        
        # Normalize score to be more evenly distributed
        if impact_score > 12:
            return "Strongly Positive"
        elif impact_score > 8:
            return "Moderately Positive"
        elif impact_score > 4:
            return "Slightly Positive"
        elif impact_score > -4:
            return "Neutral"
        elif impact_score > -8:
            return "Slightly Negative"
        elif impact_score > -12:
            return "Moderately Negative"
        else:
            return "Strongly Negative"
    
    def _generate_recommendations(self, pricing_impacts: Dict, market_conditions: Dict) -> List[str]:
        """Generate recommendations based on impacts"""
        recommendations = []
        
        # Price change recommendations
        for unit_type, impact in pricing_impacts.items():
            if abs(impact['percent_change']) >= 2:
                direction = "increase" if impact['percent_change'] > 0 else "decrease"
                recommendations.append(
                    f"Consider {direction} in {unit_type} pricing by "
                    f"{abs(impact['percent_change']):.1f}% to {impact['scenario_psf']:.2f} PSF"
                )
        
        # Market condition recommendations
        if market_conditions['supply_level'] > 18:
            recommendations.append("High supply levels suggest conservative pricing approach")
        elif market_conditions['supply_level'] < 12:
            recommendations.append("Limited supply supports current pricing strategy")
        
        if market_conditions['absorption_impact'] < -1:
            recommendations.append("Consider buyer incentives to maintain absorption rates")
        elif market_conditions['absorption_impact'] > 1:
            recommendations.append("Strong absorption supports price optimization")
        
        return recommendations
    
    def _format_scenario_description(self, scenario: Dict[str, Union[float, str]]) -> str:
        """Format scenario changes into readable description"""
        descriptions = []
        
        if 'supply_change' in scenario:
            descriptions.append(f"Supply {'increase' if scenario['supply_change'] > 0 else 'decrease'} "
                             f"of {abs(scenario['supply_change'])}%")
        
        if 'interest_rate_change' in scenario:
            descriptions.append(f"Interest rate {'increase' if scenario['interest_rate_change'] > 0 else 'decrease'} "
                             f"of {abs(scenario['interest_rate_change'])} basis points")
        
        if 'employment_change' in scenario:
            descriptions.append(f"Employment {'increase' if scenario['employment_change'] > 0 else 'decrease'} "
                             f"of {abs(scenario['employment_change'])}%")
        
        if 'competitor_pricing_change' in scenario:
            descriptions.append(f"Competitor pricing {'increase' if scenario['competitor_pricing_change'] > 0 else 'decrease'} "
                             f"of {abs(scenario['competitor_pricing_change'])}%")
        
        return " with ".join(descriptions)
    
    def generate_report(self, scenario_results: Dict) -> str:
        """Generate a user-friendly report of scenario analysis"""
        report = [
            "# Market Scenario Analysis Report",
            f"\n## Scenario Description",
            scenario_results['scenario'],
            
            "\n## Pricing Impacts by Unit Type",
            "Unit Type | Current PSF | New PSF | Change ($) | Change (%)",
            "----------|-------------|----------|------------|------------"
        ]
        
        for unit_type, impact in scenario_results['unit_impacts'].items():
            report.append(
                f"{unit_type.title()} | ${impact['base_psf']:.2f} | "
                f"${impact['scenario_psf']:.2f} | "
                f"${impact['psf_change']:.2f} | "
                f"{impact['percent_change']:.1f}%"
            )
        
        report.extend([
            "\n## Market Conditions",
            f"- Absorption Impact: {scenario_results['market_conditions']['absorption_impact']:.1f}%",
            f"- Months of Supply: {scenario_results['market_conditions']['supply_level']:.1f}",
            f"- Market Sentiment: {scenario_results['market_conditions']['market_sentiment']}",
            
            "\n## Recommendations"
        ])
        
        for i, rec in enumerate(scenario_results['recommendations'], 1):
            report.append(f"{i}. {rec}")
        
        return "\n".join(report)

def format_currency(value: float) -> str:
    """Format value as currency"""
    return f"${value:,.2f}" 