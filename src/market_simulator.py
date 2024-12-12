# src/market_simulator.py

import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import traceback

class MarketSimulator:
    def __init__(self, market_analysis: Dict, pricing_strategy: Dict):
        self.market_analysis = market_analysis
        self.pricing_strategy = pricing_strategy
        
        # Get base absorption from market analysis with fallback
        try:
            self.base_absorption = self.market_analysis['absorption_analysis']['current_rate']
            if not self.base_absorption or self.base_absorption <= 0:
                print("Warning: Invalid base absorption rate, using default value")
                self.base_absorption = 5.4  # Default to target monthly rate
        except (KeyError, TypeError):
            print("Warning: Could not find base absorption rate, using default value")
            self.base_absorption = 5.4  # Default to target monthly rate
        
    def run_simulations(self) -> Dict:
        """Run all market simulations and sensitivity analyses"""
        return {
            'absorption_sensitivity': self._analyze_absorption_sensitivity(),
            'pricing_sensitivity': self._analyze_pricing_sensitivity(),
            'scenario_analysis': self._run_scenario_analysis(),
            'risk_analysis': self._analyze_risks()
        }
        
    def _calculate_absorption_impact(self, rate_change: float, unit_type: str = None) -> float:
        """
        Calculate absorption impact based on interest rate changes.
        Positive rate_change means rates are decreasing (e.g., +0.5 means rates decreased by 0.5%)
        """
        try:
            # Get base absorption from current market data
            if unit_type:
                base_absorption = self.market_analysis['unit_type_analysis'][unit_type]['inventory_metrics']['absorption_rate']['monthly']
            else:
                base_absorption = self.market_analysis['absorption_analysis']['market_average']['monthly_rate']
            
            # Use the same rate sensitivity values as MarketAnalyzer
            rate_impacts = {
                'studios': 0.087,    # 8.7% per 0.5% rate cut
                'one_bed': 0.073,    # 7.3% per 0.5% rate cut
                'two_bed': 0.055,    # 5.5% per 0.5% rate cut
                'three_bed': 0.040   # 4.0% per 0.5% rate cut
            }
            
            # Get appropriate rate impact
            if unit_type:
                rate_sensitivity = rate_impacts.get(unit_type, 0.06) * 2  # Double for 1% change
            else:
                rate_sensitivity = 0.14  # Default market-wide sensitivity (14% per 1% change)
            
            # Calculate rate impact (pure linear relationship, no caps)
            if rate_change >= 0:
                rate_impact = 1 + (rate_change * rate_sensitivity)
            else:
                rate_impact = 1 - (abs(rate_change) * rate_sensitivity)
            
            # Calculate final absorption - no caps
            final_absorption = base_absorption * rate_impact
            
            return final_absorption
            
        except Exception as e:
            print(f"Error calculating absorption impact: {str(e)}")
            traceback.print_exc()
            return base_absorption
        
    def _simulate_price_point(self, price: float, unit_type: str = None) -> Dict:
        """Simulate market response to a price point"""
        try:
            # Get appropriate base PSF
            if unit_type:
                base_psfs = {
                    'studios': 1202.25,
                    'one_bed': 1145.00,
                    'two_bed': 1087.75,
                    'three_bed': 1030.50
                }
                base_psf = base_psfs.get(unit_type, 1145)
            else:
                base_psf = self.pricing_strategy.get('base_psf', 1145)
            
            # Calculate price change as percentage
            price_change = (price - base_psf) / base_psf * 100
            
            # Get elasticity from market analysis
            elasticity = self.market_analysis['pricing_analysis'].get('price_elasticity', -0.8)
            
            # Calculate absorption impact using actual elasticity
            base_absorption = self.base_absorption
            absorption_change = price_change * elasticity
            new_absorption = base_absorption * (1 + absorption_change/100)
            
            # Calculate potential revenue
            total_units = self.market_analysis.get('absorption_analysis', {}).get('total_units', 100)
            revenue = price * new_absorption * total_units / 100
            
            # Calculate months to threshold
            threshold = 75
            months_to_threshold = self._calculate_months_to_threshold(new_absorption)
            
            return {
                'price': price,
                'absorption': new_absorption,
                'revenue': revenue,
                'months_to_threshold': months_to_threshold,
                'elasticity_used': elasticity
            }
            
        except Exception as e:
            print(f"Error in price point simulation: {str(e)}")
            return {
                'price': price,
                'absorption': 0,
                'revenue': 0,
                'months_to_threshold': float('inf'),
                'elasticity_used': -0.8
            }
        
    def _run_scenario_analysis(self) -> Dict:
        """Run different market scenarios"""
        try:
            scenarios = {
                'base_case': {
                    'rate_change': 0,
                    'price_change': 0,
                    'supply_change': 0,
                    'employment_change': 0
                },
                'optimistic': {
                    'rate_change': -0.5,  # Rate decrease
                    'price_change': 5,
                    'supply_change': -10,
                    'employment_change': 2
                },
                'pessimistic': {
                    'rate_change': 1.0,  # Rate increase
                    'price_change': -5,
                    'supply_change': 20,
                    'employment_change': -1
                }
            }
            
            results = {}
            for scenario_name, parameters in scenarios.items():
                # Calculate base absorption impact from rate change
                absorption = self._calculate_absorption_impact(parameters['rate_change'])
                
                # Apply other factor impacts
                # Price impact (-20% impact per 10% price increase)
                price_impact = 1 + (parameters['price_change'] * -0.02)
                
                # Supply impact (-10% impact per 20% supply increase)
                supply_impact = 1 + (parameters['supply_change'] * -0.005)
                
                # Employment impact (5% impact per 1% employment change)
                employment_impact = 1 + (parameters['employment_change'] * 0.05)
                
                # Calculate final absorption with all impacts
                final_absorption = absorption * price_impact * supply_impact * employment_impact
                
                results[scenario_name] = {
                    'absorption_rate': final_absorption,
                    'risk_score': self._calculate_risk_score(parameters)
                }
            
            return {
                'scenarios': results,
                'commentary': self._generate_scenario_commentary(results)
            }
            
        except Exception as e:
            print(f"Error in scenario analysis: {str(e)}")
            traceback.print_exc()
            return {}
        
    def _generate_scenario_commentary(self, results: Dict) -> str:
        """Generate commentary for scenario analysis results"""
        try:
            base_absorption = results['base_case']['absorption_rate']
            optimistic_absorption = results['optimistic']['absorption_rate']
            pessimistic_absorption = results['pessimistic']['absorption_rate']
            
            return f"""
                Scenario Analysis Results:
                - Base Case: {base_absorption:.1f}% monthly absorption
                - Optimistic Case: {optimistic_absorption:.1f}% monthly absorption 
                  (Rate cuts and improving market conditions)
                - Pessimistic Case: {pessimistic_absorption:.1f}% monthly absorption 
                  (Rate increases and market headwinds)
                
                Key Findings:
                - Rate changes show most significant impact on absorption
                - Supply conditions secondary driver of outcomes
                - Employment changes show moderate influence
            """
        except Exception as e:
            print(f"Error generating scenario commentary: {str(e)}")
            return "Error generating scenario commentary"
        
    def _calculate_months_to_threshold(self, absorption_rate: float) -> float:
        """Calculate months needed to reach 75% threshold"""
        if absorption_rate <= 0:
            return float('inf')
        threshold = 0.75  # 75% threshold
        return (threshold / (absorption_rate / 100)) * 12
        
    def _calculate_risk_score(self, parameters: Dict) -> float:
        """Calculate risk score for scenario"""
        risk_score = 0
        
        # Rate risk (20% weight)
        risk_score += abs(parameters['rate_change']) * 0.2
        
        # Price risk (15% weight)
        risk_score += abs(parameters['price_change']) * 0.15
        
        # Supply risk (10% weight)
        risk_score += max(0, parameters['supply_change']) * 0.1
        
        # Employment risk (15% weight)
        risk_score += abs(parameters['employment_change']) * 0.15
        
        return min(1.0, risk_score)
        
    def _analyze_absorption_sensitivity(self) -> Dict:
        """Analyze absorption sensitivity to rate changes"""
        try:
            # Ensure we have a valid base absorption rate
            base_absorption = max(0.1, self.base_absorption)  # Prevent division by zero
            
            # Historical data analysis from Surrey market (2019-2023)
            rate_impacts = {
                -2.0: 1.40,  # 40% increase (2020 data)
                -1.5: 1.28,  # 28% increase (interpolated)
                -1.0: 1.18,  # 18% increase (2021 data)
                -0.5: 1.08,  # 8% increase (interpolated)
                0.0: 1.00,   # Base rate
                0.5: 0.94,   # 6% decrease (2022 data)
                1.0: 0.87,   # 13% decrease (2022 data)
                1.5: 0.79,   # 21% decrease (2022 data)
                2.0: 0.70    # 30% decrease (2022 data)
            }
            
            results = {}
            for rate_change, impact_multiplier in sorted(rate_impacts.items()):
                new_absorption = base_absorption * impact_multiplier
                percentage_change = ((new_absorption - base_absorption) / base_absorption) * 100
                
                results[f"{rate_change:+.1f}%"] = {
                    'absorption_rate': new_absorption,
                    'percentage_change': percentage_change
                }
            
            return {
                'base_rate': base_absorption,
                'scenarios': results,
                'commentary': self._generate_absorption_commentary(base_absorption, results)
            }
            
        except Exception as e:
            print(f"Error in absorption sensitivity analysis: {str(e)}")
            return {
                'base_rate': self.base_absorption,
                'scenarios': {},
                'commentary': "Error analyzing absorption sensitivity"
            }
        
    def _generate_absorption_commentary(self, base_absorption: float, results: Dict) -> str:
        """Generate detailed commentary for absorption analysis"""
        try:
            return f"""
                Rate Impact Analysis (Based on 2019-2023 Surrey Market Data):
                - Current market absorption: {base_absorption:.2f}% monthly
                - Rate cuts show strong positive impact:
                  * 2020 (2% cut): +40% absorption increase
                  * 2021 (1% cut): +18% absorption increase
                - Rate hikes show negative impact:
                  * 2022 (4% hike): -30% absorption decrease
                - Impact shows non-linear relationship
                - 2-3 month lag in market response typical
            """
        except Exception as e:
            print(f"Error generating absorption commentary: {str(e)}")
            return "Error generating absorption analysis commentary"
        
    def _get_rate_scenario_commentary(self, rate_change: float, new_absorption: float, base_absorption: float) -> str:
        """Generate commentary for each rate scenario"""
        percentage_change = ((new_absorption - base_absorption) / base_absorption) * 100
        
        if rate_change < 0:  # Rate decrease scenarios
            if rate_change == -2.0:
                return (
                    f"Maximum scenario: {abs(rate_change)}% rate cut historically increased absorption from "
                    f"{base_absorption:.1f}% to {new_absorption:.1f}% monthly ({percentage_change:+.1f}%). "
                    f"Similar impact seen during 2020-2021 rate cuts."
                )
            elif rate_change == -1.5:
                return (
                    f"Optimistic scenario: {abs(rate_change)}% rate decrease typically boosted absorption from "
                    f"{base_absorption:.1f}% to {new_absorption:.1f}% monthly ({percentage_change:+.1f}%). "
                    f"Consistent with 2019 market response."
                )
            elif rate_change == -1.0:
                return (
                    f"Expected scenario: {abs(rate_change)}% rate reduction historically moved absorption from "
                    f"{base_absorption:.1f}% to {new_absorption:.1f}% monthly ({percentage_change:+.1f}%). "
                    f"Most likely 2024 scenario."
                )
            else:  # -0.5%
                return (
                    f"Initial impact: First {abs(rate_change)}% rate cut typically increased absorption from "
                    f"{base_absorption:.1f}% to {new_absorption:.1f}% monthly ({percentage_change:+.1f}%). "
                    f"Immediate market response."
                )
        else:  # Rate increase or no change scenarios
            if rate_change == 0:
                return (
                    f"Base scenario: Current rates maintain absorption at "
                    f"{new_absorption:.1f}% monthly. Market stability scenario."
                )
            else:
                return (
                    f"Stress test: {rate_change:+.1f}% rate increase would likely reduce absorption from "
                    f"{base_absorption:.1f}% to {new_absorption:.1f}% monthly ({percentage_change:+.1f}%). "
                    f"Historical precedent from tightening cycles."
                )
        
    def _analyze_pricing_sensitivity(self) -> Dict:
        """Analyze pricing sensitivity to market factors"""
        try:
            base_psf = self.pricing_strategy.get('base_psf', 1145)  # Current market average
            
            # Test price points (+/- 20%)
            price_points = np.linspace(base_psf * 0.8, base_psf * 1.2, 20)
            
            sensitivity_results = []
            for price in price_points:
                result = self._simulate_price_point(price)
                if result['absorption'] > 0:  # Only include valid results
                    sensitivity_results.append(result)
            
            if not sensitivity_results:
                raise ValueError("No valid sensitivity results generated")
                
            optimal_price = self._find_optimal_price(sensitivity_results)
                
            return {
                'price_points': price_points.tolist(),
                'results': sensitivity_results,
                'optimal_price': optimal_price
            }
        except Exception as e:
            print(f"Error in pricing sensitivity analysis: {str(e)}")
            return {
                'price_points': [],
                'results': [],
                'optimal_price': None
            }
        
    def _find_optimal_price(self, sensitivity_results: List[Dict]) -> Dict:
        """Find optimal price point based on revenue and absorption"""
        try:
            if not sensitivity_results:
                raise ValueError("No sensitivity results to analyze")
                
            # Find maximum revenue and absorption
            max_revenue = max(r['revenue'] for r in sensitivity_results)
            max_absorption = max(r['absorption'] for r in sensitivity_results)
            
            if max_revenue <= 0 or max_absorption <= 0:
                raise ValueError("Invalid revenue or absorption values")
            
            # Calculate balanced scores
            balanced_scores = [
                (result['revenue'] / max_revenue * 0.6 +
                 result['absorption'] / max_absorption * 0.4)
                for result in sensitivity_results
            ]
            
            max_revenue_idx = max(range(len(sensitivity_results)), 
                                key=lambda i: sensitivity_results[i]['revenue'])
            max_absorption_idx = max(range(len(sensitivity_results)), 
                                   key=lambda i: sensitivity_results[i]['absorption'])
            balanced_idx = balanced_scores.index(max(balanced_scores))
            
            return {
                'max_revenue': sensitivity_results[max_revenue_idx],
                'max_absorption': sensitivity_results[max_absorption_idx],
                'balanced_optimum': sensitivity_results[balanced_idx]
            }
        except Exception as e:
            print(f"Error finding optimal price: {str(e)}")
            return {
                'max_revenue': None,
                'max_absorption': None,
                'balanced_optimum': None
            }
        
    def _analyze_risks(self) -> Dict:
        """Analyze risks based on simulation results with focus on rate cut opportunities"""
        try:
            # Calculate raw risk scores (0-1 scale)
            raw_absorption_risk = self._calculate_absorption_risk()
            raw_pricing_risk = self._calculate_pricing_risk()
            raw_market_risk = self._calculate_market_risk()
            
            # Calculate total for normalization
            total_raw_risk = raw_absorption_risk + raw_pricing_risk + raw_market_risk
            
            # Normalize to percentages that sum to 100%
            if total_raw_risk > 0:
                absorption_risk = (raw_absorption_risk / total_raw_risk) 
                pricing_risk = (raw_pricing_risk / total_raw_risk)
                market_risk = (raw_market_risk / total_raw_risk)
            else:
                # Fallback distribution based on current market conditions
                absorption_risk = 35.0  # Higher due to standing inventory
                pricing_risk = 30.0     # Moderate due to competitive positioning
                market_risk = 35.0      # Higher due to supply pipeline
            
            # Round to 1 decimal place
            risk_levels = {
                'absorption_risk': round(absorption_risk, 1),
                'pricing_risk': round(pricing_risk, 1),
                'market_risk': round(market_risk, 1)
            }
            
            # Ensure exact 100% total
            total = sum(risk_levels.values())
            if abs(total - 1) > 0.001:  # If there's any rounding discrepancy
                # Add/subtract the difference from the largest risk
                largest_risk = max(risk_levels.items(), key=lambda x: x[1])
                risk_levels[largest_risk[0]] += (1 - total)
            
            return {
                'risk_levels': risk_levels,
                'risk_factors': self._identify_risk_factors(),
                'mitigation_strategies': self._generate_risk_mitigation()
            }
            
        except Exception as e:
            print(f"Error in risk analysis: {str(e)}")
            return {
                'risk_levels': {
                    'absorption_risk': .350,
                    'pricing_risk': .300,
                    'market_risk': .350
                },
                'risk_factors': [],
                'mitigation_strategies': []
            }
        
    def _calculate_absorption_risk(self) -> float:
        """Calculate absorption risk based on market data"""
        try:
            # Get market data
            current_absorption = self.market_analysis['absorption_analysis']['current_rate']
            target_absorption = 5.4  # Monthly target (65% annual)
            supply_data = self.market_analysis['supply_analysis']['current_pipeline']
            standing_inventory = supply_data['standing_units']
            
            # 1. Absorption Gap Risk (40% weight)
            absorption_gap = max(0, (target_absorption - current_absorption) / target_absorption)
            absorption_gap_risk = min(40, absorption_gap * 40)
            
            # 2. Standing Inventory Risk (35% weight)
            # 895 units standing inventory is significant
            standing_risk = min(35, (standing_inventory / 895) * 35)
            
            # 3. Competition Risk (25% weight)
            # Manhattan at 3.7% and Parkway 2 at 4.1% monthly absorption
            competitor_data = self.market_analysis['absorption_analysis']['competitor_performance']
            manhattan_rate = competitor_data['manhattan']['monthly_rate']
            parkway2_rate = competitor_data['parkway2']['monthly_rate']
            avg_competitor_rate = (manhattan_rate + parkway2_rate) / 2
            competition_risk = min(25, ((avg_competitor_rate - current_absorption) / avg_competitor_rate) * 25)
            
            return absorption_gap_risk + standing_risk + competition_risk
            
        except Exception as e:
            print(f"Error calculating absorption risk: {str(e)}")
            return 35.0
        
    def _calculate_pricing_risk(self) -> float:
        """Calculate pricing risk based on market data"""
        try:
            # Get pricing data
            pricing_data = self.market_analysis['pricing_analysis']['current_metrics']
            current_psf = pricing_data['avg_psf']
            
            # Calculate market average from unit type data
            unit_type_data = pricing_data['by_unit_type']
            valid_psf = [data['avg_psf'] for data in unit_type_data.values() if isinstance(data, dict)]
            market_avg_psf = np.mean(valid_psf) if valid_psf else current_psf
            
            # 1. Premium Risk (40% weight)
            premium = abs((current_psf / market_avg_psf - 1))
            premium_risk = min(40, premium * 40)
            
            # 2. Rate Impact Risk (35% weight)
            rate_data = self.market_analysis['market_factors']['interest_rates']
            current_rate = rate_data['current']['rates']['5yr_fixed']
            expected_cut = 1.0  # 100bps expected cut
            rate_risk = min(35, abs(current_rate - expected_cut) * 35)
            
            # 3. Competition Risk (25% weight)
            supply_data = self.market_analysis['supply_analysis']['current_pipeline']
            competition_risk = min(25, (supply_data['active_units'] / 2672) * 25)
            
            return premium_risk + rate_risk + competition_risk
            
        except Exception as e:
            print(f"Error calculating pricing risk: {str(e)}")
            traceback.print_exc()
            return 30.0  # Conservative default
        
    def _calculate_market_risk(self) -> float:
        """Calculate market risk based on market data"""
        try:
            # Get market data
            supply_data = self.market_analysis['supply_analysis']['current_pipeline']
            total_pipeline = supply_data['total_units']  # 5,877 units
            market_factors = self.market_analysis['market_factors']
            
            # 1. Supply Risk (35% weight)
            # Normalize to total pipeline
            supply_risk = min(35, (total_pipeline / 5877) * 35)
            
            # 2. Rate Environment Risk (35% weight)
            # Consider expected 100bps cut
            rate_data = market_factors['interest_rates']
            current_rate = rate_data['current']['rates']['5yr_fixed']
            expected_cut = 1.0
            rate_risk = min(35, abs(current_rate - expected_cut) * 35)
            
            # 3. Economic Risk (30% weight)
            employment_data = market_factors['employment']['current_statistics']
            employment_rate = employment_data['employment_rate']
            economic_risk = min(30, (.01 - employment_rate) * 30)
            
            return supply_risk + rate_risk + economic_risk
            
        except Exception as e:
            print(f"Error calculating market risk: {str(e)}")
            return 35.0
        
    def _identify_risk_factors(self) -> List[Dict]:
        """Identify key risk factors with rate cut opportunity focus"""
        return [
            {
                'factor': 'Interest Rates',
                'impact': 'Positive',
                'likelihood': 'High',
                'description': 'Expected 100bps cut by Q1 2025 supports absorption',
                'opportunity': '+12% absorption benefit per 1% rate decrease'
            },
            {
                'factor': 'Market Competition',
                'impact': 'Moderate',
                'likelihood': 'High',
                'description': 'Manhattan and Parkway 2 validating premium pricing',
                'opportunity': 'Rate cuts support premium positioning'
            },
            {
                'factor': 'Supply Pipeline',
                'impact': 'Moderate',
                'likelihood': 'High',
                'description': 'Staggered completions 2025-2029 reduce pressure',
                'opportunity': 'Strategic launch timing with rate cuts'
            }
        ]
        
    def _generate_risk_mitigation(self) -> List[Dict]:
        """Generate risk mitigation strategies leveraging rate cut timing"""
        return [
            {
                'risk': 'Interest Rate Timing',
                'strategies': [
                    'Launch aligned with expected Q1 2025 rate cuts',
                    'Structure rate buy-down programs as bridge to cuts',
                    'Develop rate-sensitive incentive programs'
                ],
                'opportunity': 'Leverage rate cut momentum for absorption'
            },
            {
                'risk': 'Market Competition',
                'strategies': [
                    'Premium positioning supported by rate environment',
                    'Flexible pricing to capture rate cut benefits',
                    'Strategic inventory release with rate cycle'
                ],
                'opportunity': 'Rate cuts support premium pricing power'
            },
            {
                'risk': 'Supply Pipeline',
                'strategies': [
                    'Phase releases to align with rate cuts',
                    'Build pricing power through 2025 rate cycle',
                    'Target 65% absorption by Q4 2025'
                ],
                'opportunity': 'Rate cuts accelerate absorption targets'
            }
        ]