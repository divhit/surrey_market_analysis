import json
from scenario_analyzer import PricingScenarioAnalyzer
from typing import Dict, Any

def load_data() -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Load project and macro data"""
    with open('data/processed/surrey_project_data.json', 'r') as f:
        project_data = json.load(f)['content']['project_data']
    
    with open('data/processed/surrey_macro_data.json', 'r') as f:
        macro_data = json.load(f)['content']['macro_indicators']
    
    return project_data, macro_data

def get_scenario_input() -> Dict[str, float]:
    """Get scenario parameters from user"""
    print("\nMarket Scenario Analysis")
    print("------------------------")
    print("Enter changes for the following parameters (press Enter to skip):")
    
    scenario = {}
    
    # Supply change
    supply = input("\nSupply change (e.g. 10 for 10% increase): ")
    if supply.strip():
        scenario['supply_change'] = float(supply)
    
    # Interest rate change
    rates = input("Interest rate change in basis points (e.g. -50 for 0.5% decrease): ")
    if rates.strip():
        scenario['interest_rate_change'] = float(rates)
    
    # Employment change
    employment = input("Employment rate change (e.g. 2 for 2% increase): ")
    if employment.strip():
        scenario['employment_change'] = float(employment)
    
    # Competitor pricing
    pricing = input("Competitor pricing change (e.g. -5 for 5% decrease): ")
    if pricing.strip():
        scenario['competitor_pricing_change'] = float(pricing)
    
    return scenario

def main():
    """Main CLI interface"""
    try:
        # Load data
        print("Loading market data...")
        project_data, macro_data = load_data()
        
        # Initialize analyzer
        analyzer = PricingScenarioAnalyzer(project_data, macro_data)
        
        while True:
            # Get scenario input
            scenario = get_scenario_input()
            
            if not scenario:
                print("\nNo changes specified. Exiting...")
                break
            
            # Run analysis
            print("\nAnalyzing scenario...")
            results = analyzer.analyze_scenario(scenario)
            
            # Generate and print report
            report = analyzer.generate_report(results)
            print("\n" + report)
            
            # Ask to continue
            again = input("\nWould you like to analyze another scenario? (y/n): ")
            if again.lower() != 'y':
                break
    
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Please check your input and try again.")

if __name__ == "__main__":
    main() 