# main.py

import json
import os
from data_processor import DataProcessor
from market_analyzer import MarketAnalyzer
from market_simulator import MarketSimulator
from report_generator import ReportGenerator

def run_market_analysis(completions_file: str, 
                       pricing_file: str, 
                       rates_file: str,
                       excel_output: str = 'Surrey_Market_Analysis.xlsx',
                       pdf_output: str = 'Surrey_Market_Analysis.pdf',
                       data_dir: str = 'data',
                       use_cached_data: bool = True) -> None:
    """
    Run complete market analysis and generate reports
    """
    print("Starting market analysis...")
    
    processed_data = {}
    
    # Try to load existing processed data if requested
    if use_cached_data:
        try:
            print("Loading cached processed data...")
            project_data_path = f"{data_dir}/processed/surrey_project_data.json"
            macro_data_path = f"{data_dir}/processed/surrey_macro_data.json"
            
            if os.path.exists(project_data_path) and os.path.exists(macro_data_path):
                with open(project_data_path, 'r') as f:
                    project_data = json.load(f)
                    processed_data['project_data'] = project_data['content']['project_data']
                    
                with open(macro_data_path, 'r') as f:
                    macro_data = json.load(f)
                    processed_data['macro_data'] = macro_data['content']['macro_indicators']
                    
                print("Successfully loaded cached data")
            else:
                use_cached_data = False
                print("Cached data files not found")
        except Exception as e:
            use_cached_data = False
            print(f"Error loading cached data: {str(e)}")
    
    # Process data if needed
    if not use_cached_data:
        print("Processing input data...")
        processor = DataProcessor(data_dir)
        processed_data = processor.process_all_data(
            completions_file,
            pricing_file,
            rates_file
        )
    
    # 2. Analyze Market
    print("Analyzing market conditions...")
    analyzer = MarketAnalyzer(processed_data['project_data'], processed_data['macro_data'])
    market_analysis = analyzer.analyze_market()
    
    # Get market metrics for pricing strategy
    market_metrics = processed_data['project_data']['market_metrics']
    base_psf = market_metrics['pricing_trends']['market_average_psf']
    
    # 3. Run Simulations
    print("Running market simulations...")
    simulator = MarketSimulator(market_analysis, {
        'base_psf': base_psf,
        'absorption_trends': market_metrics['absorption_trends'],
        'price_sensitivity': market_metrics['absorption_trends']['price_sensitivity']
    })
    
    simulation_results = simulator.run_simulations()
    
    # 4. Generate Reports
    print("Generating reports...")
    report_gen = ReportGenerator(
        market_analyzer=analyzer,
        pricing_strategy={
            'base_psf': base_psf,
            'absorption_trends': market_metrics['absorption_trends'],
            'pricing_trends': market_metrics['pricing_trends']
        },
        simulation_results=simulation_results
    )
    
    # Generate Excel report
    print("Generating Excel report...")
    report_gen.generate_report(excel_output)
    
    # Generate PDF report
    print("Generating PDF report...")
    report_gen.generate_pdf_report(pdf_output)
    
    print(f"Analysis complete! Results saved to:")
    print(f"- Excel: {excel_output}")
    print(f"- PDF: {pdf_output}")

# Example usage
if __name__ == "__main__":
    run_market_analysis(
        'Condo_completions_owner_Surrey.csv',
        'Surrey pricing.csv',
        'discounted_mortgage_rates.csv',
        excel_output='Surrey_Market_Analysis.xlsx',
        pdf_output='Surrey_Market_Analysis.pdf',
        data_dir='data',
        use_cached_data=True
    )