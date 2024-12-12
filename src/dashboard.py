import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scenario_analyzer import PricingScenarioAnalyzer
from market_analyzer import MarketAnalyzer
from report_generator import ReportGenerator
import os
from datetime import datetime
from openai import OpenAI
import subprocess
import glob
import time
from typing import Dict, Union

# Initialize OpenAI client with API key from Streamlit secrets
if 'OPENAI_API_KEY' not in st.secrets:
    st.error('OPENAI_API_KEY is not set in the Streamlit secrets.')
    st.stop()

client = OpenAI(api_key=st.secrets['OPENAI_API_KEY'])

# Page config
st.set_page_config(
    page_title="Surrey Market Pricing Dashboard",
    page_icon="📊",
    layout="wide"
)

@st.cache_data
def load_data():
    """Load and cache market data"""
    with open('data/processed/surrey_project_data.json', 'r') as f:
        project_data = json.load(f)['content']['project_data']
    
    with open('data/processed/surrey_macro_data.json', 'r') as f:
        macro_data = json.load(f)['content']['macro_indicators']
    
    return project_data, macro_data

def create_psf_chart(unit_impacts):
    # Create bar chart data
    unit_types = list(unit_impacts.keys())
    base_psf = [impact['base_psf'] for impact in unit_impacts.values()]
    new_psf = [impact['scenario_psf'] for impact in unit_impacts.values()]
    
    fig = go.Figure(data=[
        go.Bar(name='Base PSF', x=unit_types, y=base_psf),
        go.Bar(name='New PSF', x=unit_types, y=new_psf)
    ])
    
    fig.update_layout(
        title='PSF Comparison by Unit Type',
        yaxis_title='Price per Square Foot ($)',
        barmode='group'
    )
    
    return fig

def generate_excel_report(scenario_results, scenario_description):
    """Generate Excel report with detailed analysis"""
    df_dict = {}
    
    # Pricing Analysis
    pricing_df = pd.DataFrame([
        {
            'Unit Type': ut.title(),
            'Current PSF': impact['base_psf'],
            'New PSF': impact['scenario_psf'],
            'Change ($)': impact['psf_change'],
            'Change (%)': impact['percent_change']
        }
        for ut, impact in scenario_results['unit_impacts'].items()
    ])
    df_dict['Pricing Analysis'] = pricing_df
    
    # Revenue Analysis
    unit_mix = {
        'studios': {'count': 34, 'avg_size': 379},
        'one_bed': {'count': 204, 'avg_size': 474},
        'two_bed': {'count': 120, 'avg_size': 795},
        'three_bed': {'count': 18, 'avg_size': 941}
    }
    
    revenue_data = []
    for unit_type, impact in scenario_results['unit_impacts'].items():
        mix = unit_mix[unit_type]
        current_revenue = impact['base_psf'] * mix['avg_size'] * mix['count']
        new_revenue = impact['scenario_psf'] * mix['avg_size'] * mix['count']
        
        revenue_data.append({
            'Unit Type': unit_type.title(),
            'Unit Count': mix['count'],
            'Average Size': mix['avg_size'],
            'Current Revenue': current_revenue,
            'New Revenue': new_revenue,
            'Revenue Change': new_revenue - current_revenue,
            'Revenue Change (%)': ((new_revenue / current_revenue) - 1) * 100 if current_revenue > 0 else 0
        })
    
    df_dict['Revenue Analysis'] = pd.DataFrame(revenue_data)
    
    # Market Conditions
    market_df = pd.DataFrame([{
        'Metric': 'Monthly Absorption Rate',
        'Current': 5.4,
        'New': 5.4 * (1 + scenario_results['market_conditions']['absorption_impact']/100),
        'Change (%)': scenario_results['market_conditions']['absorption_impact']
    }])
    df_dict['Market Conditions'] = market_df
    
    # Save to Excel
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'scenario_analysis_{timestamp}.xlsx'
    
    with pd.ExcelWriter(filename) as writer:
        for sheet_name, df in df_dict.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            worksheet = writer.sheets[sheet_name]
            for idx, col in enumerate(df.columns):
                max_length = max(df[col].astype(str).apply(len).max(), len(col))
                worksheet.column_dimensions[chr(65 + idx)].width = max_length + 2
    
    return filename

def process_query_with_gpt(query: str) -> dict:
    """Process natural language query using GPT-4-mini to extract scenario parameters"""
    
    system_prompt = """You are a real estate market analysis assistant. Extract numerical parameters from queries about market scenarios.
    Focus on these parameters:
    - interest_rate_change (in bps)
    - supply_change (in %)
    - employment_change (in %)
    - competitor_pricing_change (in %)
    
    Return only a JSON object with the parameters you find. Use negative numbers for decreases.
    Example: For "what if rates drop 50 bps" return {"interest_rate_change": -50}"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Using exact model name as specified
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.1
        )
        
        # Extract the JSON response
        result = json.loads(response.choices[0].message.content)
        
        # Ensure employment_change is included if not present
        if 'employment_change' not in result:
            result['employment_change'] = 0
            
        return result
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")
        return {}

def run_staged_analysis():
    """Run the base analysis in stages with informative outputs"""
    
    # Initialize all stages as empty containers
    supply_container = st.empty()
    supply_progress = st.empty()
    supply_expander = st.empty()
    
    historical_container = st.empty()
    historical_progress = st.empty()
    historical_expander = st.empty()
    
    competitive_container = st.empty()
    competitive_progress = st.empty()
    competitive_expander = st.empty()
    
    macro_container = st.empty()
    macro_progress = st.empty()
    macro_expander = st.empty()
    
    pricing_container = st.empty()
    pricing_progress = st.empty()
    pricing_expander = st.empty()
    
    try:
        # Load data
        project_data, macro_data = load_data()
        
        # Stage 1: Market Supply Analysis
        supply_container.write("🔄 Stage 1: Analyzing Market Supply...")
        for i in range(100):
            supply_progress.progress(i + 1)
            time.sleep(0.02)
        
        with supply_expander.expander("Supply Analysis Results", expanded=True):
            st.write("Current Inventory Status:")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Active Projects", "2,672 units")
                st.metric("Standing Inventory", "895 units")
            with col2:
                st.metric("Total Pipeline", "7,053 units")
                st.metric("Months of Supply", "14.2 months")
            
            st.write("Construction Activity (Monthly Trend):")
            construction_data = {
                'Period': [
                    # 2019
                    '2019-01', '2019-02', '2019-03', '2019-04', '2019-05', '2019-06',
                    '2019-07', '2019-08', '2019-09', '2019-10', '2019-11', '2019-12',
                    # 2020
                    '2020-01', '2020-02', '2020-03', '2020-04', '2020-05', '2020-06',
                    '2020-07', '2020-08', '2020-09', '2020-10', '2020-11', '2020-12',
                    # 2021
                    '2021-01', '2021-02', '2021-03', '2021-04', '2021-05', '2021-06',
                    '2021-07', '2021-08', '2021-09', '2021-10', '2021-11', '2021-12',
                    # 2022
                    '2022-01', '2022-02', '2022-03', '2022-04', '2022-05', '2022-06',
                    '2022-07', '2022-08', '2022-09', '2022-10', '2022-11', '2022-12',
                    # 2023
                    '2023-01', '2023-02', '2023-03', '2023-04', '2023-05', '2023-06',
                    '2023-07', '2023-08', '2023-09', '2023-10', '2023-11', '2023-12',
                    # 2024
                    '2024-01', '2024-02', '2024-03', '2024-04', '2024-05', '2024-06',
                    '2024-07', '2024-08', '2024-09'
                ],
                'Starts': [
                    # 2019
                    0, 0, 0, 492, 72, 422, 0, 54, 0, 231, 328, 71,
                    # 2020
                    27, 58, 0, 0, 0, 0, 507, 0, 79, 351, 0, 40,
                    # 2021
                    0, 0, 539, 112, 147, 732, 93, 108, 123, 300, 0, 0,
                    # 2022
                    0, 109, 0, 0, 221, 0, 36, 86, 9, 619, 339, 940,
                    # 2023
                    193, 287, 631, 758, 47, 658, 971, 335, 262, 817, 201, 93,
                    # 2024
                    331, 795, 574, 153, 231, 85, 573, 574, 166
                ],
                'Completions': [
                    # 2019
                    126, 0, 117, 89, 124, 0, 178, 0, 485, 93, 0, 0,
                    # 2020
                    29, 305, 0, 0, 398, 0, 0, 0, 3, 0, 0, 190,
                    # 2021
                    0, 0, 55, 161, 0, 125, 0, 0, 765, 0, 0, 67,
                    # 2022
                    112, 0, 140, 0, 550, 156, 166, 0, 0, 0, 351, 0,
                    # 2023
                    71, 0, 0, 241, 0, 0, 108, 598, 0, 0, 179, 145,
                    # 2024
                    294, 0, 36, 141, 35, 77, 96, 0, 514
                ]
            }
            df_construction = pd.DataFrame(construction_data)
            df_construction.set_index('Period', inplace=True)
            st.line_chart(df_construction)
        
        # Stage 2: Historical Performance Analysis
        historical_container.write("🔄 Stage 2: Analyzing Historical Performance...")
        for i in range(100):
            historical_progress.progress(i + 1)
            time.sleep(0.02)
        
        with historical_expander.expander("Historical Analysis Results", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Market Monthly Absorption", "3.6%")
                st.metric("Sales Velocity Index", "1.2")
            with col2:
                st.metric("Historical Price Growth", "+24.6%")
                st.metric("Price Growth CAGR", "4.5%")
        time.sleep(1)
        
        # Stage 3: Competitive Analysis
        competitive_container.write("🔄 Stage 3: Analyzing Competitive Landscape...")
        for i in range(100):
            competitive_progress.progress(i + 1)
            time.sleep(0.02)
        
        with competitive_expander.expander("Competitive Analysis Results", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Market Average PSF", "$1,145")
                st.metric("Our Position vs Market", "Above Average")
            with col2:
                st.metric("Active Competitors", "7 projects")
                st.metric("Competitive Price Range", "$1,018 - $1,230 PSF")
        time.sleep(1)
        
        # Stage 4: Macro Factor Analysis
        macro_container.write("🔄 Stage 4: Analyzing Macro Factors...")
        for i in range(100):
            macro_progress.progress(i + 1)
            time.sleep(0.02)
        
        with macro_expander.expander("Macro Analysis Results", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Employment Rate", "62.4%", "+0.2%")
                st.metric("Median Household Income", "$98,670", "+3.2%")
            with col2:
                st.metric("5-Year Fixed Rate", "4.52%", "-0.27%")
                st.metric("Prime Rate", "6.95%", "+0.25%")
            
            st.write("Historical Interest Rate Trends:")
            rate_data = {
                'Period': [
                    '2020-Q1', '2020-Q2', '2020-Q3', '2020-Q4',
                    '2021-Q1', '2021-Q2', '2021-Q3', '2021-Q4',
                    '2022-Q1', '2022-Q2', '2022-Q3', '2022-Q4',
                    '2023-Q1', '2023-Q2', '2023-Q3', '2023-Q4',
                    '2024-Q1', '2024-Q2'  # Added 2024 projections
                ],
                '5-Year Fixed': [
                    2.89, 2.45, 2.14, 1.99,
                    2.14, 2.45, 2.89, 3.24,
                    3.89, 4.25, 4.89, 5.12,
                    5.24, 4.89, 4.67, 4.52,
                    4.45, 4.25  # 2024 projections
                ],
                'Prime Rate': [
                    3.95, 3.45, 2.95, 2.45,
                    2.45, 2.45, 2.45, 2.45,
                    3.70, 4.70, 5.45, 6.45,
                    6.70, 6.95, 6.95, 6.95,
                    6.70, 6.45  # 2024 projections
                ]
            }
            df_rates = pd.DataFrame(rate_data)
            df_rates.set_index('Period', inplace=True)
            st.line_chart(df_rates)
        time.sleep(1)
        
        # Stage 5: Price Optimization
        pricing_container.write("🔄 Stage 5: Optimizing Pricing Strategy...")
        for i in range(100):
            pricing_progress.progress(i + 1)
            time.sleep(0.02)
        
        with pricing_expander.expander("Pricing Strategy Results", expanded=True):
            pricing_data = {
                'Unit Type': ['Studios', 'One Bed', 'Two Bed', 'Three Bed'],
                'Target PSF': ['$1,241.95', '$1,146.74', '$1,068.63', '$1,036.32'],
                'Monthly Absorption': ['5.4%', '5.4%', '5.4%', '5.4%']
            }
            st.dataframe(pd.DataFrame(pricing_data))
        time.sleep(1)
        
        # Final Report Generation
        st.success("✅ Analysis Complete! Generating Excel Report...")
        
        # Initialize analyzers
        analyzer = MarketAnalyzer(project_data, macro_data)
        market_analysis = analyzer.analyze_market()
        
        # Get market metrics for pricing strategy
        market_metrics = project_data['market_metrics']
        base_psf = market_metrics['pricing_trends']['market_average_psf']
        
        # Generate Excel report directly
        report_gen = ReportGenerator(
            market_analyzer=analyzer,
            pricing_strategy={
                'base_psf': base_psf,
                'absorption_trends': market_metrics['absorption_trends'],
                'pricing_trends': market_metrics['pricing_trends']
            },
            simulation_results=None  # We're not running simulations in the dashboard
        )
        
        excel_output = 'Surrey_Market_Analysis.xlsx'
        report_gen.generate_report(excel_output)
        
        # Offer download button for the generated Excel file
        with open(excel_output, 'rb') as f:
            st.download_button(
                label="📊 Download Complete Market Analysis",
                data=f,
                file_name='Surrey_Market_Analysis.xlsx',
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="base_download"
            )
            
    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")
        st.stop()

def run_scenario_analysis(scenario: dict, results: dict) -> tuple[str, str]:
    """Run the main.py script with the given scenario parameters and calculated PSFs"""
    try:
        # Create a temporary JSON file with the new PSFs
        scenario_data = {
            'studios': results['unit_impacts']['studios']['scenario_psf'],
            'one_bed': results['unit_impacts']['one_bed']['scenario_psf'],
            'two_bed': results['unit_impacts']['two_bed']['scenario_psf'],
            'three_bed': results['unit_impacts']['three_bed']['scenario_psf']
        }
        
        with open('scenario_psfs.json', 'w') as f:
            json.dump(scenario_data, f)
        
        # Run main.py with the scenario file
        subprocess.run(["python", "src/main.py", "--scenario-file", "scenario_psfs.json"], check=True)
        
        # Clean up
        os.remove('scenario_psfs.json')
        
        if os.path.exists('Surrey_Market_Analysis.xlsx'):
            return "Scenario analysis completed successfully!", 'Surrey_Market_Analysis.xlsx'
        return "Analysis completed but couldn't find the Surrey Market Analysis file.", ""
    except subprocess.CalledProcessError as e:
        return f"Error running analysis: {str(e)}", ""
    except Exception as e:
        return f"Error during analysis: {str(e)}", ""

def generate_revenue_tables(results: dict) -> str:
    """Generate Excel file with revenue tables based on scenario PSFs"""
    # Unit mix data
    unit_mix = {
        'studios': {'count': 34, 'avg_size': 379},
        'one_bed': {'count': 204, 'avg_size': 474},
        'two_bed': {'count': 120, 'avg_size': 795},
        'three_bed': {'count': 18, 'avg_size': 941}
    }
    
    # Fixed absorption targets
    absorption_targets = {
        '3_Month': {'target': 0.50, 'months': 3},    # 50% by month 3
        '12_Month': {'target': 0.65, 'months': 12},   # 65% by month 12
        '24_Month': {'target': 0.825, 'months': 24},  # 82.5% by month 24
        '36_Month': {'target': 1.00, 'months': 36}    # 100% by month 36
    }
    
    # Monthly absorption targets by year
    yearly_targets = {
        1: {'April': 0.1667, 'May': 0.1667, 'June': 0.1667,  # Year 1 (16.67% for Q1)
            'July': 0.0167, 'August': 0.0167, 'September': 0.0167,
            'October': 0.0167, 'November': 0.0167, 'December': 0.0167,
            'January': 0.0167, 'February': 0.0167, 'March': 0.0167},
        2: {'April': 0.0146, 'May': 0.0146, 'June': 0.0146,  # Year 2 (1.46% each month)
            'July': 0.0146, 'August': 0.0146, 'September': 0.0146,
            'October': 0.0146, 'November': 0.0146, 'December': 0.0146,
            'January': 0.0146, 'February': 0.0146, 'March': 0.0146},
        3: {'April': 0.0146, 'May': 0.0146, 'June': 0.0146,  # Year 3 (1.46% each month)
            'July': 0.0146, 'August': 0.0146, 'September': 0.0146,
            'October': 0.0146, 'November': 0.0146, 'December': 0.0146,
            'January': 0.0146, 'February': 0.0146, 'March': 0.0146}
    }
    
    # Monthly pricing patterns with incentives
    monthly_patterns = {
        'April': {'price_factor': 1.000, 'incentives': {'studios': 5.5, 'one_bed': 5.5, 'two_bed': 6.0, 'three_bed': 6.5}},
        'May': {'price_factor': 1.000, 'incentives': {'studios': 5.5, 'one_bed': 5.5, 'two_bed': 6.0, 'three_bed': 6.5}},
        'June': {'price_factor': 0.995, 'incentives': {'studios': 5.0, 'one_bed': 5.0, 'two_bed': 5.5, 'three_bed': 6.0}},
        'July': {'price_factor': 0.970, 'incentives': {'studios': 2.5, 'one_bed': 2.5, 'two_bed': 3.0, 'three_bed': 3.5}},
        'August': {'price_factor': 0.970, 'incentives': {'studios': 2.5, 'one_bed': 2.5, 'two_bed': 3.0, 'three_bed': 3.5}},
        'September': {'price_factor': 0.970, 'incentives': {'studios': 2.5, 'one_bed': 2.5, 'two_bed': 3.0, 'three_bed': 3.5}},
        'October': {'price_factor': 0.975, 'incentives': {'studios': 3.0, 'one_bed': 3.0, 'two_bed': 3.5, 'three_bed': 4.0}},
        'November': {'price_factor': 0.980, 'incentives': {'studios': 3.5, 'one_bed': 3.5, 'two_bed': 4.0, 'three_bed': 4.5}},
        'December': {'price_factor': 0.980, 'incentives': {'studios': 3.5, 'one_bed': 3.5, 'two_bed': 4.0, 'three_bed': 4.5}},
        'January': {'price_factor': 0.985, 'incentives': {'studios': 4.0, 'one_bed': 4.0, 'two_bed': 4.5, 'three_bed': 5.0}},
        'February': {'price_factor': 0.990, 'incentives': {'studios': 4.5, 'one_bed': 4.5, 'two_bed': 5.0, 'three_bed': 5.5}},
        'March': {'price_factor': 0.990, 'incentives': {'studios': 4.5, 'one_bed': 4.5, 'two_bed': 5.0, 'three_bed': 5.5}}
    }
    
    # Create a timestamp for the filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'Surrey_Market_Analysis_Scenario_{timestamp}.xlsx'
    
    # Create Excel writer
    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        workbook = writer.book
        
        # Formats
        money_fmt = workbook.add_format({'num_format': '$#,##0'})
        money_psf_fmt = workbook.add_format({'num_format': '$#,##0.00'})
        percent_fmt = workbook.add_format({'num_format': '0.0%'})
        percent_small_fmt = workbook.add_format({'num_format': '0.0%'})
        
        # Calculate weighted average incentives for each period
        period_incentives = {}
        for period_name, period_info in absorption_targets.items():
            months_to_consider = period_info['months']
            total_absorption = 0
            weighted_incentives = {
                'studios': 0,
                'one_bed': 0,
                'two_bed': 0,
                'three_bed': 0
            }
            
            # Calculate weighted incentives based on monthly absorption
            month_count = 0
            for year in range(1, 4):  # 3 years
                for month, absorption in yearly_targets[year].items():
                    if month_count >= months_to_consider:
                        break
                    
                    # Add weighted incentives for each unit type
                    for unit_type in weighted_incentives.keys():
                        incentive = monthly_patterns[month]['incentives'][unit_type]
                        weighted_incentives[unit_type] += incentive * absorption
                    
                    total_absorption += absorption
                    month_count += 1
                if month_count >= months_to_consider:
                    break
            
            # Normalize weighted incentives
            if total_absorption > 0:
                for unit_type in weighted_incentives:
                    weighted_incentives[unit_type] /= total_absorption
            
            period_incentives[period_name] = weighted_incentives
        
        for period_name, target_absorption in absorption_targets.items():
            # Calculate units for this absorption target
            revenue_data = []
            revenue_data_net = []
            total_volume = 0
            total_volume_net = 0
            
            for unit_type, impact in results['unit_impacts'].items():
                mix = unit_mix[unit_type]
                total_units = mix['count']
                units_sold = round(total_units * target_absorption['target'])
                net_psf = impact['scenario_psf']  # This is the net PSF
                incentive_pct = period_incentives[period_name][unit_type] / 100
                
                # Back-calculate gross PSF from net PSF and incentive
                gross_psf = net_psf / (1 - incentive_pct)
                
                # Gross Revenue Analysis
                volume = units_sold * gross_psf * mix['avg_size']
                total_volume += volume
                
                revenue_data.append({
                    'Unit Type': unit_type.title(),
                    'Units': units_sold,
                    '% Total': units_sold / (376 * target_absorption['target']),
                    '$ Volume': volume,
                    '% Total $': 0,  # Will be calculated after total is known
                    'Total SF': units_sold * mix['avg_size'],
                    'Incentive': f"{incentive_pct*100:.1f}%",
                    'Avg PSF': gross_psf,
                    'Avg Size': mix['avg_size']
                })
                
                # Net Revenue Analysis (After Incentives)
                volume_net = units_sold * net_psf * mix['avg_size']
                total_volume_net += volume_net
                
                revenue_data_net.append({
                    'Unit Type': unit_type.title(),
                    'Units': units_sold,
                    '% Total': units_sold / (376 * target_absorption['target']),
                    '$ Volume': volume_net,
                    '% Total $': 0,  # Will be calculated after total is known
                    'Total SF': units_sold * mix['avg_size'],
                    'Incentive': f"{incentive_pct*100:.1f}%",
                    'Avg PSF': net_psf,
                    'Avg Size': mix['avg_size']
                })
            
            # Calculate percentage of total volume
            for data in revenue_data:
                data['% Total $'] = data['$ Volume'] / total_volume
            for data in revenue_data_net:
                data['% Total $'] = data['$ Volume'] / total_volume_net
            
            # Create DataFrames
            df_gross = pd.DataFrame(revenue_data)
            df_net = pd.DataFrame(revenue_data_net)
            
            # Write to Excel
            sheet_name = f'{period_name}_Revenue'
            
            # Write title
            worksheet = workbook.add_worksheet(sheet_name)
            worksheet.write(0, 0, f'Revenue Analysis - Target {int(target_absorption["target"] * 100)}% Absorption')
            
            # Write Gross Revenue Analysis
            worksheet.write(2, 0, 'Gross Revenue Analysis')
            start_row = 4
            headers = ['Unit Type', 'Units', '% Total', '$ Volume', '% Total $', 'Total SF', 'Incentive', 'Avg PSF', 'Avg Size']
            
            for col, header in enumerate(headers):
                worksheet.write(start_row, col, header)
            
            row = start_row + 1
            for data in revenue_data:
                worksheet.write(row, 0, data['Unit Type'])
                worksheet.write(row, 1, data['Units'])
                worksheet.write(row, 2, data['% Total'], percent_fmt)
                worksheet.write(row, 3, data['$ Volume'], money_fmt)
                worksheet.write(row, 4, data['% Total $'], percent_fmt)
                worksheet.write(row, 5, data['Total SF'])
                worksheet.write(row, 6, data['Incentive'])
                worksheet.write(row, 7, data['Avg PSF'], money_psf_fmt)
                worksheet.write(row, 8, data['Avg Size'])
                row += 1
            
            # Write total row
            worksheet.write(row, 0, 'Total')
            worksheet.write_formula(row, 1, f'=SUM(B{start_row+2}:B{row})')
            worksheet.write_formula(row, 2, '=1', percent_fmt)
            worksheet.write_formula(row, 3, f'=SUM(D{start_row+2}:D{row})', money_fmt)
            worksheet.write_formula(row, 4, '=1', percent_fmt)
            worksheet.write_formula(row, 5, f'=SUM(F{start_row+2}:F{row})')
            
            # Write Net Revenue Analysis
            row += 3
            worksheet.write(row, 0, 'Net Revenue Analysis (After Incentives)')
            row += 1
            
            for col, header in enumerate(headers):
                worksheet.write(row, col, header)
            
            start_row = row
            row += 1
            
            for data in revenue_data_net:
                worksheet.write(row, 0, data['Unit Type'])
                worksheet.write(row, 1, data['Units'])
                worksheet.write(row, 2, data['% Total'], percent_fmt)
                worksheet.write(row, 3, data['$ Volume'], money_fmt)
                worksheet.write(row, 4, data['% Total $'], percent_fmt)
                worksheet.write(row, 5, data['Total SF'])
                worksheet.write(row, 6, data['Incentive'])
                worksheet.write(row, 7, data['Avg PSF'], money_psf_fmt)
                worksheet.write(row, 8, data['Avg Size'])
                row += 1
            
            # Write total row
            worksheet.write(row, 0, 'Total')
            worksheet.write_formula(row, 1, f'=SUM(B{start_row+2}:B{row})')
            worksheet.write_formula(row, 2, '=1', percent_fmt)
            worksheet.write_formula(row, 3, f'=SUM(D{start_row+2}:D{row})', money_fmt)
            worksheet.write_formula(row, 4, '=1', percent_fmt)
            worksheet.write_formula(row, 5, f'=SUM(F{start_row+2}:F{row})')
            
            # Format columns
            worksheet.set_column('A:A', 15)
            worksheet.set_column('B:B', 10)
            worksheet.set_column('C:C', 10)
            worksheet.set_column('D:D', 15)
            worksheet.set_column('E:E', 10)
            worksheet.set_column('F:F', 12)
            worksheet.set_column('G:G', 12)
            worksheet.set_column('H:H', 12)
            worksheet.set_column('I:I', 10)
        
        # Monthly Pricing Strategy
        monthly_data = []
        base_months = ['April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December', 'January', 'February', 'March']
        
        # Generate 36 months of data
        for year in range(1, 4):
            for month in base_months:
                pattern = monthly_patterns[month]
                target = yearly_targets[year][month]
                
                row_data = {'Month': month, 'Target %': target}
                
                for unit_type, impact in results['unit_impacts'].items():
                    net_psf = impact['scenario_psf']  # This is the net PSF
                    incentive = pattern['incentives'][unit_type]
                    # Back-calculate gross PSF
                    gross_psf = net_psf / (1 - incentive/100)
                    
                    row_data[f'{unit_type}_gross'] = gross_psf
                    row_data[f'{unit_type}_incentive'] = incentive
                    row_data[f'{unit_type}_net'] = net_psf
                
                monthly_data.append(row_data)
        
        # Create monthly pricing sheet
        monthly_df = pd.DataFrame(monthly_data)
        worksheet = workbook.add_worksheet('Monthly_Pricing')
        worksheet.write(0, 0, 'Monthly Absorption & Pricing Strategy')
        
        # Write headers
        row = 1
        worksheet.write(row, 0, 'Month')
        worksheet.write(row, 1, 'Target %')
        
        col = 2
        for unit_type in ['Studios', 'One Bed', 'Two Bed', 'Three Bed']:
            worksheet.write(row, col, unit_type)
            worksheet.write(row + 1, col, 'Gross PSF')
            worksheet.write(row + 1, col + 1, 'Incentive')
            worksheet.write(row + 1, col + 2, 'Net PSF')
            col += 3
        
        # Write data
        row = 3
        for _, data in monthly_df.iterrows():
            worksheet.write(row, 0, data['Month'])
            worksheet.write(row, 1, data['Target %'], percent_fmt)
            
            col = 2
            for unit_type in ['studios', 'one_bed', 'two_bed', 'three_bed']:
                worksheet.write(row, col, data[f'{unit_type}_gross'], money_psf_fmt)
                worksheet.write(row, col + 1, f"{data[f'{unit_type}_incentive']}%")
                worksheet.write(row, col + 2, data[f'{unit_type}_net'], money_psf_fmt)
                col += 3
            row += 1
        
        # Format columns
        worksheet.set_column('A:A', 12)
        worksheet.set_column('B:B', 10)
        for i in range(2, 14, 3):
            worksheet.set_column(f'{chr(65+i)}:{chr(65+i+2)}', 12)
    
    return filename

def calculate_total_revenue(psf_data: dict, is_scenario: bool = False) -> tuple[dict, dict]:
    """Calculate total revenue for given PSFs and absorption target"""
    unit_mix = {
        'studios': {'count': 34, 'avg_size': 379},
        'one_bed': {'count': 204, 'avg_size': 474},
        'two_bed': {'count': 120, 'avg_size': 795},
        'three_bed': {'count': 18, 'avg_size': 941}
    }
    
    revenue = {
        '3_Month': 0,
        '12_Month': 0,
        '24_Month': 0,
        '36_Month': 0
    }
    
    # Track revenue by unit type
    revenue_by_type = {
        'studios': {'3_Month': 0, '12_Month': 0, '24_Month': 0, '36_Month': 0},
        'one_bed': {'3_Month': 0, '12_Month': 0, '24_Month': 0, '36_Month': 0},
        'two_bed': {'3_Month': 0, '12_Month': 0, '24_Month': 0, '36_Month': 0},
        'three_bed': {'3_Month': 0, '12_Month': 0, '24_Month': 0, '36_Month': 0}
    }
    
    absorption_targets = {
        '3_Month': 0.50,    # 50% by month 3
        '12_Month': 0.65,   # 65% by month 12
        '24_Month': 0.825,  # 82.5% by month 24
        '36_Month': 1.00    # 100% by month 36
    }
    
    for period, target in absorption_targets.items():
        period_revenue = 0
        for unit_type, mix in unit_mix.items():
            units_sold = round(mix['count'] * target)
            # Get PSF based on whether this is scenario or base data
            if is_scenario:
                psf = psf_data[unit_type]['scenario_psf']
            else:
                psf = psf_data[unit_type]
            revenue_unit_type = units_sold * psf * mix['avg_size']
            period_revenue += revenue_unit_type
            revenue_by_type[unit_type][period] = revenue_unit_type
        revenue[period] = period_revenue
    
    return revenue, revenue_by_type

def create_sentiment_gauge(sentiment: str) -> go.Figure:
    """Create a gauge chart for market sentiment"""
    # Map sentiment to numerical value
    sentiment_map = {
        'Strongly Negative': 1,
        'Moderately Negative': 2,
        'Slightly Negative': 3,
        'Neutral': 4,
        'Slightly Positive': 5,
        'Moderately Positive': 6,
        'Strongly Positive': 7
    }
    
    # Print sentiment for debugging
    print(f"Received sentiment: {sentiment}")
    
    value = sentiment_map.get(sentiment, 4)  # Default to Neutral if sentiment not found
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",  # Removed delta as it's not needed
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {
                'range': [1, 7], 
                'ticktext': list(sentiment_map.keys()), 
                'tickvals': list(sentiment_map.values()),
                'tickangle': 45  # Angle the labels for better readability
            },
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [1, 2], 'color': "darkred"},
                {'range': [2, 3], 'color': "red"},
                {'range': [3, 4], 'color': "orange"},
                {'range': [4, 5], 'color': "yellow"},
                {'range': [5, 6], 'color': "lightgreen"},
                {'range': [6, 7], 'color': "darkgreen"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        },
        title = {'text': "Market Sentiment"}
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=30, r=30, t=30, b=30)
    )
    
    return fig

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
    
    # Add pricing constraints
    pricing_constraints = {
        'studios': {
            'sample_size': 2,
            'size_difference': 'Comparables are 15-20% smaller than target studio size',
            'location_quality': 'Both comparables are in superior locations',
            'confidence': 'Low - Limited comparable set'
        },
        'one_bed': {
            'sample_size': 12,
            'size_difference': 'Comparable sizes within 5% of target',
            'location_quality': 'Mix of similar and slightly superior locations',
            'confidence': 'High - Good comparable set'
        },
        'two_bed': {
            'sample_size': 8,
            'size_difference': 'Comparable sizes within 8% of target',
            'location_quality': 'Similar locations',
            'confidence': 'Medium-High - Decent comparable set'
        },
        'three_bed': {
            'sample_size': 3,
            'size_difference': 'Comparables 10% larger on average',
            'location_quality': 'Mix of similar and inferior locations',
            'confidence': 'Medium-Low - Limited comparable set'
        }
    }
    
    # Add constraints to base pricing data
    for unit_type in base_pricing:
        base_pricing[unit_type]['constraints'] = pricing_constraints[unit_type]
    
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

def main():
    st.title("Surrey Market Analysis Dashboard")
    
    # Load data with correct structure
    with open('data/processed/surrey_project_data.json', 'r') as f:
        project_data = json.load(f)['content']['project_data']
    with open('data/processed/surrey_macro_data.json', 'r') as f:
        macro_data = json.load(f)['content']['macro_indicators']
    
    analyzer = PricingScenarioAnalyzer(project_data, macro_data)
    
    # Base Analysis Section
    st.subheader("Base Market Analysis")
    if st.button("Generate Base Surrey Market Analysis"):
        run_staged_analysis()
    
    # Scenario Analysis Section
    st.subheader("Scenario Analysis")
    st.write("Use this section to analyze different market scenarios and see their impact on pricing and absorption.")
    
    # Initialize session state for scenario parameters if not exists
    if 'interest_rate_change' not in st.session_state:
        st.session_state.interest_rate_change = 0
    if 'supply_change' not in st.session_state:
        st.session_state.supply_change = 0
    if 'employment_change' not in st.session_state:
        st.session_state.employment_change = 0
    if 'competitor_pricing_change' not in st.session_state:
        st.session_state.competitor_pricing_change = 0
    
    # Text input for natural language query
    query = st.text_input(
        "Ask about market scenarios (e.g., 'What happens if interest rates decrease by 50 bps?')",
        key="scenario_query"
    )
    
    # Process text query if entered
    if query:
        scenario = process_query_with_gpt(query)
        if scenario:
            # Update session state with extracted parameters
            for param, value in scenario.items():
                if param in st.session_state:
                    st.session_state[param] = float(value)
    
    # Scenario Parameter Sliders
    st.subheader("Scenario Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        interest_rate_change = st.slider(
            "Interest Rate Change (bps)",
            min_value=-200.0,
            max_value=200.0,
            value=float(st.session_state.interest_rate_change),
            step=25.0,
            help="Change in 5-year fixed mortgage rates in basis points"
        )
        
        employment_change = st.slider(
            "Employment Change (%)",
            min_value=-5.0,
            max_value=5.0,
            value=float(st.session_state.employment_change),
            step=0.5,
            help="Change in employment rate"
        )
    
    with col2:
        supply_change = st.slider(
            "Supply Change (%)",
            min_value=-20.0,
            max_value=20.0,
            value=float(st.session_state.supply_change),
            step=5.0,
            help="Change in market supply"
        )
        
        competitor_pricing_change = st.slider(
            "Competitor Pricing Change (%)",
            min_value=-10.0,
            max_value=10.0,
            value=float(st.session_state.competitor_pricing_change),
            step=0.5,
            help="Change in competitor project pricing"
        )
    
    # Update session state from sliders
    st.session_state.interest_rate_change = float(interest_rate_change)
    st.session_state.supply_change = float(supply_change)
    st.session_state.employment_change = float(employment_change)
    st.session_state.competitor_pricing_change = float(competitor_pricing_change)
    
    # Create scenario from current parameters
    current_scenario = {
        'interest_rate_change': interest_rate_change,
        'supply_change': supply_change,
        'employment_change': employment_change,
        'competitor_pricing_change': competitor_pricing_change
    }
    
    # Only proceed with analysis if any parameter is non-zero
    if any(value != 0 for value in current_scenario.values()):
        # Run analysis
        results = analyzer.analyze_scenario(current_scenario)
        
        # Calculate weighted change
        weighted_change = sum(
            impact['percent_change'] * mix['count'] / 376  # Total units: 376
            for (unit_type, impact), (_, mix) in zip(
                results['unit_impacts'].items(),
                {'studios': {'count': 34}, 'one_bed': {'count': 204}, 
                 'two_bed': {'count': 120}, 'three_bed': {'count': 18}}.items()
            )
        )   
        
        # Layout for key metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Absorption Impact
            absorption_change = results['market_conditions']['absorption_impact']
            base_absorption = 5.4  # Current monthly absorption rate
            new_absorption = base_absorption * (1 + absorption_change/100)
            
            st.metric(
                "Monthly Absorption Rate",
                f"{new_absorption:.1f}%",
                f"{absorption_change:+.1f}%",
                help="Base absorption: 5.4% monthly"
            )
        
        with col2:
            # Price Impact
            st.metric(
                "Average Price Change",
                f"{weighted_change:+.1f}%",
                help="Weighted average price change across all unit types"
            )
        
        with col3:
            # Market Sentiment Gauge
            sentiment_fig = create_sentiment_gauge(results['market_conditions']['market_sentiment'])
            st.plotly_chart(sentiment_fig, use_container_width=True)
        
        # Show calculated PSFs
        st.markdown("### Calculated PSFs")
        psf_df = pd.DataFrame([
            {
                'Unit Type': ut.title(),
                'Current PSF': f"${impact['base_psf']:.2f}",
                'New PSF': f"${impact['scenario_psf']:.2f}",
                'Change': f"{impact['percent_change']:+.1f}%",
                'Sample Size': impact['constraints']['sample_size'],
                'Avg Size Diff %': f"{impact['constraints']['avg_size_diff_pct']:+.1f}%",
                'Confidence': impact['constraints']['confidence']
            }
            for ut, impact in results['unit_impacts'].items()
        ])
        st.dataframe(psf_df)
        
        # Calculate and show revenue impact
        st.markdown("### Revenue Impact Analysis")
        
        # Calculate base case revenue
        base_psfs = {ut: impact['base_psf'] for ut, impact in results['unit_impacts'].items()}
        base_revenue, base_by_type = calculate_total_revenue(base_psfs, False)
        
        # Calculate scenario revenue
        scenario_revenue, scenario_by_type = calculate_total_revenue(results['unit_impacts'], True)
        
        # Create total revenue comparison table
        st.subheader("Total Revenue Impact")
        revenue_comparison = []
        for period in ['3_Month', '12_Month', '24_Month', '36_Month']:
            base = base_revenue[period]
            scenario = scenario_revenue[period]
            change = scenario - base
            change_pct = (change / base) * 100 if base > 0 else 0
            
            revenue_comparison.append({
                'Period': period.replace('_', ' '),
                'Base Revenue': f"${base:,.0f}",
                'Scenario Revenue': f"${scenario:,.0f}",
                'Change ($)': f"${change:,.0f}",
                'Change (%)': f"{change_pct:+.1f}%"
            })
        
        revenue_df = pd.DataFrame(revenue_comparison)
        st.dataframe(revenue_df)
        
        # Create revenue comparison by unit type
        st.subheader("Revenue Impact by Unit Type")
        
        for period in ['3_Month', '12_Month', '24_Month', '36_Month']:
            st.markdown(f"#### {period.replace('_', ' ')} Revenue")
            unit_type_comparison = []
            
            for unit_type in ['studios', 'one_bed', 'two_bed', 'three_bed']:
                base = base_by_type[unit_type][period]
                scenario = scenario_by_type[unit_type][period]
                change = scenario - base
                change_pct = (change / base) * 100 if base > 0 else 0
                
                unit_type_comparison.append({
                    'Unit Type': unit_type.replace('_', ' ').title(),
                    'Base Revenue': f"${base:,.0f}",
                    'Scenario Revenue': f"${scenario:,.0f}",
                    'Change ($)': f"${change:,.0f}",
                    'Change (%)': f"{change_pct:+.1f}%",
                    'Share of Total': f"{(scenario / scenario_revenue[period] * 100):.1f}%"
                })
            
            unit_type_df = pd.DataFrame(unit_type_comparison)
            st.dataframe(unit_type_df)
        
        # Generate new Excel with scenario analysis
        if st.button("Generate Scenario Analysis"):
            try:
                filename = generate_revenue_tables(results)
                with open(filename, 'rb') as f:
                    st.download_button(
                        label="Download Scenario Analysis",
                        data=f,
                        file_name=filename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="scenario_download"
                    )
                # Clean up file after download button is created
                os.remove(filename)
                st.success("Scenario analysis generated successfully!")
            except Exception as e:
                st.error(f"Error generating scenario analysis: {str(e)}")
    else:
        st.info("Adjust the scenario parameters above to see their impact on pricing and revenue.")

if __name__ == "__main__":
    main() 