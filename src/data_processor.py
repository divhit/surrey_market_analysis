# src/data_processor.py

import pandas as pd
import numpy as np
from datetime import datetime
import json
from typing import Dict, List, Any

class DataProcessor:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.processed_data = {
            'project_data': {},
            'macro_data': {}
        }
        
        # Column mappings for different data files
        self.launch_columns = {
            'name': 'Unnamed: 0',  # Sales Start - Past 5Years
            'developer': 'Unnamed: 1',  # Developer
            'address': 'Doubtful to start construction',  # Address
            'total_units': 'Unnamed: 3',  # Total Units
            'units_sold': 'Unnamed: 4',  # Units Sold
            'sales_start': 'Unnamed: 5',  # Sales Start
            'completion': 'Unnamed: 6',  # Completion
            'status': 'Unnamed: 7',  # Q4 notes - we'll need to derive status from this
            'construction_status': 'Doubtful to start construction'  # We'll derive this from the same column as address
        }
        
        self.pricing_columns = {
            'source_url': 'source_url',
            'beds': 'beds',
            'sqft': 'sqft',
            'price': 'price',
            'psf': 'psf'
        }

    def _print_dataframe_info(self, df: pd.DataFrame, file_name: str) -> None:
        """Print DataFrame column information for debugging"""
        print(f"\nColumns in {file_name}:")
        for col in df.columns:
            print(f"  - {col}")
            # Print first non-null value as example
            first_val = df[col].dropna().iloc[0] if not df[col].empty else 'No data'
            print(f"    Example value: {first_val}")

    def process_all_data(self, completions_file: str, pricing_file: str, rates_file: str) -> Dict:
        """
        Process all input data files
        
        Args:
            completions_file: Path to completions CSV
            pricing_file: Path to pricing CSV
            rates_file: Path to rates CSV
            
        Returns:
            Dict containing all processed data
        """
        # Process project data
        self.process_project_data()
        
        # Process macro data
        self.process_macro_data()
        
        # Save processed data
        self.save_processed_data()
        
        return self.processed_data

    def process_project_data(self) -> Dict:
        """Process project level data"""
        try:
            # Read project data
            launches_df = pd.read_csv(f"{self.data_dir}/raw/Surrey_Concrete_Launches_correct.csv", skiprows=1)  # Skip header row
            pricing_df = pd.read_csv(f"{self.data_dir}/raw/Surrey_pricing.csv")
            
            # Print column information for debugging
            self._print_dataframe_info(launches_df, "Surrey_Concrete_Launches_correct.csv")
            self._print_dataframe_info(pricing_df, "Surrey_pricing.csv")
            
            # Clean and preprocess launches data
            launches_df = self._preprocess_launches_data(launches_df)
            
            # Map columns to expected names
            launches_df = launches_df.rename(columns={v: k for k, v in self.launch_columns.items()})
            
            # Derive status and construction status
            launches_df['status'] = launches_df['status'].apply(self._derive_project_status)
            launches_df['construction_status'] = launches_df['address'].apply(self._derive_construction_status)
            
            # Process active projects
            active_projects = self._process_active_projects(launches_df, pricing_df)
            
            # Process sold out projects
            sold_out_projects = self._process_sold_out_projects(launches_df)
            
            # Calculate market metrics
            market_metrics = self._calculate_market_metrics(launches_df)
            
            self.processed_data['project_data'] = {
                'active_projects': active_projects,
                'sold_out_projects': sold_out_projects,
                'market_metrics': market_metrics
            }
            
            return self.processed_data['project_data']
            
        except FileNotFoundError as e:
            print(f"Error: Could not find data file - {str(e)}")
            raise
        except pd.errors.EmptyDataError:
            print("Error: One or more data files are empty")
            raise
        except Exception as e:
            print(f"Error processing project data: {str(e)}")
            raise

    def process_macro_data(self) -> Dict:
        """Process macro economic and market data"""
        # Read macro data
        completions_df = pd.read_csv(f"{self.data_dir}/raw/Condo_completions_Surrey.csv")
        starts_df = pd.read_csv(f"{self.data_dir}/raw/Condo_starts_Surrey.csv")
        income_df = pd.read_csv(f"{self.data_dir}/raw/census_household_income.csv")
        rates_df = pd.read_csv(f"{self.data_dir}/raw/discounted_mortgage_rates.csv")
        
        # Process each component
        supply_metrics = self._process_supply_data(completions_df, starts_df)
        demographic_metrics = self._process_demographic_data(income_df)
        interest_rate_trends = self._process_rate_data(rates_df)
        
        self.processed_data['macro_data'] = {
            'supply_metrics': supply_metrics,
            'demographic_metrics': demographic_metrics,
            'interest_rate_trends': interest_rate_trends
        }
        
        return self.processed_data['macro_data']

    def _process_active_projects(self, launches_df: pd.DataFrame, pricing_df: pd.DataFrame) -> Dict:
        """Process active projects data"""
        try:
            # Ensure status column exists and has expected values
            if 'status' not in launches_df.columns:
                # Try to derive status from other columns if possible
                if 'project_status' in launches_df.columns:
                    launches_df['status'] = launches_df['project_status']
                else:
                    # Create a default status based on units_sold
                    launches_df['status'] = launches_df.apply(
                        lambda row: 'active' if row['units_sold'] < row['total_units'] else 'sold_out', 
                        axis=1
                    )
            
            active_projects = launches_df[launches_df['status'].str.lower() == 'active'].copy()
            
            if len(active_projects) == 0:
                print("Warning: No active projects found in the data")
                return {
                    'total_count': 0,
                    'total_units': 0,
                    'total_sold': 0,
                    'current_absorption': 0,
                    'projects': []
                }
            
            projects_data = []
            for _, project in active_projects.iterrows():
                try:
                    project_pricing = pricing_df[
                        pricing_df['source_url'].str.contains(str(project['name']), 
                                                            na=False, 
                                                            case=False)
                    ]
                    
                    project_data = {
                        'name': project['name'],
                        'developer': project.get('developer', 'Unknown'),
                        'address': project.get('address', 'Unknown'),
                        'total_units': int(project['total_units']),
                        'units_sold': int(project['units_sold']),
                        'sales_start': project.get('sales_start', None),
                        'completion': project.get('completion', None),
                        'status': project['status'],
                        'current_absorption': (
                            float(project['units_sold']) / float(project['total_units']) * 100
                            if float(project['total_units']) > 0 else 0
                        ),
                        'construction_status': project.get('construction_status', 'Unknown'),
                        'pricing': self._process_project_pricing(project_pricing)
                    }
                    projects_data.append(project_data)
                    
                except Exception as e:
                    print(f"Warning: Error processing project {project.get('name', 'Unknown')}: {str(e)}")
                    continue
            
            return {
                'total_count': len(projects_data),
                'total_units': sum(p['total_units'] for p in projects_data),
                'total_sold': sum(p['units_sold'] for p in projects_data),
                'current_absorption': (
                    sum(p['units_sold'] for p in projects_data) / 
                    sum(p['total_units'] for p in projects_data) * 100
                    if sum(p['total_units'] for p in projects_data) > 0 else 0
                ),
                'projects': projects_data
            }
            
        except Exception as e:
            print(f"Error processing active projects: {str(e)}")
            raise

    def _process_project_pricing(self, pricing_df: pd.DataFrame) -> Dict:
        """Process pricing data for a project"""
        pricing_data = {}
        for bed_type in pricing_df['beds'].unique():
            if pd.isna(bed_type):
                continue
                
            type_data = pricing_df[pricing_df['beds'] == bed_type]
            pricing_data[f"{int(bed_type)}bed"] = {
                'size_range': [type_data['sqft'].min(), type_data['sqft'].max()],
                'price_range': [
                    type_data['price'].str.replace('$', '').str.replace(',', '').astype(float).min(),
                    type_data['price'].str.replace('$', '').str.replace(',', '').astype(float).max()
                ],
                'avg_psf': type_data['psf'].astype(float).mean(),
                'available_units': len(type_data)
            }
            
        return pricing_data

    def save_processed_data(self):
        """Save processed data to JSON files"""
        # Save project data
        with open(f"{self.data_dir}/processed/surrey_project_data.json", 'w') as f:
            json.dump(self.processed_data['project_data'], f, indent=2)
            
        # Save macro data
        with open(f"{self.data_dir}/processed/surrey_macro_data.json", 'w') as f:
            json.dump(self.processed_data['macro_data'], f, indent=2)

    def _preprocess_launches_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess launches data to clean and standardize"""
        # Remove any completely empty rows
        df = df.dropna(how='all')
        
        # Convert numeric columns
        numeric_cols = ['Unnamed: 3', 'Unnamed: 4']  # total_units and units_sold
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert date columns
        date_cols = ['Unnamed: 5', 'Unnamed: 6']  # sales_start and completion
        for col in date_cols:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            
        return df

    def _derive_project_status(self, notes: str) -> str:
        """Derive project status from Q4 notes"""
        if pd.isna(notes):
            return 'unknown'
        
        notes = str(notes).lower()
        if 'sold out' in notes:
            return 'sold_out'
        elif 'complete' in notes:
            return 'completed'
        elif 'construction' in notes:
            return 'under_construction'
        elif 'sales' in notes:
            return 'active'
        else:
            return 'active'  # Default to active if status unclear

    def _derive_construction_status(self, status: str) -> str:
        """Derive construction status from the status column"""
        if pd.isna(status):
            return 'unknown'
        
        status = str(status).lower()
        if 'construction' in status:
            return 'under_construction'
        elif 'complete' in status:
            return 'completed'
        elif 'doubtful' in status:
            return 'pre_construction'
        else:
            return 'unknown'