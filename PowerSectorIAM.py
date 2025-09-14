Python 3.13.7 (v3.13.7:bcee1c32211, Aug 14 2025, 19:10:51) [Clang 16.0.0 (clang-1600.0.26.6)] on darwin
Enter "help" below or click "Help" above for more information.
>>> import numpy as np
... import pandas as pd
... from typing import Dict, List, Tuple, Optional
... import matplotlib.pyplot as plt
... import seaborn as sns
... from scipy.stats import truncnorm, uniform, lognorm
... import warnings
... warnings.filterwarnings('ignore')
... 
... class PowerSectorIAM:
...     def __init__(self, tech_list: List[str], region_list: List[str], 
...                  base_year: int, end_year: int, time_step: int = 1):
...         """
...         Integrated Assessment Model specialized for the power sector
...         
...         Parameters:
...             tech_list: List of power technologies (e.g., ['solar_pv', 'wind_onshore', 'coal', 'gas_ccgt'])
...             region_list: List of regions
...             base_year: Base year for simulation
...             end_year: End year for simulation
...             time_step: Time step (years)
...         """
...         self.tech_list = tech_list
...         self.region_list = region_list
...         self.years = list(range(base_year, end_year + 1, time_step))
...         self.current_year = base_year
...         self.time_step = time_step
...         
...         # Initialize data structures
...         self.initialize_data_structures()
...         
...         # Set default distributions
...         self.uncertainty_distributions = self.set_default_distributions()
...         
...     def initialize_data_structures(self):
...         """Initialize all data structures"""
        n_techs = len(self.tech_list)
        n_regions = len(self.region_list)
        n_years = len(self.years)
        
        # Power technology specific parameters
        self.capacity_factors = np.zeros((n_techs, n_regions, n_years))  # Capacity factors
        self.efficiency = np.zeros((n_techs, n_regions, n_years))  # Generation efficiency (for fuel-based plants)
        
        # Cost components: [Investment($/kW), O&M($/MWh), Fuel($/MWh), Carbon cost($/MWh)]
        self.costs = np.zeros((n_techs, n_regions, n_years, 4))
        
        # Capacity: [Cumulative capacity(MW), New capacity(MW), Maximum potential(MW)]
        self.capacity = np.zeros((n_techs, n_regions, n_years, 3))
        
        # Electricity generation (GWh)
        self.generation = np.zeros((n_techs, n_regions, n_years))
        
        # Learning parameters: [Learning rate, Initial cumulative capacity, Initial cost]
        self.learning_params = np.zeros((n_techs, 3))
        
        # Spillover matrices: Technology spillover [from_tech, to_tech]
        self.spillover_tech = np.zeros((n_techs, n_techs))
        
        # Regional spillover matrix [from_region, to_region, tech]
        self.spillover_region = np.zeros((n_regions, n_regions, n_techs))
        
        # Electricity demand (GWh)
        self.electricity_demand = np.zeros((n_regions, n_years))
        
        # Create mapping dictionaries for easy access
        self.tech_map = {tech: i for i, tech in enumerate(self.tech_list)}
        self.region_map = {region: i for i, region in enumerate(self.region_list)}
        self.year_map = {year: i for i, year in enumerate(self.years)}
        
    def set_default_distributions(self) -> Dict:
        """Set uncertainty distributions for parameters"""
        distributions = {
            # Policy-related parameters
            'carbon_tax_start': {'dist': 'uniform', 'params': [40, 60]},  # Initial carbon price in 2025 ($/ton)
            'carbon_tax_growth': {'dist': 'uniform', 'params': [0.03, 0.07]},  # Carbon price annual growth rate
            'renewable_subsidy': {'dist': 'uniform', 'params': [0.05, 0.15]},  # Renewable energy subsidy rate
            
            # Socio-economic parameters
            'gdp_growth': {'dist': 'normal', 'params': [0.03, 0.01]},  # GDP annual growth rate (mean, std)
            'demand_response': {'dist': 'uniform', 'params': [0.8, 1.2]},  # Electricity demand elasticity multiplier
            
            # Technology parameters
            'solar_learning_rate': {'dist': 'normal', 'params': [0.20, 0.02]},  # Solar learning rate
            'wind_learning_rate': {'dist': 'normal', 'params': [0.15, 0.02]},  # Wind learning rate
            'battery_cost_reduction': {'dist': 'uniform', 'params': [0.15, 0.25]},  # Battery cost annual reduction rate
            'solar_capacity_factor': {'dist': 'normal', 'params': [0.18, 0.02]},  # Solar capacity factor
            'gas_price': {'dist': 'lognorm', 'params': [0.1, 0.05]}  # Natural gas price volatility (s, scale)
        }
        return distributions
    
    def sample_parameters(self, n_samples: int = 1) -> List[Dict]:
        """Sample parameter values from distributions"""
        samples = []
        
        for _ in range(n_samples):
            sample = {}
            
            for param, dist_info in self.uncertainty_distributions.items():
                dist_type = dist_info['dist']
                params = dist_info['params']
                
                if dist_type == 'uniform':
                    sample[param] = np.random.uniform(params[0], params[1])
                elif dist_type == 'normal':
                    sample[param] = np.random.normal(params[0], params[1])
                elif dist_type == 'lognorm':
                    sample[param] = np.random.lognormal(params[0], params[1])
                elif dist_type == 'truncnorm':
                    # Truncated normal distribution: params = [mean, sd, lower, upper]
                    a = (params[2] - params[0]) / params[1] if len(params) > 2 else -np.inf
                    b = (params[3] - params[0]) / params[1] if len(params) > 3 else np.inf
                    sample[param] = truncnorm.rvs(a, b, loc=params[0], scale=params[1])
            
            samples.append(sample)
        
        return samples
    
    def set_initial_conditions(self, initial_conditions: Dict):
        """
        Set initial conditions for the power sector
        
        Parameters:
            initial_conditions: Dictionary containing all initial conditions
        """
        # Set technology-specific parameters
        for tech, params in initial_conditions['tech_params'].items():
            if tech in self.tech_map:
                t_idx = self.tech_map[tech]
                
                for region, values in params.items():
                    if region in self.region_map:
                        r_idx = self.region_map[region]
                        year_idx = self.year_map[self.current_year]
                        
                        # Set capacity factor
                        if 'capacity_factor' in values:
                            self.capacity_factors[t_idx, r_idx, year_idx] = values['capacity_factor']
                        
                        # Set efficiency (for fuel-based plants)
                        if 'efficiency' in values:
                            self.efficiency[t_idx, r_idx, year_idx] = values['efficiency']
        
        # Set cost data
        for tech, regions_data in initial_conditions['costs'].items():
            t_idx = self.tech_map[tech]
            self.learning_params[t_idx, 0] = initial_conditions['learning_rates'][tech]  # Learning rate
            
            for region, cost_data in regions_data.items():
                r_idx = self.region_map[region]
                year_idx = self.year_map[self.current_year]
                
                # Set initial costs
                self.costs[t_idx, r_idx, year_idx] = cost_data
                
                # Set initial cumulative capacity and maximum potential
                self.capacity[t_idx, r_idx, year_idx, 0] = initial_conditions['initial_capacity'][tech][region]
                self.capacity[t_idx, r_idx, year_idx, 2] = initial_conditions['max_potential'][tech][region]
                
                # Record initial values for learning curve calculation
                if year_idx == 0:  # First year
                    self.learning_params[t_idx, 1] += initial_conditions['initial_capacity'][tech][region]  # Initial cumulative capacity
                    self.learning_params[t_idx, 2] = cost_data[0]  # Initial investment cost
        
        # Set initial electricity demand
        for region, demand in initial_conditions['initial_demand'].items():
            r_idx = self.region_map[region]
            self.electricity_demand[r_idx, 0] = demand
    
    def set_spillover_matrices(self, tech_spillover: np.ndarray, region_spillover: Optional[np.ndarray] = None):
        """
        Set spillover matrices
        
        Parameters:
            tech_spillover: Technology spillover matrix [from_tech, to_tech]
            region_spillover: Regional spillover matrix [from_region, to_region, tech]
        """
        self.spillover_tech = tech_spillover
        
        if region_spillover is not None:
            self.region_spillover = region_spillover
        else:
            # Default regional spillover matrix
            n_regions = len(self.region_list)
            self.spillover_region = np.zeros((n_regions, n_regions, len(self.tech_list)))
            for i in range(n_regions):
                for j in range(n_regions):
                    for k in range(len(self.tech_list)):
                        # Full spillover within same country, partial between countries
                        self.spillover_region[i, j, k] = 1.0 if i == j else 0.3
    
    def set_scenario(self, scenario_params: Dict, sampled_params: Dict):
        """
        Set scenario parameters, integrating sampled uncertainty parameters
        
        Parameters:
            scenario_params: Base scenario parameters
            sampled_params: Parameter values sampled from distributions
        """
        # Merge base scenario and sampled parameters
        self.scenario = {**scenario_params, **sampled_params}
    
    def calculate_lcoe(self, tech_idx: int, region_idx: int, year_idx: int, 
                       discount_rate: float = 0.05) -> float:
        """
        Calculate Levelized Cost of Electricity (LCOE) for power technologies
        
        Parameters:
            tech_idx: Technology index
            region_idx: Region index
            year_idx: Year index
            discount_rate: Discount rate
            
        Returns:
            LCOE value ($/MWh)
        """
        # Get cost data
        cost_inv, cost_om, cost_fuel, cost_carbon = self.costs[tech_idx, region_idx, year_idx]
        
        # Get technology parameters
        capacity_factor = self.capacity_factors[tech_idx, region_idx, year_idx]
        tech_efficiency = self.efficiency[tech_idx, region_idx, year_idx] if self.efficiency[tech_idx, region_idx, year_idx] > 0 else 1.0
        
        # Get capacity data
        capacity = self.capacity[tech_idx, region_idx, year_idx, 0]
        if capacity == 0:
            return float('inf')
        
        # Power sector specific LCOE calculation
        # Annualized investment cost ($/kW/year)
        lifetime = 25  # Power plant lifetime
        annualized_inv = cost_inv * discount_rate / (1 - (1 + discount_rate) ** -lifetime)
        
        # Annual electricity generation (MWh/kW)
        hours_per_year = 8760
        generation_per_kw = capacity_factor * hours_per_year
        
        # Investment cost component ($/MWh)
        inv_per_mwh = annualized_inv / generation_per_kw
        
        # Fuel cost component (considering efficiency)
        fuel_per_mwh = cost_fuel / tech_efficiency if tech_efficiency > 0 else 0
        
        # Total LCOE
        lcoe = inv_per_mwh + cost_om + fuel_per_mwh + cost_carbon
        
        return lcoe
    
    def update_costs_with_learning(self, year_idx: int):
        """Update costs based on learning curves (considering technology and regional spillovers)"""
        prev_year_idx = year_idx - 1 if year_idx > 0 else 0
        
        for t_idx in range(len(self.tech_list)):
            # Calculate global cumulative capacity change (considering regional spillover)
            global_cap_change = 0
            for r_idx in range(len(self.region_list)):
                cap_change = self.capacity[t_idx, r_idx, year_idx, 1]  # New capacity
                
                # Apply regional spillover
                for r2_idx in range(len(self.region_list)):
                    global_cap_change += cap_change * self.spillover_region[r_idx, r2_idx, t_idx]
            
            # Calculate technology spillover effects
            tech_spillover_effect = 0
            for t2_idx in range(len(self.tech_list)):
                if t_idx != t2_idx:
                    # Spillover from other technologies' global cumulative capacity changes
                    tech_cap_change = 0
                    for r_idx in range(len(self.region_list)):
                        tech_cap_change += self.capacity[t2_idx, r_idx, year_idx, 1]
                    
                    tech_spillover_effect += tech_cap_change * self.spillover_tech[t2_idx, t_idx]
            
            # Total effective learning
            total_learning = global_cap_change + tech_spillover_effect
            
            # Apply learning curve (affects only investment and O&M costs)
            if total_learning > 0 and self.learning_params[t_idx, 1] > 0:
                learning_rate = self.learning_params[t_idx, 0]
                
                # Calculate cost reduction ratio
                cumulative_ratio = (self.learning_params[t_idx, 1] + total_learning) / self.learning_params[t_idx, 1]
                cost_reduction_factor = cumulative_ratio ** (-learning_rate)
                
                # Apply technology scenario learning rate multiplier (if available)
                tech_name = self.tech_list[t_idx]
                if 'solar_learning_rate' in self.scenario and 'solar' in tech_name:
                    actual_learning_rate = self.scenario['solar_learning_rate']
                    cost_reduction_factor = cumulative_ratio ** (-actual_learning_rate)
                elif 'wind_learning_rate' in self.scenario and 'wind' in tech_name:
                    actual_learning_rate = self.scenario['wind_learning_rate']
                    cost_reduction_factor = cumulative_ratio ** (-actual_learning_rate)
                
                # Update costs for all regions
                for r_idx in range(len(self.region_list)):
                    # Investment and O&M costs affected by learning
                    self.costs[t_idx, r_idx, year_idx, 0] = self.costs[t_idx, r_idx, prev_year_idx, 0] * cost_reduction_factor
                    self.costs[t_idx, r_idx, year_idx, 1] = self.costs[t_idx, r_idx, prev_year_idx, 1] * cost_reduction_factor
                    
                    # Fuel and carbon costs not affected by learning, but by policy and markets
                    self.costs[t_idx, r_idx, year_idx, 2] = self.costs[t_idx, r_idx, prev_year_idx, 2]
                    self.costs[t_idx, r_idx, year_idx, 3] = self.costs[t_idx, r_idx, prev_year_idx, 3]
            
            # Update cumulative capacity
            self.learning_params[t_idx, 1] += total_learning
    
    def apply_policy_scenarios(self, year_idx: int):
        """Apply policy scenarios, considering uncertainty"""
        year = self.years[year_idx]
        
        # Carbon pricing policy (considering uncertainty)
        if 'carbon_tax_start' in self.scenario and 'carbon_tax_growth' in self.scenario:
            if 'carbon_tax_start_year' in self.scenario:
                start_year = self.scenario['carbon_tax_start_year']
            else:
                start_year = 2025  # Default start year
                
            if year >= start_year:
                start_value = self.scenario['carbon_tax_start']
                growth_rate = self.scenario['carbon_tax_growth']
                years_since_start = year - start_year
                current_carbon_tax = start_value * (1 + growth_rate) ** years_since_start
                
                # Apply carbon price to all emitting technologies
                for t_idx, tech in enumerate(self.tech_list):
                    if any(fossil in tech for fossil in ['coal', 'gas', 'oil']):  # Fossil fuel technologies
                        for r_idx in range(len(self.region_list)):
                            # Carbon cost = Carbon intensity * Carbon price
                            # Assumed carbon intensities: coal ~ 0.8-1.0 tCO2/MWh, natural gas ~ 0.4-0.5 tCO2/MWh
                            if 'coal' in tech:
                                carbon_intensity = 0.9
                            elif 'gas' in tech:
                                carbon_intensity = 0.45
                            else:
                                carbon_intensity = 0.7  # Default value
                                
                            self.costs[t_idx, r_idx, year_idx, 3] = carbon_intensity * current_carbon_tax
        
        # Subsidy policies (considering uncertainty)
        if 'renewable_subsidy' in self.scenario:
            subsidy_rate = self.scenario['renewable_subsidy']
            
            for t_idx, tech in enumerate(self.tech_list):
                if any(renewable in tech for renewable in ['solar', 'wind', 'hydro', 'geothermal']):
                    for r_idx in range(len(self.region_list)):
                        # Reduce investment cost
                        self.costs[t_idx, r_idx, year_idx, 0] *= (1 - subsidy_rate)
        
        # Fuel price uncertainty
        if 'gas_price' in self.scenario:
            gas_price_multiplier = self.scenario['gas_price']
            for t_idx, tech in enumerate(self.tech_list):
                if 'gas' in tech:
                    for r_idx in range(len(self.region_list)):
                        # Adjust natural gas price
                        self.costs[t_idx, r_idx, year_idx, 2] *= gas_price_multiplier
    
    def update_electricity_demand(self, year_idx: int):
        """Update electricity demand, considering GDP growth and uncertainty"""
        if year_idx == 0:
            return  # First year already has initial demand set
        
        prev_year_idx = year_idx - 1
        
        for r_idx, region in enumerate(self.region_list):
            # Base demand growth (related to GDP growth)
            if 'gdp_growth' in self.scenario:
                gdp_growth = self.scenario['gdp_growth']
            else:
                gdp_growth = 0.03  # Default GDP growth rate
                
            # Demand elasticity (considering uncertainty)
            if 'demand_response' in self.scenario:
                elasticity = self.scenario['demand_response']
            else:
                elasticity = 1.0  # Default no elasticity change
                
            # Calculate demand growth
            prev_demand = self.electricity_demand[r_idx, prev_year_idx]
            demand_growth = prev_demand * gdp_growth * elasticity * self.time_step
            
            # Update demand
            self.electricity_demand[r_idx, year_idx] = prev_demand + demand_growth
    
    def calculate_investment(self, year_idx: int) -> np.ndarray:
        """
        Calculate investment allocation based on technology competitiveness and demand
        
        Returns:
            Investment allocation matrix [tech, region]
        """
        # Calculate demand increase
        demand_increase = np.zeros(len(self.region_list))
        for r_idx in range(len(self.region_list)):
            if year_idx > 0:
                demand_increase[r_idx] = self.electricity_demand[r_idx, year_idx] - self.electricity_demand[r_idx, year_idx-1]
            else:
                demand_increase[r_idx] = self.electricity_demand[r_idx, 0] * 0.05  # Default 5% growth
        
        # Calculate technology competitiveness (based on LCOE)
        competitiveness = np.zeros((len(self.tech_list), len(self.region_list)))
        for t_idx in range(len(self.tech_list)):
            for r_idx in range(len(self.region_list)):
                lcoe = self.calculate_lcoe(t_idx, r_idx, year_idx)
                competitiveness[t_idx, r_idx] = 1 / lcoe  # Lower LCOE means higher competitiveness
        
        # Consider non-monetary factors (social acceptance)
        social_acceptance = self.scenario.get('social_acceptance', {})
        for tech, multiplier in social_acceptance.items():
            if tech in self.tech_map:
                t_idx = self.tech_map[tech]
                competitiveness[t_idx, :] *= multiplier
        
        # Normalize competitiveness scores
        competitiveness_norm = competitiveness / np.sum(competitiveness, axis=0)
        
        # Calculate investment allocation
        investment = np.zeros((len(self.tech_list), len(self.region_list)))
        for r_idx in range(len(self.region_list)):
            # Total investment proportional to demand growth (assume $2M investment per GWh demand increase)
            investment_intensity = 2.0  # Million dollars/GWh
            total_investment = demand_increase[r_idx] * investment_intensity
            
            # Allocate investment based on competitiveness
            investment[:, r_idx] = competitiveness_norm[:, r_idx] * total_investment
        
        return investment
    
    def update_capacity_from_investment(self, investment: np.ndarray, year_idx: int):
        """Update capacity based on investment"""
        prev_year_idx = year_idx - 1 if year_idx > 0 else 0
        
        for t_idx in range(len(self.tech_list)):
            for r_idx in range(len(self.region_list)):
                # Calculate new capacity: Investment amount / Unit investment cost
                inv_cost = self.costs[t_idx, r_idx, year_idx, 0]
                if inv_cost > 0:
                    new_capacity = investment[t_idx, r_idx] * 1000 / inv_cost  # Investment in million dollars, cost in $/kW
                    
                    # Consider maximum potential constraint
                    max_potential = self.capacity[t_idx, r_idx, year_idx, 2]
                    existing_capacity = self.capacity[t_idx, r_idx, prev_year_idx, 0]
                    
                    if existing_capacity + new_capacity > max_potential:
                        new_capacity = max(0, max_potential - existing_capacity)
                    
                    # Update capacity
                    self.capacity[t_idx, r_idx, year_idx, 0] = existing_capacity + new_capacity
                    self.capacity[t_idx, r_idx, year_idx, 1] = new_capacity  # Record new capacity
    
    def calculate_generation(self, year_idx: int):
        """Calculate electricity generation"""
        for t_idx in range(len(self.tech_list)):
            for r_idx in range(len(self.region_list)):
                capacity = self.capacity[t_idx, r_idx, year_idx, 0]
                capacity_factor = self.capacity_factors[t_idx, r_idx, year_idx]
                
                # Generation = Capacity * Capacity factor * Hours
                self.generation[t_idx, r_idx, year_idx] = capacity * capacity_factor * 8760 / 1000  # Convert to GWh
    
    def run_simulation(self) -> Dict:
        """Run simulation"""
        results = {
            'lcoe': np.zeros((len(self.tech_list), len(self.region_list), len(self.years))),
            'capacity': np.zeros((len(self.tech_list), len(self.region_list), len(self.years))),
            'generation': np.zeros((len(self.tech_list), len(self.region_list), len(self.years))),
            'investment': np.zeros((len(self.tech_list), len(self.region_list), len(self.years))),
            'demand': np.zeros((len(self.region_list), len(self.years)))
        }
        
        for year_idx, year in enumerate(self.years):
            self.current_year = year
            
            # Skip year 0 (initial conditions)
            if year_idx == 0:
                # Calculate initial LCOE and generation
                for t_idx in range(len(self.tech_list)):
                    for r_idx in range(len(self.region_list)):
                        results['lcoe'][t_idx, r_idx, year_idx] = self.calculate_lcoe(t_idx, r_idx, year_idx)
                self.calculate_generation(year_idx)
                results['generation'][:, :, year_idx] = self.generation[:, :, year_idx]
                results['capacity'][:, :, year_idx] = self.capacity[:, :, year_idx, 0]
                results['demand'][:, year_idx] = self.electricity_demand[:, year_idx]
                continue
            
            # 1. Update electricity demand
            self.update_electricity_demand(year_idx)
            results['demand'][:, year_idx] = self.electricity_demand[:, year_idx]
            
            # 2. Apply policy scenarios
            self.apply_policy_scenarios(year_idx)
            
            # 3. Calculate investment allocation
            investment = self.calculate_investment(year_idx)
            results['investment'][:, :, year_idx] = investment
            
            # 4. Update capacity
            self.update_capacity_from_investment(investment, year_idx)
            results['capacity'][:, :, year_idx] = self.capacity[:, :, year_idx, 0]
            
            # 5. Apply learning effects to update costs
            self.update_costs_with_learning(year_idx)
            
            # 6. Calculate LCOE
            for t_idx in range(len(self.tech_list)):
                for r_idx in range(len(self.region_list)):
                    results['lcoe'][t_idx, r_idx, year_idx] = self.calculate_lcoe(
                        t_idx, r_idx, year_idx)
            
            # 7. Calculate generation
            self.calculate_generation(year_idx)
            results['generation'][:, :, year_idx] = self.generation[:, :, year_idx]
        
        return results
    
    def check_tipping_point(self, results: Dict, threshold: float = 0.5) -> Dict:
        """
        Check for renewable energy dominance tipping point
        
        Parameters:
            results: Simulation results
            threshold: Tipping point threshold (renewable share of generation)
            
        Returns:
            Tipping point information
        """
        tipping_points = {}
        
        for r_idx, region in enumerate(self.region_list):
            # Calculate renewable generation share
            renewable_share = np.zeros(len(self.years))
            for t_idx, tech in enumerate(self.tech_list):
                if any(renewable in tech for renewable in ['solar', 'wind', 'hydro', 'geothermal', 'biomass']):
                    renewable_share += results['generation'][t_idx, r_idx, :]
            
            total_generation = np.sum(results['generation'][:, r_idx, :], axis=0)
            renewable_share = renewable_share / total_generation
            
            # Find the year when threshold is exceeded
            tipping_year = None
            for year_idx, share in enumerate(renewable_share):
                if share >= threshold and tipping_year is None:
                    tipping_year = self.years[year_idx]
                    break
            
            tipping_points[region] = {
                'year': tipping_year,
                'share_trajectory': renewable_share,
                'final_share': renewable_share[-1] if len(renewable_share) > 0 else 0
            }
        
        return tipping_points

    def run_monte_carlo(self, base_scenario: Dict, n_samples: int = 100) -> Dict:
        """
        Run Monte Carlo simulation considering parameter uncertainty
        
        Parameters:
            base_scenario: Base scenario parameters
            n_samples: Number of samples
            
        Returns:
            Monte Carlo simulation results
        """
        # Sample parameters
        sampled_params_list = self.sample_parameters(n_samples)
        
        # Store all simulation results
        all_results = {
            'lcoe': [],
            'capacity': [],
            'generation': [],
            'investment': [],
            'demand': [],
            'tipping_points': []
        }
        
        # Run multiple simulations
        for i, sampled_params in enumerate(sampled_params_list):
            print(f"Running simulation {i+1}/{n_samples}")
            
            # Reset model state
            self.initialize_data_structures()
            self.set_initial_conditions(base_scenario['initial_conditions'])
            self.set_spillover_matrices(base_scenario['spillover_matrices']['tech'])
            
            # Set scenario (base scenario + sampled parameters)
            self.set_scenario(base_scenario['scenario_params'], sampled_params)
            
            # Run simulation
            results = self.run_simulation()
            
            # Check tipping points
            tipping_points = self.check_tipping_point(results)
            
            # Store results
            for key in ['lcoe', 'capacity', 'generation', 'investment', 'demand']:
                all_results[key].append(results[key])
            all_results['tipping_points'].append(tipping_points)
        
        return all_results
    
    def analyze_monte_carlo_results(self, results: Dict) -> Dict:
        """
        Analyze Monte Carlo simulation results
        
        Parameters:
            results: Monte Carlo simulation results
            
        Returns:
            Analysis results
        """
        analysis = {}
        
        # Analyze tipping point years
        tipping_years = {}
        for region in self.region_list:
            tipping_years[region] = []
        
        for tipping_points in results['tipping_points']:
            for region, data in tipping_points.items():
                if data['year'] is not None:
                    tipping_years[region].append(data['year'])
        
        # Calculate statistics
        analysis['tipping_point_stats'] = {}
        for region, years in tipping_years.items():
            if years:
                analysis['tipping_point_stats'][region] = {
                    'mean': np.mean(years),
                    'median': np.median(years),
                    'std': np.std(years),
                    '5th_percentile': np.percentile(years, 5),
                    '95th_percentile': np.percentile(years, 95)
                }
        
        # Analyze final renewable share
        final_shares = {}
        for region in self.region_list:
            final_shares[region] = []
        
        for tipping_points in results['tipping_points']:
            for region, data in tipping_points.items():
                final_shares[region].append(data['final_share'])
        
        analysis['final_share_stats'] = {}
        for region, shares in final_shares.items():
            if shares:
                analysis['final_share_stats'][region] = {
                    'mean': np.mean(shares),
                    'median': np.median(shares),
                    'std': np.std(shares),
                    '5th_percentile': np.percentile(shares, 5),
                    '95th_percentile': np.percentile(shares, 95)
                }
        
        return analysis

# ===================== Example Usage =====================
if __name__ == "__main__":
    # Initialize model (power sector specific)
    power_techs = ['solar_pv', 'wind_onshore', 'coal', 'gas_ccgt', 'nuclear', 'hydro']
    regions = ['EU', 'US', 'China']
    model = PowerSectorIAM(power_techs, regions, 2020, 2060, 5)
    
    # Set initial conditions
    initial_conditions = {
        'tech_params': {
            'solar_pv': {
                'EU': {'capacity_factor': 0.15},
                'US': {'capacity_factor': 0.18},
                'China': {'capacity_factor': 0.16}
            },
            'wind_onshore': {
                'EU': {'capacity_factor': 0.25},
                'US': {'capacity_factor': 0.28},
                'China': {'capacity_factor': 0.22}
            },
            'coal': {
                'EU': {'capacity_factor': 0.60, 'efficiency': 0.38},
                'US': {'capacity_factor': 0.65, 'efficiency': 0.40},
                'China': {'capacity_factor': 0.55, 'efficiency': 0.35}
            },
            'gas_ccgt': {
                'EU': {'capacity_factor': 0.55, 'efficiency': 0.50},
                'US': {'capacity_factor': 0.60, 'efficiency': 0.52},
                'China': {'capacity_factor': 0.50, 'efficiency': 0.48}
            },
            'nuclear': {
                'EU': {'capacity_factor': 0.85},
                'US': {'capacity_factor': 0.90},
                'China': {'capacity_factor': 0.80}
            },
            'hydro': {
                'EU': {'capacity_factor': 0.45},
                'US': {'capacity_factor': 0.40},
                'China': {'capacity_factor': 0.35}
            }
        },
        'costs': {
            'solar_pv': {
                'EU': [800, 15, 0, 0],
                'US': [750, 14, 0, 0],
                'China': [600, 12, 0, 0]
            },
            'wind_onshore': {
                'EU': [1100, 25, 0, 0],
                'US': [1000, 22, 0, 0],
                'China': [900, 20, 0, 0]
            },
            'coal': {
                'EU': [2000, 30, 40, 20],
                'US': [1800, 28, 35, 15],
                'China': [1500, 25, 30, 10]
            },
            'gas_ccgt': {
                'EU': [1000, 25, 60, 15],
                'US': [900, 22, 55, 12],
                'China': [800, 20, 50, 8]
            },
            'nuclear': {
                'EU': [5000, 40, 0, 0],
                'US': [4500, 35, 0, 0],
                'China': [4000, 30, 0, 0]
            },
            'hydro': {
                'EU': [3000, 20, 0, 0],
                'US': [2800, 18, 0, 0],
                'China': [2500, 15, 0, 0]
            }
        },
        'initial_capacity': {
            'solar_pv': {'EU': 500, 'US': 600, 'China': 800},
            'wind_onshore': {'EU': 700, 'US': 800, 'China': 1000},
            'coal': {'EU': 1500, 'US': 1800, 'China': 2500},
            'gas_ccgt': {'EU': 1000, 'US': 1200, 'China': 1500},
            'nuclear': {'EU': 800, 'US': 900, 'China': 500},
            'hydro': {'EU': 600, 'US': 700, 'China': 1200}
        },
        'max_potential': {
            'solar_pv': {'EU': 8000, 'US': 10000, 'China': 15000},
            'wind_onshore': {'EU': 6000, 'US': 8000, 'China': 12000},
            'coal': {'EU': 3000, 'US': 3500, 'China': 4000},
            'gas_ccgt': {'EU': 2000, 'US': 2500, 'China': 3000},
            'nuclear': {'EU': 1000, 'US': 1200, 'China': 800},
            'hydro': {'EU': 800, 'US': 900, 'China': 1500}
        },
        'learning_rates': {
            'solar_pv': 0.20,
            'wind_onshore': 0.15,
            'coal': 0.01,
            'gas_ccgt': 0.02,
            'nuclear': 0.05,
            'hydro': 0.01
        },
        'initial_demand': {
            'EU': 300000,  # GWh
            'US': 400000,
            'China': 600000
        }
    }
    
    # Set spillover matrices
    n_techs = len(power_techs)
    tech_spillover = np.eye(n_techs)  # Start with identity matrix (no spillover)
    
    # Set spillovers between renewable technologies
    solar_idx = model.tech_map['solar_pv']
    wind_idx = model.tech_map['wind_onshore']
    tech_spillover[solar_idx, wind_idx] = 0.1  # Solar to wind spillover
    tech_spillover[wind_idx, solar_idx] = 0.05  # Wind to solar spillover
    
    # Base scenario parameters
    base_scenario = {
        'initial_conditions': initial_conditions,
        'spillover_matrices': {
            'tech': tech_spillover
        },
        'scenario_params': {
            'carbon_tax_start_year': 2025,
            'social_acceptance': {
                'solar_pv': 1.2,
                'wind_onshore': 1.1,
                'coal': 0.8,
                'nuclear': 0.9
            }
        }
    }
    
    # Run Monte Carlo simulation
    mc_results = model.run_monte_carlo(base_scenario, n_samples=50)
    
    # Analyze results
    analysis = model.analyze_monte_carlo_results(mc_results)
    
    print("Monte Carlo Simulation Results Analysis:")
    print("\nTipping Point Year Statistics (Renewable Share > 50%):")
    for region, stats in analysis['tipping_point_stats'].items():
        print(f"{region}: {stats}")
    
    print("\n2060 Renewable Share Statistics:")
    for region, stats in analysis['final_share_stats'].items():
        print(f"{region}: {stats}")
    
    # Visualize results
    plt.figure(figsize=(15, 10))
    
    # Plot global solar capacity growth with uncertainty
    solar_capacity = np.zeros((len(mc_results['capacity']), len(model.years)))
    for i, sim_result in enumerate(mc_results['capacity']):
        solar_idx = model.tech_map['solar_pv']
        solar_capacity[i, :] = np.sum(sim_result[solar_idx, :, :], axis=0)
    
    # Calculate percentiles
    p5 = np.percentile(solar_capacity, 5, axis=0)
    p50 = np.percentile(solar_capacity, 50, axis=0)
    p95 = np.percentile(solar_capacity, 95, axis=0)
    
    plt.subplot(2, 2, 1)
    plt.fill_between(model.years, p5, p95, alpha=0.3, label='90% Confidence Interval')
    plt.plot(model.years, p50, label='Median', linewidth=2)
    plt.xlabel('Year')
    plt.ylabel('Global Solar Capacity (MW)')
    plt.title('Global Solar Capacity Projection with Uncertainty')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot tipping point trajectories
    plt.subplot(2, 2, 2)
    tipping_data = []
    labels = []
    for region, stats in analysis['tipping_point_stats'].items():
        tipping_data.append(mc_results['tipping_points'][0][region]['share_trajectory'])
        labels.append(region)
    
    for i, data in enumerate(tipping_data):
        plt.plot(model.years, data, label=labels[i], linewidth=2)
    
    plt.axhline(y=0.5, color='r', linestyle='--', label='50% Threshold')
    plt.xlabel('Year')
    plt.ylabel('Renewable Share')
    plt.title('Renewable Share Trajectory')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot LCOE changes
    plt.subplot(2, 2, 3)
    solar_lcoe = np.zeros((len(mc_results['lcoe']), len(model.years)))
    for i, sim_result in enumerate(mc_results['lcoe']):
        solar_idx = model.tech_map['solar_pv']
        solar_lcoe[i, :] = np.mean(sim_result[solar_idx, :, :], axis=0)
    
    p5_lcoe = np.percentile(solar_lcoe, 5, axis=0)
    p50_lcoe = np.percentile(solar_lcoe, 50, axis=0)
    p95_lcoe = np.percentile(solar_lcoe, 95, axis=0)
    
    plt.fill_between(model.years, p5_lcoe, p95_lcoe, alpha=0.3, label='90% Confidence Interval')
    plt.plot(model.years, p50_lcoe, label='Median', linewidth=2)
    plt.xlabel('Year')
    plt.ylabel('Solar LCOE ($/MWh)')
    plt.title('Solar LCOE Projection with Uncertainty')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot tipping point year distribution
    plt.subplot(2, 2, 4)
    tipping_years = []
    region_labels = []
    for region, stats in analysis['tipping_point_stats'].items():
        # Collect all tipping point years from simulations
        years = []
        for tipping_points in mc_results['tipping_points']:
            if tipping_points[region]['year'] is not None:
                years.append(tipping_points[region]['year'])
        
        if years:
            tipping_years.append(years)
            region_labels.append(region)
    
    plt.boxplot(tipping_years, labels=region_labels)
    plt.ylabel('Tipping Point Year')
    plt.title('Distribution of Tipping Point Years')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
