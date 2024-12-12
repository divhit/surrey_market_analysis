# Surrey Market Pricing Strategy

## Executive Summary
Our pricing strategy is built on comprehensive market data analysis, incorporating historical trends, current market conditions, and forward-looking indicators. The model uses statistically derived sensitivities from actual market data to ensure pricing accuracy while maximizing absorption potential.

## Core Components

### 1. Base Price Calculation
- **Market Comparables**: Weighted analysis of recent comparable sales
- **Time Weighting**: 6-month half-life decay for sales data relevancy
- **Volume Weighting**: Higher weight given to projects with strong absorption

### 2. Size-Based Adjustments
Elasticity factors derived from historical market data:

| Unit Type | Size Elasticity | Rationale |
|-----------|----------------|------------|
| Studios | -0.35 | Higher sensitivity due to investment focus, derived from historical premium decay |
| One Beds | -0.15 | Standard elasticity, matches historical market behavior |
| Two Beds | -0.12 | Slightly reduced elasticity due to end-user focus |
| Three Beds | -0.12 | Matches two-bed elasticity for consistency |

### 3. Market Impact Factors

#### Employment Impact (40% Weight)
- Historical correlation: 0.72
- Price sensitivity: 0.8% per 1% employment change
- Unit-specific multipliers:
  - Studios/One Beds: 1.1x (young buyer sensitivity)
  - Two/Three Beds: 1.0x (standard sensitivity)

#### Interest Rate Impact (30% Weight)
- Historical correlation: -0.42
- Price sensitivity: -2.5% per 1% rate increase
- Unit-specific multipliers:
  - Studios: 1.2x (investment sensitivity)
  - One Beds: 1.0x
  - Two Beds: 1.0x
  - Three Beds: 0.9x (equity-driven buyers)

#### Supply Impact (30% Weight)
- Based on current and pipeline inventory levels
- Absorption-correlated pricing adjustments
- Shared equally across unit types

## Current Market Positioning

### Target PSF by Unit Type
| Unit Type | Target PSF | Size (sf) | Key Drivers |
|-----------|------------|-----------|-------------|
| Studios | $1,317.55 | 379 | Premium for efficiency, size-adjusted |
| One Beds | $1,148.47 | 474 | Market benchmark pricing |
| Two Beds | $1,062.08 | 795 | End-user focused pricing |
| Three Beds | $1,039.00 | 941 | Family-oriented pricing |

### Competitive Analysis
Key competitors and their positioning:
- Manhattan: Premium positioning, 7.5% initial premium
- Parkway 2: Market leader, 2.0% initial premium

## Market Validation

### Historical Correlations
Our pricing model is validated against historical market data:
- Employment to price correlation: 0.72
- Interest rate to price correlation: -0.42
- Income to price correlation: 0.78

### Absorption Analysis
Current market absorption rates support our pricing strategy:
- Studios: 3.1% monthly (9.0% market share)
- One Beds: 3.2% monthly (54.3% market share)
- Two Beds: 3.3% monthly (31.9% market share)
- Three Beds: 3.0% monthly (4.8% market share)

## Price Evolution Strategy

### Premium Decay
- Initial premium structure based on project positioning
- Controlled premium decay to maintain absorption
- Unit-type specific adjustment paths

### Market Response Mechanisms
Built-in adjustments for:
- Employment changes
- Interest rate movements
- Supply pipeline changes
- Absorption rate variations

## Implementation Guidelines

### Launch Phase
1. Initial pricing reflects current market strength
2. Premium positioning for studios and one-beds
3. Competitive positioning for larger units

### Monitoring & Adjustment
- Monthly absorption rate tracking
- Quarterly pricing review
- Market condition impact assessment
- Competitor pricing analysis

### Risk Management
- Built-in elasticity factors for market changes
- Supply-sensitive pricing adjustments
- Unit-type specific sensitivity factors

## Conclusion
Our pricing strategy combines rigorous data analysis with market-proven sensitivities to optimize both pricing and absorption. The model's foundation in historical data while maintaining flexibility for market conditions ensures sustainable sales velocity while maximizing revenue potential. 