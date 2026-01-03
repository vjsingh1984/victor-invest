"""
Monte Carlo Simulation Module

Provides probabilistic price forecasting using various stochastic models.
Foundation for portfolio risk analysis and scenario planning.
"""
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class MonteCarloResult:
    """Results from Monte Carlo simulation"""
    symbol: str
    current_price: float
    simulations: int
    time_horizon_days: int
    model_type: str

    # Summary statistics
    mean_price: float
    median_price: float
    std_dev: float

    # Percentiles
    percentile_5: float   # Black swan scenario
    percentile_25: float  # Bear case
    percentile_50: float  # Base case
    percentile_75: float  # Bull case
    percentile_95: float  # Optimistic scenario

    # Risk metrics
    var_95: float         # Value at Risk (95% confidence)
    cvar_95: float        # Conditional VaR (Expected Shortfall)
    max_drawdown: float   # Maximum loss from peak
    probability_above_current: float
    probability_profit: float

    # Full distribution for charting
    distribution: np.ndarray

    # Metadata
    volatility_annual: float
    drift_annual: float
    simulation_date: datetime


class MonteCarloSimulator:
    """Monte Carlo simulation engine for price forecasting"""

    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize Monte Carlo simulator

        Args:
            random_seed: Optional random seed for reproducibility
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        self.random_seed = random_seed
        logger.info(f"Initialized MonteCarloSimulator (seed={random_seed})")

    def simulate_geometric_brownian_motion(
        self,
        symbol: str,
        current_price: float,
        volatility_annual: float,
        drift_annual: float = 0.0,
        time_horizon_days: int = 252,  # 1 year
        simulations: int = 10000
    ) -> MonteCarloResult:
        """
        Run Monte Carlo simulation using Geometric Brownian Motion

        GBM Model: S(t) = S(0) * exp((μ - σ²/2)t + σW(t))
        Where:
        - S(t) = price at time t
        - μ = drift (expected return)
        - σ = volatility
        - W(t) = Wiener process (random walk)

        Args:
            symbol: Stock symbol
            current_price: Current stock price
            volatility_annual: Annual volatility (standard deviation)
            drift_annual: Annual drift (expected return)
            time_horizon_days: Days to simulate
            simulations: Number of simulation paths

        Returns:
            MonteCarloResult with statistics and distribution
        """
        logger.info(f"Running GBM simulation for {symbol}: "
                   f"{simulations} paths, {time_horizon_days} days")

        # Convert annual parameters to daily
        dt = 1 / 252  # Daily time step (252 trading days/year)
        volatility_daily = volatility_annual / np.sqrt(252)
        drift_daily = drift_annual / 252

        # Calculate number of steps
        num_steps = time_horizon_days

        # Generate random walks
        # Shape: (simulations, num_steps)
        random_shocks = np.random.standard_normal((simulations, num_steps))

        # Calculate price paths using GBM
        # S(t) = S(0) * exp((μ - σ²/2)t + σW(t))
        # For each daily step: (drift - 0.5*vol²)*dt + vol*sqrt(dt)*dW
        drift_term = (drift_daily - 0.5 * volatility_daily ** 2)
        diffusion_term = volatility_daily * random_shocks

        # Daily log returns
        log_returns = drift_term + diffusion_term

        # Cumulative sum of returns
        cumulative_returns = np.cumsum(log_returns, axis=1)

        # Final prices for each simulation
        final_prices = current_price * np.exp(cumulative_returns[:, -1])

        # Calculate statistics
        result = self._calculate_statistics(
            symbol=symbol,
            current_price=current_price,
            final_prices=final_prices,
            simulations=simulations,
            time_horizon_days=time_horizon_days,
            model_type='GBM',
            volatility_annual=volatility_annual,
            drift_annual=drift_annual
        )

        logger.info(f"GBM simulation complete for {symbol}: "
                   f"Mean=${result.mean_price:.2f}, "
                   f"P(profit)={result.probability_profit:.1%}")

        return result

    def simulate_from_historical_data(
        self,
        symbol: str,
        current_price: float,
        historical_returns: np.ndarray,
        time_horizon_days: int = 252,
        simulations: int = 10000
    ) -> MonteCarloResult:
        """
        Run Monte Carlo simulation using historical returns distribution

        Samples from actual historical returns (non-parametric approach)

        Args:
            symbol: Stock symbol
            current_price: Current stock price
            historical_returns: Array of historical daily returns
            time_horizon_days: Days to simulate
            simulations: Number of simulation paths

        Returns:
            MonteCarloResult with statistics and distribution
        """
        logger.info(f"Running historical simulation for {symbol}: "
                   f"{simulations} paths, {len(historical_returns)} historical returns")

        # Calculate volatility and drift from historical data
        volatility_daily = np.std(historical_returns)
        volatility_annual = volatility_daily * np.sqrt(252)
        drift_daily = np.mean(historical_returns)
        drift_annual = drift_daily * 252

        # Randomly sample returns with replacement (bootstrap)
        sampled_returns = np.random.choice(
            historical_returns,
            size=(simulations, time_horizon_days),
            replace=True
        )

        # Calculate cumulative returns
        cumulative_returns = np.sum(sampled_returns, axis=1)

        # Final prices
        final_prices = current_price * np.exp(cumulative_returns)

        # Calculate statistics
        result = self._calculate_statistics(
            symbol=symbol,
            current_price=current_price,
            final_prices=final_prices,
            simulations=simulations,
            time_horizon_days=time_horizon_days,
            model_type='Historical',
            volatility_annual=volatility_annual,
            drift_annual=drift_annual
        )

        logger.info(f"Historical simulation complete for {symbol}")

        return result

    def _calculate_statistics(
        self,
        symbol: str,
        current_price: float,
        final_prices: np.ndarray,
        simulations: int,
        time_horizon_days: int,
        model_type: str,
        volatility_annual: float,
        drift_annual: float
    ) -> MonteCarloResult:
        """
        Calculate statistics from simulation results

        Args:
            symbol: Stock symbol
            current_price: Current price
            final_prices: Array of final prices from simulations
            simulations: Number of simulations
            time_horizon_days: Time horizon
            model_type: Type of model used
            volatility_annual: Annual volatility
            drift_annual: Annual drift

        Returns:
            MonteCarloResult with all statistics
        """
        # Sort prices for percentile calculations
        sorted_prices = np.sort(final_prices)

        # Summary statistics
        mean_price = float(np.mean(final_prices))
        median_price = float(np.median(final_prices))
        std_dev = float(np.std(final_prices))

        # Percentiles
        percentile_5 = float(np.percentile(sorted_prices, 5))
        percentile_25 = float(np.percentile(sorted_prices, 25))
        percentile_50 = float(np.percentile(sorted_prices, 50))
        percentile_75 = float(np.percentile(sorted_prices, 75))
        percentile_95 = float(np.percentile(sorted_prices, 95))

        # Risk metrics

        # Value at Risk (VaR 95%): Maximum loss with 95% confidence
        # VaR = Current Price - 5th percentile price
        var_95 = float(current_price - percentile_5)

        # Conditional VaR (CVaR, Expected Shortfall)
        # Average loss in worst 5% of cases
        worst_5_percent = sorted_prices[:int(0.05 * simulations)]
        cvar_95 = float(current_price - np.mean(worst_5_percent)) if len(worst_5_percent) > 0 else var_95

        # Maximum drawdown (simplified - worst outcome)
        max_drawdown = float((percentile_5 - current_price) / current_price * 100)

        # Probabilities
        probability_above_current = float(np.sum(final_prices > current_price) / simulations)
        probability_profit = probability_above_current  # Same for now

        return MonteCarloResult(
            symbol=symbol,
            current_price=current_price,
            simulations=simulations,
            time_horizon_days=time_horizon_days,
            model_type=model_type,
            mean_price=mean_price,
            median_price=median_price,
            std_dev=std_dev,
            percentile_5=percentile_5,
            percentile_25=percentile_25,
            percentile_50=percentile_50,
            percentile_75=percentile_75,
            percentile_95=percentile_95,
            var_95=var_95,
            cvar_95=cvar_95,
            max_drawdown=max_drawdown,
            probability_above_current=probability_above_current,
            probability_profit=probability_profit,
            distribution=final_prices,
            volatility_annual=volatility_annual,
            drift_annual=drift_annual,
            simulation_date=datetime.now()
        )

    def generate_scenarios(
        self,
        symbol: str,
        current_price: float,
        volatility_annual: float,
        drift_annual: float = 0.0,
        time_horizon_days: int = 252,
        simulations: int = 10000
    ) -> Dict[str, Dict]:
        """
        Generate standard investment scenarios

        Returns:
            Dictionary with bull/base/bear/black_swan scenarios
        """
        result = self.simulate_geometric_brownian_motion(
            symbol=symbol,
            current_price=current_price,
            volatility_annual=volatility_annual,
            drift_annual=drift_annual,
            time_horizon_days=time_horizon_days,
            simulations=simulations
        )

        return {
            'black_swan': {
                'scenario': 'Black Swan Event',
                'probability': 5.0,
                'price_target': result.percentile_5,
                'return_pct': (result.percentile_5 - current_price) / current_price * 100,
                'description': 'Worst 5% of outcomes'
            },
            'bear': {
                'scenario': 'Bear Case',
                'probability': 25.0,
                'price_target': result.percentile_25,
                'return_pct': (result.percentile_25 - current_price) / current_price * 100,
                'description': 'Below-average performance'
            },
            'base': {
                'scenario': 'Base Case',
                'probability': 50.0,
                'price_target': result.percentile_50,
                'return_pct': (result.percentile_50 - current_price) / current_price * 100,
                'description': 'Most likely outcome'
            },
            'bull': {
                'scenario': 'Bull Case',
                'probability': 75.0,
                'price_target': result.percentile_75,
                'return_pct': (result.percentile_75 - current_price) / current_price * 100,
                'description': 'Above-average performance'
            },
            'optimistic': {
                'scenario': 'Optimistic Case',
                'probability': 95.0,
                'price_target': result.percentile_95,
                'return_pct': (result.percentile_95 - current_price) / current_price * 100,
                'description': 'Best 5% of outcomes'
            }
        }


def calculate_volatility_from_returns(returns: np.ndarray, annual: bool = True) -> float:
    """
    Calculate volatility (standard deviation) from returns

    Args:
        returns: Array of returns
        annual: If True, annualize the volatility

    Returns:
        Volatility (standard deviation)
    """
    volatility = float(np.std(returns))

    if annual:
        volatility *= np.sqrt(252)  # Annualize assuming 252 trading days

    return volatility


def calculate_historical_drift(returns: np.ndarray, annual: bool = True) -> float:
    """
    Calculate historical drift (mean return) from returns

    Args:
        returns: Array of returns
        annual: If True, annualize the drift

    Returns:
        Drift (mean return)
    """
    drift = float(np.mean(returns))

    if annual:
        drift *= 252  # Annualize

    return drift
