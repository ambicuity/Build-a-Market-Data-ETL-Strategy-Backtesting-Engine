"""Options and futures support for derivatives trading."""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from dataclasses import dataclass
from enum import Enum
from scipy.stats import norm


class OptionType(Enum):
    """Option type enumeration."""
    CALL = "call"
    PUT = "put"


class FuturesType(Enum):
    """Futures contract type."""
    COMMODITY = "commodity"
    FINANCIAL = "financial"
    CURRENCY = "currency"
    INDEX = "index"


@dataclass
class Option:
    """Option contract data structure."""
    symbol: str
    underlying: str
    option_type: OptionType
    strike: float
    expiry: pd.Timestamp
    premium: float = 0
    quantity: int = 0
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.underlying} {self.strike} {self.option_type.value} {self.expiry.date()}"


@dataclass
class FuturesContract:
    """Futures contract data structure."""
    symbol: str
    underlying: str
    contract_type: FuturesType
    expiry: pd.Timestamp
    contract_size: float
    price: float = 0
    quantity: int = 0
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.underlying} Futures {self.expiry.date()}"


class BlackScholesModel:
    """Black-Scholes option pricing model."""
    
    def __init__(self, risk_free_rate: float = 0.02):
        """Initialize Black-Scholes model.
        
        Args:
            risk_free_rate: Risk-free interest rate
        """
        self.risk_free_rate = risk_free_rate
    
    def calculate_d1_d2(
        self,
        spot_price: float,
        strike: float,
        time_to_expiry: float,
        volatility: float
    ) -> tuple:
        """Calculate d1 and d2 for Black-Scholes formula.
        
        Args:
            spot_price: Current price of underlying
            strike: Strike price
            time_to_expiry: Time to expiry in years
            volatility: Implied volatility
            
        Returns:
            Tuple of (d1, d2)
        """
        if time_to_expiry <= 0:
            return 0, 0
        
        d1 = (np.log(spot_price / strike) + 
              (self.risk_free_rate + 0.5 * volatility ** 2) * time_to_expiry) / \
             (volatility * np.sqrt(time_to_expiry))
        
        d2 = d1 - volatility * np.sqrt(time_to_expiry)
        
        return d1, d2
    
    def price_call(
        self,
        spot_price: float,
        strike: float,
        time_to_expiry: float,
        volatility: float
    ) -> float:
        """Price a European call option.
        
        Args:
            spot_price: Current price of underlying
            strike: Strike price
            time_to_expiry: Time to expiry in years
            volatility: Implied volatility
            
        Returns:
            Call option price
        """
        if time_to_expiry <= 0:
            return max(spot_price - strike, 0)
        
        d1, d2 = self.calculate_d1_d2(spot_price, strike, time_to_expiry, volatility)
        
        call_price = (spot_price * norm.cdf(d1) - 
                     strike * np.exp(-self.risk_free_rate * time_to_expiry) * norm.cdf(d2))
        
        return call_price
    
    def price_put(
        self,
        spot_price: float,
        strike: float,
        time_to_expiry: float,
        volatility: float
    ) -> float:
        """Price a European put option.
        
        Args:
            spot_price: Current price of underlying
            strike: Strike price
            time_to_expiry: Time to expiry in years
            volatility: Implied volatility
            
        Returns:
            Put option price
        """
        if time_to_expiry <= 0:
            return max(strike - spot_price, 0)
        
        d1, d2 = self.calculate_d1_d2(spot_price, strike, time_to_expiry, volatility)
        
        put_price = (strike * np.exp(-self.risk_free_rate * time_to_expiry) * norm.cdf(-d2) - 
                    spot_price * norm.cdf(-d1))
        
        return put_price
    
    def calculate_greeks(
        self,
        option_type: OptionType,
        spot_price: float,
        strike: float,
        time_to_expiry: float,
        volatility: float
    ) -> Dict[str, float]:
        """Calculate option Greeks.
        
        Args:
            option_type: Call or put
            spot_price: Current price of underlying
            strike: Strike price
            time_to_expiry: Time to expiry in years
            volatility: Implied volatility
            
        Returns:
            Dictionary of Greeks (delta, gamma, theta, vega, rho)
        """
        if time_to_expiry <= 0:
            return {"delta": 0, "gamma": 0, "theta": 0, "vega": 0, "rho": 0}
        
        d1, d2 = self.calculate_d1_d2(spot_price, strike, time_to_expiry, volatility)
        
        # Delta
        if option_type == OptionType.CALL:
            delta = norm.cdf(d1)
        else:
            delta = -norm.cdf(-d1)
        
        # Gamma (same for call and put)
        gamma = norm.pdf(d1) / (spot_price * volatility * np.sqrt(time_to_expiry))
        
        # Vega (same for call and put)
        vega = spot_price * norm.pdf(d1) * np.sqrt(time_to_expiry) / 100
        
        # Theta
        if option_type == OptionType.CALL:
            theta = (-spot_price * norm.pdf(d1) * volatility / (2 * np.sqrt(time_to_expiry)) -
                    self.risk_free_rate * strike * np.exp(-self.risk_free_rate * time_to_expiry) * norm.cdf(d2)) / 365
        else:
            theta = (-spot_price * norm.pdf(d1) * volatility / (2 * np.sqrt(time_to_expiry)) +
                    self.risk_free_rate * strike * np.exp(-self.risk_free_rate * time_to_expiry) * norm.cdf(-d2)) / 365
        
        # Rho
        if option_type == OptionType.CALL:
            rho = strike * time_to_expiry * np.exp(-self.risk_free_rate * time_to_expiry) * norm.cdf(d2) / 100
        else:
            rho = -strike * time_to_expiry * np.exp(-self.risk_free_rate * time_to_expiry) * norm.cdf(-d2) / 100
        
        return {
            "delta": delta,
            "gamma": gamma,
            "theta": theta,
            "vega": vega,
            "rho": rho
        }


class OptionStrategy:
    """Common option trading strategies."""
    
    def __init__(self, model: BlackScholesModel):
        """Initialize option strategy.
        
        Args:
            model: Black-Scholes pricing model
        """
        self.model = model
    
    def covered_call(
        self,
        spot_price: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        shares: int = 100
    ) -> Dict[str, float]:
        """Calculate covered call strategy P&L.
        
        Args:
            spot_price: Current stock price
            strike: Call strike price
            time_to_expiry: Time to expiry
            volatility: Implied volatility
            shares: Number of shares owned
            
        Returns:
            Dictionary with strategy details
        """
        call_premium = self.model.price_call(spot_price, strike, time_to_expiry, volatility)
        
        return {
            "stock_value": spot_price * shares,
            "call_premium_received": call_premium * shares,
            "max_profit": (strike - spot_price) * shares + call_premium * shares,
            "breakeven": spot_price - call_premium,
            "max_loss": spot_price * shares - call_premium * shares
        }
    
    def protective_put(
        self,
        spot_price: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        shares: int = 100
    ) -> Dict[str, float]:
        """Calculate protective put strategy P&L.
        
        Args:
            spot_price: Current stock price
            strike: Put strike price
            time_to_expiry: Time to expiry
            volatility: Implied volatility
            shares: Number of shares owned
            
        Returns:
            Dictionary with strategy details
        """
        put_premium = self.model.price_put(spot_price, strike, time_to_expiry, volatility)
        
        return {
            "stock_value": spot_price * shares,
            "put_premium_paid": put_premium * shares,
            "max_loss": (spot_price - strike) * shares + put_premium * shares,
            "breakeven": spot_price + put_premium,
            "max_profit": float('inf')  # Unlimited upside
        }
    
    def straddle(
        self,
        spot_price: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        contracts: int = 1
    ) -> Dict[str, float]:
        """Calculate long straddle strategy.
        
        Args:
            spot_price: Current stock price
            strike: Strike price (same for call and put)
            time_to_expiry: Time to expiry
            volatility: Implied volatility
            contracts: Number of contracts
            
        Returns:
            Dictionary with strategy details
        """
        call_premium = self.model.price_call(spot_price, strike, time_to_expiry, volatility)
        put_premium = self.model.price_put(spot_price, strike, time_to_expiry, volatility)
        
        total_premium = (call_premium + put_premium) * contracts * 100
        
        return {
            "call_premium": call_premium * contracts * 100,
            "put_premium": put_premium * contracts * 100,
            "total_cost": total_premium,
            "upper_breakeven": strike + (call_premium + put_premium),
            "lower_breakeven": strike - (call_premium + put_premium),
            "max_loss": total_premium
        }
    
    def iron_condor(
        self,
        spot_price: float,
        lower_put_strike: float,
        upper_put_strike: float,
        lower_call_strike: float,
        upper_call_strike: float,
        time_to_expiry: float,
        volatility: float,
        contracts: int = 1
    ) -> Dict[str, float]:
        """Calculate iron condor strategy.
        
        Args:
            spot_price: Current stock price
            lower_put_strike: Lower put strike (buy)
            upper_put_strike: Upper put strike (sell)
            lower_call_strike: Lower call strike (sell)
            upper_call_strike: Upper call strike (buy)
            time_to_expiry: Time to expiry
            volatility: Implied volatility
            contracts: Number of contracts
            
        Returns:
            Dictionary with strategy details
        """
        # Buy lower put
        lower_put = self.model.price_put(spot_price, lower_put_strike, time_to_expiry, volatility)
        # Sell upper put
        upper_put = self.model.price_put(spot_price, upper_put_strike, time_to_expiry, volatility)
        # Sell lower call
        lower_call = self.model.price_call(spot_price, lower_call_strike, time_to_expiry, volatility)
        # Buy upper call
        upper_call = self.model.price_call(spot_price, upper_call_strike, time_to_expiry, volatility)
        
        net_credit = (-lower_put + upper_put + lower_call - upper_call) * contracts * 100
        max_loss = ((upper_put_strike - lower_put_strike) - net_credit / (contracts * 100)) * contracts * 100
        
        return {
            "net_credit": net_credit,
            "max_profit": net_credit,
            "max_loss": max_loss,
            "lower_breakeven": upper_put_strike - net_credit / (contracts * 100),
            "upper_breakeven": lower_call_strike + net_credit / (contracts * 100)
        }


class FuturesCalculator:
    """Futures contract calculations."""
    
    def __init__(self, margin_rate: float = 0.10):
        """Initialize futures calculator.
        
        Args:
            margin_rate: Initial margin requirement as fraction
        """
        self.margin_rate = margin_rate
    
    def calculate_margin(
        self,
        contract: FuturesContract,
        quantity: int
    ) -> float:
        """Calculate margin requirement.
        
        Args:
            contract: Futures contract
            quantity: Number of contracts
            
        Returns:
            Margin requirement
        """
        contract_value = contract.price * contract.contract_size * abs(quantity)
        return contract_value * self.margin_rate
    
    def calculate_pnl(
        self,
        entry_price: float,
        current_price: float,
        contract_size: float,
        quantity: int
    ) -> float:
        """Calculate futures P&L.
        
        Args:
            entry_price: Entry price
            current_price: Current price
            contract_size: Contract size (multiplier)
            quantity: Number of contracts (positive for long, negative for short)
            
        Returns:
            Profit/Loss
        """
        price_change = current_price - entry_price
        return price_change * contract_size * quantity
    
    def calculate_basis(
        self,
        spot_price: float,
        futures_price: float
    ) -> float:
        """Calculate basis (spot - futures).
        
        Args:
            spot_price: Spot price
            futures_price: Futures price
            
        Returns:
            Basis
        """
        return spot_price - futures_price
    
    def calculate_carry_cost(
        self,
        spot_price: float,
        risk_free_rate: float,
        dividend_yield: float,
        time_to_expiry: float
    ) -> float:
        """Calculate cost of carry for futures.
        
        Args:
            spot_price: Spot price
            risk_free_rate: Risk-free rate
            dividend_yield: Dividend yield
            time_to_expiry: Time to expiry in years
            
        Returns:
            Fair futures price
        """
        return spot_price * np.exp((risk_free_rate - dividend_yield) * time_to_expiry)


class DerivativesPortfolio:
    """Portfolio manager for derivatives."""
    
    def __init__(self, initial_cash: float = 1_000_000):
        """Initialize derivatives portfolio.
        
        Args:
            initial_cash: Initial cash balance
        """
        self.cash = initial_cash
        self.options: List[Option] = []
        self.futures: List[FuturesContract] = []
        self.pricing_model = BlackScholesModel()
    
    def add_option(
        self,
        option: Option,
        quantity: int,
        premium: float
    ) -> bool:
        """Add option position to portfolio.
        
        Args:
            option: Option contract
            quantity: Number of contracts (positive for buy, negative for sell)
            premium: Premium paid/received per contract
            
        Returns:
            True if successful
        """
        cost = premium * abs(quantity) * 100  # Options are typically for 100 shares
        
        if quantity > 0 and self.cash < cost:
            return False
        
        option.quantity = quantity
        option.premium = premium
        self.options.append(option)
        
        # Update cash
        if quantity > 0:
            self.cash -= cost
        else:
            self.cash += cost
        
        return True
    
    def add_futures(
        self,
        contract: FuturesContract,
        quantity: int,
        margin_required: float
    ) -> bool:
        """Add futures position to portfolio.
        
        Args:
            contract: Futures contract
            quantity: Number of contracts
            margin_required: Margin requirement
            
        Returns:
            True if successful
        """
        if self.cash < margin_required:
            return False
        
        contract.quantity = quantity
        self.futures.append(contract)
        self.cash -= margin_required
        
        return True
    
    def calculate_portfolio_value(
        self,
        underlying_prices: Dict[str, float],
        volatilities: Dict[str, float],
        current_time: pd.Timestamp
    ) -> float:
        """Calculate total portfolio value.
        
        Args:
            underlying_prices: Current prices of underlying assets
            volatilities: Current implied volatilities
            current_time: Current timestamp
            
        Returns:
            Total portfolio value
        """
        total_value = self.cash
        
        # Value options
        for option in self.options:
            spot_price = underlying_prices.get(option.underlying, 0)
            volatility = volatilities.get(option.underlying, 0.20)
            time_to_expiry = (option.expiry - current_time).total_seconds() / (365 * 24 * 3600)
            
            if time_to_expiry > 0:
                if option.option_type == OptionType.CALL:
                    current_premium = self.pricing_model.price_call(
                        spot_price, option.strike, time_to_expiry, volatility
                    )
                else:
                    current_premium = self.pricing_model.price_put(
                        spot_price, option.strike, time_to_expiry, volatility
                    )
                
                # P&L from option position
                total_value += (current_premium - option.premium) * option.quantity * 100
        
        # Value futures
        for contract in self.futures:
            current_price = underlying_prices.get(contract.underlying, contract.price)
            pnl = (current_price - contract.price) * contract.contract_size * contract.quantity
            total_value += pnl
        
        return total_value
