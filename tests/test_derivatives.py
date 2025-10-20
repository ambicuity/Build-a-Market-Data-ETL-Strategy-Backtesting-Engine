"""Tests for derivatives module (options and futures)."""

import pytest
import pandas as pd
import numpy as np
from backtesting.derivatives import (
    BlackScholesModel, OptionStrategy, FuturesCalculator,
    DerivativesPortfolio, Option, FuturesContract,
    OptionType, FuturesType
)


class TestBlackScholesModel:
    """Test Black-Scholes option pricing model."""
    
    def test_model_init(self):
        """Test model initialization."""
        model = BlackScholesModel(risk_free_rate=0.02)
        assert model.risk_free_rate == 0.02
    
    def test_calculate_d1_d2(self):
        """Test d1 and d2 calculation."""
        model = BlackScholesModel(risk_free_rate=0.02)
        
        spot_price = 100.0
        strike = 100.0
        time_to_expiry = 1.0
        volatility = 0.20
        
        d1, d2 = model.calculate_d1_d2(spot_price, strike, time_to_expiry, volatility)
        
        assert isinstance(d1, float)
        assert isinstance(d2, float)
        assert d1 > d2  # d1 is always greater than d2
    
    def test_price_call_atm(self):
        """Test pricing an at-the-money call option."""
        model = BlackScholesModel(risk_free_rate=0.02)
        
        spot_price = 100.0
        strike = 100.0
        time_to_expiry = 1.0
        volatility = 0.20
        
        call_price = model.price_call(spot_price, strike, time_to_expiry, volatility)
        
        assert isinstance(call_price, float)
        assert call_price > 0
        # ATM call should be worth roughly 0.5 * spot * volatility * sqrt(time)
        assert 5.0 < call_price < 15.0
    
    def test_price_call_itm(self):
        """Test pricing an in-the-money call option."""
        model = BlackScholesModel(risk_free_rate=0.02)
        
        spot_price = 110.0
        strike = 100.0
        time_to_expiry = 1.0
        volatility = 0.20
        
        call_price = model.price_call(spot_price, strike, time_to_expiry, volatility)
        
        # ITM call should be worth at least intrinsic value
        intrinsic_value = spot_price - strike
        assert call_price >= intrinsic_value
    
    def test_price_put_atm(self):
        """Test pricing an at-the-money put option."""
        model = BlackScholesModel(risk_free_rate=0.02)
        
        spot_price = 100.0
        strike = 100.0
        time_to_expiry = 1.0
        volatility = 0.20
        
        put_price = model.price_put(spot_price, strike, time_to_expiry, volatility)
        
        assert isinstance(put_price, float)
        assert put_price > 0
    
    def test_put_call_parity(self):
        """Test put-call parity relationship."""
        model = BlackScholesModel(risk_free_rate=0.02)
        
        spot_price = 100.0
        strike = 100.0
        time_to_expiry = 1.0
        volatility = 0.20
        
        call_price = model.price_call(spot_price, strike, time_to_expiry, volatility)
        put_price = model.price_put(spot_price, strike, time_to_expiry, volatility)
        
        # Put-Call Parity: C - P = S - K*e^(-r*T)
        pv_strike = strike * np.exp(-model.risk_free_rate * time_to_expiry)
        parity_diff = call_price - put_price - (spot_price - pv_strike)
        
        assert abs(parity_diff) < 0.01  # Should be very close
    
    def test_calculate_greeks_call(self):
        """Test Greeks calculation for call option."""
        model = BlackScholesModel(risk_free_rate=0.02)
        
        greeks = model.calculate_greeks(
            option_type=OptionType.CALL,
            spot_price=100.0,
            strike=100.0,
            time_to_expiry=1.0,
            volatility=0.20
        )
        
        assert 'delta' in greeks
        assert 'gamma' in greeks
        assert 'theta' in greeks
        assert 'vega' in greeks
        assert 'rho' in greeks
        
        # For ATM call, delta should be around 0.5
        assert 0.4 < greeks['delta'] < 0.6
        
        # Gamma should be positive
        assert greeks['gamma'] > 0
        
        # Vega should be positive
        assert greeks['vega'] > 0
    
    def test_calculate_greeks_put(self):
        """Test Greeks calculation for put option."""
        model = BlackScholesModel(risk_free_rate=0.02)
        
        greeks = model.calculate_greeks(
            option_type=OptionType.PUT,
            spot_price=100.0,
            strike=100.0,
            time_to_expiry=1.0,
            volatility=0.20
        )
        
        # For ATM put, delta should be around -0.5
        assert -0.6 < greeks['delta'] < -0.4
        
        # Gamma should be positive (same as call)
        assert greeks['gamma'] > 0


class TestOptionStrategy:
    """Test option trading strategies."""
    
    def test_covered_call(self):
        """Test covered call strategy."""
        model = BlackScholesModel(risk_free_rate=0.02)
        strategy = OptionStrategy(model)
        
        result = strategy.covered_call(
            spot_price=100.0,
            strike=105.0,
            time_to_expiry=0.25,
            volatility=0.20,
            shares=100
        )
        
        assert 'stock_value' in result
        assert 'call_premium_received' in result
        assert 'max_profit' in result
        assert 'breakeven' in result
        
        assert result['stock_value'] == 10000.0
        assert result['call_premium_received'] > 0
    
    def test_protective_put(self):
        """Test protective put strategy."""
        model = BlackScholesModel(risk_free_rate=0.02)
        strategy = OptionStrategy(model)
        
        result = strategy.protective_put(
            spot_price=100.0,
            strike=95.0,
            time_to_expiry=0.25,
            volatility=0.20,
            shares=100
        )
        
        assert 'stock_value' in result
        assert 'put_premium_paid' in result
        assert 'max_loss' in result
        
        assert result['put_premium_paid'] > 0
        assert result['max_loss'] > 0
    
    def test_straddle(self):
        """Test long straddle strategy."""
        model = BlackScholesModel(risk_free_rate=0.02)
        strategy = OptionStrategy(model)
        
        result = strategy.straddle(
            spot_price=100.0,
            strike=100.0,
            time_to_expiry=0.25,
            volatility=0.20,
            contracts=1
        )
        
        assert 'call_premium' in result
        assert 'put_premium' in result
        assert 'total_cost' in result
        assert 'upper_breakeven' in result
        assert 'lower_breakeven' in result
        
        # Total cost should be sum of both premiums
        assert result['total_cost'] == result['call_premium'] + result['put_premium']
    
    def test_iron_condor(self):
        """Test iron condor strategy."""
        model = BlackScholesModel(risk_free_rate=0.02)
        strategy = OptionStrategy(model)
        
        result = strategy.iron_condor(
            spot_price=100.0,
            lower_put_strike=90.0,
            upper_put_strike=95.0,
            lower_call_strike=105.0,
            upper_call_strike=110.0,
            time_to_expiry=0.25,
            volatility=0.20,
            contracts=1
        )
        
        assert 'net_credit' in result
        assert 'max_profit' in result
        assert 'max_loss' in result
        
        # Max profit should equal net credit
        assert result['max_profit'] == result['net_credit']


class TestFuturesCalculator:
    """Test futures calculator."""
    
    def test_calculator_init(self):
        """Test calculator initialization."""
        calc = FuturesCalculator(margin_rate=0.10)
        assert calc.margin_rate == 0.10
    
    def test_calculate_margin(self):
        """Test margin calculation."""
        calc = FuturesCalculator(margin_rate=0.10)
        
        contract = FuturesContract(
            symbol="ESZ3",
            underlying="ES",
            contract_type=FuturesType.INDEX,
            expiry=pd.Timestamp('2023-12-15'),
            contract_size=50,
            price=4500.0
        )
        
        margin = calc.calculate_margin(contract, quantity=2)
        
        # Margin = 2 * 4500 * 50 * 0.10 = 45000
        assert margin == 45000.0
    
    def test_calculate_pnl_long(self):
        """Test P&L calculation for long position."""
        calc = FuturesCalculator()
        
        entry_price = 4500.0
        current_price = 4550.0
        contract_size = 50
        quantity = 2
        
        pnl = calc.calculate_pnl(entry_price, current_price, contract_size, quantity)
        
        # P&L = (4550 - 4500) * 50 * 2 = 5000
        assert pnl == 5000.0
    
    def test_calculate_pnl_short(self):
        """Test P&L calculation for short position."""
        calc = FuturesCalculator()
        
        entry_price = 4500.0
        current_price = 4450.0
        contract_size = 50
        quantity = -2
        
        pnl = calc.calculate_pnl(entry_price, current_price, contract_size, quantity)
        
        # P&L = (4450 - 4500) * 50 * -2 = 5000
        assert pnl == 5000.0
    
    def test_calculate_basis(self):
        """Test basis calculation."""
        calc = FuturesCalculator()
        
        spot_price = 100.0
        futures_price = 102.0
        
        basis = calc.calculate_basis(spot_price, futures_price)
        
        assert basis == -2.0
    
    def test_calculate_carry_cost(self):
        """Test cost of carry calculation."""
        calc = FuturesCalculator()
        
        spot_price = 100.0
        risk_free_rate = 0.02
        dividend_yield = 0.01
        time_to_expiry = 1.0
        
        fair_price = calc.calculate_carry_cost(
            spot_price, risk_free_rate, dividend_yield, time_to_expiry
        )
        
        # Fair price = spot * e^((r - q) * T)
        expected = 100.0 * np.exp((0.02 - 0.01) * 1.0)
        assert abs(fair_price - expected) < 0.01


class TestDerivativesPortfolio:
    """Test derivatives portfolio management."""
    
    def test_portfolio_init(self):
        """Test portfolio initialization."""
        portfolio = DerivativesPortfolio(initial_cash=1_000_000)
        
        assert portfolio.cash == 1_000_000
        assert len(portfolio.options) == 0
        assert len(portfolio.futures) == 0
    
    def test_add_option_buy(self):
        """Test adding a long option position."""
        portfolio = DerivativesPortfolio(initial_cash=100_000)
        
        option = Option(
            symbol="AAPL230915C00150000",
            underlying="AAPL",
            option_type=OptionType.CALL,
            strike=150.0,
            expiry=pd.Timestamp('2023-09-15')
        )
        
        success = portfolio.add_option(option, quantity=1, premium=5.0)
        
        assert success
        assert len(portfolio.options) == 1
        # Cost = 5.0 * 1 * 100 = 500
        assert portfolio.cash == 100_000 - 500
    
    def test_add_option_sell(self):
        """Test adding a short option position."""
        portfolio = DerivativesPortfolio(initial_cash=100_000)
        
        option = Option(
            symbol="AAPL230915C00150000",
            underlying="AAPL",
            option_type=OptionType.CALL,
            strike=150.0,
            expiry=pd.Timestamp('2023-09-15')
        )
        
        success = portfolio.add_option(option, quantity=-1, premium=5.0)
        
        assert success
        # Credit = 5.0 * 1 * 100 = 500
        assert portfolio.cash == 100_000 + 500
    
    def test_add_option_insufficient_cash(self):
        """Test adding option with insufficient cash."""
        portfolio = DerivativesPortfolio(initial_cash=100)
        
        option = Option(
            symbol="AAPL230915C00150000",
            underlying="AAPL",
            option_type=OptionType.CALL,
            strike=150.0,
            expiry=pd.Timestamp('2023-09-15')
        )
        
        success = portfolio.add_option(option, quantity=1, premium=5.0)
        
        assert not success
        assert portfolio.cash == 100  # Cash unchanged
    
    def test_add_futures(self):
        """Test adding futures position."""
        portfolio = DerivativesPortfolio(initial_cash=100_000)
        
        contract = FuturesContract(
            symbol="ESZ3",
            underlying="ES",
            contract_type=FuturesType.INDEX,
            expiry=pd.Timestamp('2023-12-15'),
            contract_size=50,
            price=4500.0
        )
        
        margin_required = 10_000
        success = portfolio.add_futures(contract, quantity=1, margin_required=margin_required)
        
        assert success
        assert len(portfolio.futures) == 1
        assert portfolio.cash == 100_000 - margin_required
    
    def test_calculate_portfolio_value(self):
        """Test portfolio value calculation."""
        portfolio = DerivativesPortfolio(initial_cash=100_000)
        
        # Add a call option
        option = Option(
            symbol="AAPL230915C00150000",
            underlying="AAPL",
            option_type=OptionType.CALL,
            strike=150.0,
            expiry=pd.Timestamp('2023-09-15')
        )
        portfolio.add_option(option, quantity=1, premium=5.0)
        
        # Calculate value
        underlying_prices = {"AAPL": 155.0}
        volatilities = {"AAPL": 0.20}
        current_time = pd.Timestamp('2023-06-15')
        
        portfolio_value = portfolio.calculate_portfolio_value(
            underlying_prices, volatilities, current_time
        )
        
        assert isinstance(portfolio_value, float)
        assert portfolio_value > 0


def test_option_data_structure():
    """Test Option data structure."""
    option = Option(
        symbol="AAPL230915C00150000",
        underlying="AAPL",
        option_type=OptionType.CALL,
        strike=150.0,
        expiry=pd.Timestamp('2023-09-15'),
        premium=5.0,
        quantity=1
    )
    
    assert option.symbol == "AAPL230915C00150000"
    assert option.underlying == "AAPL"
    assert option.option_type == OptionType.CALL
    assert option.strike == 150.0
    assert option.premium == 5.0


def test_futures_data_structure():
    """Test FuturesContract data structure."""
    contract = FuturesContract(
        symbol="ESZ3",
        underlying="ES",
        contract_type=FuturesType.INDEX,
        expiry=pd.Timestamp('2023-12-15'),
        contract_size=50,
        price=4500.0,
        quantity=2
    )
    
    assert contract.symbol == "ESZ3"
    assert contract.underlying == "ES"
    assert contract.contract_type == FuturesType.INDEX
    assert contract.contract_size == 50
    assert contract.quantity == 2
