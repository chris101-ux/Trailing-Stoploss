"""
MetaTrader5 ATR-Based Multi-Level Trailing Stop Manager
Dynamically adjusts stop loss based on ATR and profit levels for maximum gains
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import logging
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

# ================== CONFIGURATION ==================

# Symbol Configuration
SYMBOL = "XAUUSD"
CHECK_INTERVAL_SECONDS = 30  # Check every 30 seconds

# ATR Configuration
ATR_PERIOD = 14
ATR_TIMEFRAME = mt5.TIMEFRAME_M5  # 5-minute timeframe for ATR calculation
ATR_SMOOTHING = 1.0  # Smoothing factor for ATR (1.0 = no smoothing)

# Initial Stop Loss
INITIAL_STOP_ATR_MULTIPLIER = 1.5  # Initial SL at 1.5x ATR

# Profit Level Configuration (in ATR multiples)
PROFIT_LEVELS = [
    {'min_profit_atr': 0.0,  'trail_atr': None},    # No trailing until 0.5 ATR profit
    {'min_profit_atr': 0.5,  'trail_atr': 0.0},     # Breakeven at 0.5 ATR profit
    {'min_profit_atr': 1.0,  'trail_atr': 0.5},     # Trail with 0.5 ATR at 1 ATR profit
    {'min_profit_atr': 2.0,  'trail_atr': 1.0},     # Trail with 1.0 ATR at 2 ATR profit
    {'min_profit_atr': 3.0,  'trail_atr': 1.5},     # Trail with 1.5 ATR at 3 ATR profit
    {'min_profit_atr': 5.0,  'trail_atr': 2.0},     # Trail with 2.0 ATR at 5 ATR profit
]

# Volatility Adjustment
VOLATILITY_ADJUSTMENT = True
HIGH_VOLATILITY_MULTIPLIER = 1.2  # Increase trail distance by 20% in high volatility
LOW_VOLATILITY_MULTIPLIER = 0.8   # Decrease trail distance by 20% in low volatility

# Time-based Adjustments
TIME_BASED_ADJUSTMENT = True
NEWS_PROTECTION_MINUTES = 30  # Tighten stops 30 minutes before major news
WEEKEND_PROTECTION = True  # Tighten stops on Friday

# Performance Settings
MIN_POINTS_TO_UPDATE = 10  # Minimum points change before updating SL
MAX_SLIPPAGE = 20  # Maximum allowed slippage in points

# ================== DATA STRUCTURES ==================

@dataclass
class SymbolInfo:
    """Store symbol-specific information."""
    point: float
    tick_size: float
    stops_level: int
    digits: int
    spread: float

@dataclass
class PositionAnalysis:
    """Analysis data for a position."""
    position: any
    current_price: float
    entry_price: float
    profit_points: float
    profit_atr: float
    current_sl: float
    suggested_sl: float
    trailing_distance_atr: float
    should_update: bool

# ================== LOGGING SETUP ==================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ================== MT5 FUNCTIONS ==================

def initialize_mt5() -> bool:
    """Initialize MT5 connection."""
    if not mt5.initialize():
        logger.error(f"MT5 initialization failed, error code: {mt5.last_error()}")
        return False
    
    logger.info("MT5 initialized successfully")
    account_info = mt5.account_info()
    if account_info:
        logger.info(f"Connected to account: {account_info.login} ({account_info.server})")
        logger.info(f"Account balance: {account_info.balance} {account_info.currency}")
    return True

def get_symbol_info(symbol: str) -> Optional[SymbolInfo]:
    """Get symbol specifications."""
    info = mt5.symbol_info(symbol)
    if info is None:
        logger.error(f"Failed to get symbol info for {symbol}")
        return None
    
    tick = mt5.symbol_info_tick(symbol)
    spread = 0
    if tick:
        spread = tick.ask - tick.bid
    
    return SymbolInfo(
        point=info.point,
        tick_size=info.trade_tick_size,
        stops_level=info.trade_stops_level,
        digits=info.digits,
        spread=spread
    )

# ================== ATR CALCULATION ==================

def calculate_atr(symbol: str, timeframe: int, period: int, smoothing: float = 1.0) -> Optional[float]:
    """
    Calculate Average True Range for the symbol.
    Returns ATR in points (not pips).
    """
    # Get historical data
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, period + 1)
    
    if rates is None or len(rates) < period + 1:
        logger.error(f"Failed to get historical data for ATR calculation")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(rates)
    
    # Calculate True Range
    df['hl'] = df['high'] - df['low']
    df['hc'] = abs(df['high'] - df['close'].shift(1))
    df['lc'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['hl', 'hc', 'lc']].max(axis=1)
    
    # Calculate ATR
    atr_value = df['tr'].rolling(window=period).mean().iloc[-1]
    
    # Apply smoothing if needed
    if smoothing != 1.0:
        # Get previous ATR for smoothing
        prev_atr = df['tr'].rolling(window=period).mean().iloc[-2]
        atr_value = (atr_value * smoothing + prev_atr * (1 - smoothing))
    
    return atr_value

def get_average_atr(symbol: str, timeframe: int, period: int, lookback: int = 50) -> Optional[float]:
    """Calculate average ATR over a longer period for volatility comparison."""
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, lookback + period)
    
    if rates is None or len(rates) < lookback + period:
        return None
    
    df = pd.DataFrame(rates)
    
    # Calculate TR for all periods
    df['hl'] = df['high'] - df['low']
    df['hc'] = abs(df['high'] - df['close'].shift(1))
    df['lc'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['hl', 'hc', 'lc']].max(axis=1)
    
    # Calculate rolling ATR
    df['atr'] = df['tr'].rolling(window=period).mean()
    
    # Get average of last 'lookback' ATR values
    return df['atr'].iloc[-lookback:].mean()

# ================== POSITION ANALYSIS ==================

def analyze_position(position, symbol_info: SymbolInfo, current_atr: float) -> PositionAnalysis:
    """Analyze a position and determine optimal stop loss."""
    position_type = position.type
    
    # Get current price
    tick = mt5.symbol_info_tick(position.symbol)
    if position_type == 0:  # BUY
        current_price = tick.bid
        profit_points = current_price - position.price_open
    else:  # SELL
        current_price = tick.ask
        profit_points = position.price_open - current_price
    
    # Calculate profit in ATR multiples
    profit_atr = profit_points / current_atr if current_atr > 0 else 0
    
    # Determine trailing distance based on profit level
    trailing_distance_atr = get_trailing_distance_for_profit(profit_atr)
    
    # Calculate suggested stop loss
    if trailing_distance_atr is None:
        # No trailing yet, keep initial stop or current stop
        suggested_sl = position.sl if position.sl > 0 else calculate_initial_stop(
            position, symbol_info, current_atr
        )
    else:
        # Calculate new trailing stop
        if position_type == 0:  # BUY
            suggested_sl = current_price - (trailing_distance_atr * current_atr)
        else:  # SELL
            suggested_sl = current_price + (trailing_distance_atr * current_atr)
        
        # Apply volatility adjustments if enabled
        if VOLATILITY_ADJUSTMENT:
            suggested_sl = apply_volatility_adjustment(
                suggested_sl, current_price, position_type, 
                symbol_info.symbol, current_atr
            )
    
    # Round to valid tick size
    suggested_sl = round(suggested_sl / symbol_info.tick_size) * symbol_info.tick_size
    
    # Determine if update is needed
    should_update = should_update_stop_loss(
        position, suggested_sl, symbol_info.point * MIN_POINTS_TO_UPDATE
    )
    
    return PositionAnalysis(
        position=position,
        current_price=current_price,
        entry_price=position.price_open,
        profit_points=profit_points,
        profit_atr=profit_atr,
        current_sl=position.sl,
        suggested_sl=suggested_sl,
        trailing_distance_atr=trailing_distance_atr,
        should_update=should_update
    )

def get_trailing_distance_for_profit(profit_atr: float) -> Optional[float]:
    """Get trailing distance in ATR based on profit level."""
    for level in reversed(PROFIT_LEVELS):
        if profit_atr >= level['min_profit_atr']:
            return level['trail_atr']
    return None

def calculate_initial_stop(position, symbol_info: SymbolInfo, current_atr: float) -> float:
    """Calculate initial stop loss if not set."""
    stop_distance = current_atr * INITIAL_STOP_ATR_MULTIPLIER
    
    if position.type == 0:  # BUY
        return position.price_open - stop_distance
    else:  # SELL
        return position.price_open + stop_distance

def should_update_stop_loss(position, new_sl: float, min_change: float) -> bool:
    """Check if stop loss should be updated."""
    if position.sl == 0:
        return True
    
    if position.type == 0:  # BUY
        # Only move stop up, never down
        return new_sl > position.sl and (new_sl - position.sl) >= min_change
    else:  # SELL
        # Only move stop down, never up
        return new_sl < position.sl and (position.sl - new_sl) >= min_change

# ================== ADJUSTMENTS ==================

def apply_volatility_adjustment(
    suggested_sl: float, 
    current_price: float, 
    position_type: int,
    symbol: str,
    current_atr: float
) -> float:
    """Adjust stop loss based on current volatility."""
    avg_atr = get_average_atr(symbol, ATR_TIMEFRAME, ATR_PERIOD, lookback=50)
    
    if avg_atr is None:
        return suggested_sl
    
    volatility_ratio = current_atr / avg_atr
    
    if volatility_ratio > 1.3:  # High volatility
        multiplier = HIGH_VOLATILITY_MULTIPLIER
        logger.debug(f"High volatility detected (ratio: {volatility_ratio:.2f}), adjusting stops")
    elif volatility_ratio < 0.7:  # Low volatility
        multiplier = LOW_VOLATILITY_MULTIPLIER
        logger.debug(f"Low volatility detected (ratio: {volatility_ratio:.2f}), tightening stops")
    else:
        return suggested_sl
    
    # Adjust the distance from current price
    if position_type == 0:  # BUY
        distance = current_price - suggested_sl
        new_distance = distance * multiplier
        return current_price - new_distance
    else:  # SELL
        distance = suggested_sl - current_price
        new_distance = distance * multiplier
        return current_price + new_distance

def apply_time_based_adjustments(suggested_sl: float, current_price: float, position_type: int) -> float:
    """Apply time-based adjustments (news, weekend protection)."""
    if not TIME_BASED_ADJUSTMENT:
        return suggested_sl
    
    current_time = datetime.now()
    tighten_factor = 1.0
    
    # Weekend protection (Friday after 18:00)
    if WEEKEND_PROTECTION and current_time.weekday() == 4 and current_time.hour >= 18:
        tighten_factor = 0.7
        logger.debug("Weekend protection active, tightening stops")
    
    # Apply tightening
    if tighten_factor < 1.0:
        if position_type == 0:  # BUY
            distance = current_price - suggested_sl
            return current_price - (distance * tighten_factor)
        else:  # SELL
            distance = suggested_sl - current_price
            return current_price + (distance * tighten_factor)
    
    return suggested_sl

# ================== ORDER EXECUTION ==================

def update_stop_loss(position_ticket: int, new_sl: float, current_tp: float, symbol: str) -> bool:
    """Send request to update stop loss."""
    # Prepare request
    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "position": position_ticket,
        "symbol": symbol,
        "sl": new_sl,
        "tp": current_tp,
        "deviation": MAX_SLIPPAGE,
        "magic": 0,
        "comment": "ATR Trailing Stop"
    }
    
    # Send order
    result = mt5.order_send(request)
    
    if result is None:
        logger.error(f"order_send failed, no result returned")
        return False
    
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error(f"Failed to update SL. Retcode: {result.retcode}, Comment: {result.comment}")
        return False
    
    logger.info(f"✅ Stop Loss updated successfully to {new_sl:.5f}")
    return True

# ================== MAIN PROCESSING ==================

def process_positions(symbol: str, symbol_info: SymbolInfo) -> None:
    """Process all open positions for the symbol."""
    # Get current ATR
    current_atr = calculate_atr(symbol, ATR_TIMEFRAME, ATR_PERIOD, ATR_SMOOTHING)
    
    if current_atr is None:
        logger.error("Failed to calculate ATR, skipping this cycle")
        return
    
    logger.info(f"Current ATR: {current_atr:.5f} ({current_atr/symbol_info.point:.1f} pips)")
    
    # Get open positions
    positions = mt5.positions_get(symbol=symbol)
    
    if positions is None:
        logger.error(f"Failed to get positions, error: {mt5.last_error()}")
        return
    
    if len(positions) == 0:
        logger.info("No open positions found")
        return
    
    logger.info(f"Found {len(positions)} position(s)")
    logger.info("=" * 60)
    
    for pos in positions:
        # Analyze position
        analysis = analyze_position(pos, symbol_info, current_atr)
        
        # Log position details
        position_type = "BUY" if pos.type == 0 else "SELL"
        profit_pips = analysis.profit_points / symbol_info.point
        
        logger.info(f"Position #{pos.ticket} ({position_type}):")
        logger.info(f"  Entry: {analysis.entry_price:.5f} | Current: {analysis.current_price:.5f}")
        logger.info(f"  Profit: {profit_pips:.1f} pips ({analysis.profit_atr:.2f} ATR)")
        logger.info(f"  Current SL: {analysis.current_sl:.5f} | Suggested: {analysis.suggested_sl:.5f}")
        
        # Determine profit level
        level_desc = "No trailing"
        for level in reversed(PROFIT_LEVELS):
            if analysis.profit_atr >= level['min_profit_atr']:
                if level['trail_atr'] is None:
                    level_desc = "Below trailing threshold"
                elif level['trail_atr'] == 0:
                    level_desc = "Breakeven mode"
                else:
                    level_desc = f"Trailing with {level['trail_atr']:.1f} ATR"
                break
        
        logger.info(f"  Status: {level_desc}")
        
        # Update if needed
        if analysis.should_update:
            # Apply time-based adjustments
            final_sl = apply_time_based_adjustments(
                analysis.suggested_sl, 
                analysis.current_price, 
                pos.type
            )
            
            sl_change_pips = abs(final_sl - analysis.current_sl) / symbol_info.point
            logger.info(f"  📈 Updating SL (change: {sl_change_pips:.1f} pips)")
            
            success = update_stop_loss(pos.ticket, final_sl, pos.tp, symbol)
            if not success:
                logger.warning(f"  ⚠️ Failed to update stop loss")
        else:
            logger.info(f"  ✓ No update needed")
        
        logger.info("-" * 40)

def display_configuration():
    """Display current configuration."""
    logger.info("=" * 60)
    logger.info("ATR MULTI-LEVEL TRAILING STOP CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"Symbol: {SYMBOL}")
    logger.info(f"Check Interval: {CHECK_INTERVAL_SECONDS} seconds")
    logger.info(f"ATR Period: {ATR_PERIOD} | Timeframe: M{ATR_TIMEFRAME//60}")
    logger.info(f"Initial Stop: {INITIAL_STOP_ATR_MULTIPLIER}x ATR")
    logger.info("\nProfit Levels:")
    for level in PROFIT_LEVELS:
        if level['trail_atr'] is None:
            trail_desc = "No trailing"
        elif level['trail_atr'] == 0:
            trail_desc = "Breakeven"
        else:
            trail_desc = f"{level['trail_atr']}x ATR trail"
        logger.info(f"  At {level['min_profit_atr']}x ATR profit → {trail_desc}")
    logger.info(f"\nVolatility Adjustment: {'ON' if VOLATILITY_ADJUSTMENT else 'OFF'}")
    logger.info(f"Time-based Adjustment: {'ON' if TIME_BASED_ADJUSTMENT else 'OFF'}")
    logger.info("=" * 60)

# ================== MAIN LOOP ==================

def main():
    """Main trading loop."""
    try:
        # Display header
        logger.info("=" * 60)
        logger.info("🚀 ATR-BASED MULTI-LEVEL TRAILING STOP MANAGER")
        logger.info("=" * 60)
        
        # Initialize MT5
        if not initialize_mt5():
            return
        
        # Get symbol information
        symbol_info = get_symbol_info(SYMBOL)
        if symbol_info is None:
            mt5.shutdown()
            return
        
        logger.info(f"Symbol Point: {symbol_info.point} | Digits: {symbol_info.digits}")
        logger.info(f"Current Spread: {symbol_info.spread/symbol_info.point:.1f} pips")
        
        # Display configuration
        display_configuration()
        
        # Test ATR calculation
        test_atr = calculate_atr(SYMBOL, ATR_TIMEFRAME, ATR_PERIOD)
        if test_atr:
            logger.info(f"Initial ATR: {test_atr:.5f} ({test_atr/symbol_info.point:.1f} pips)")
        else:
            logger.error("Failed to calculate initial ATR, please check settings")
            mt5.shutdown()
            return
        
        logger.info("\n🔄 Starting main loop...\n")
        
        # Main loop
        cycle_count = 0
        while True:
            cycle_count += 1
            logger.info(f"\n{'='*60}")
            logger.info(f"Cycle #{cycle_count} - {datetime.now().strftime('%H:%M:%S')}")
            logger.info(f"{'='*60}")
            
            # Process positions
            process_positions(SYMBOL, symbol_info)
            
            # Wait for next cycle
            logger.info(f"\n⏰ Next check in {CHECK_INTERVAL_SECONDS} seconds...")
            time.sleep(CHECK_INTERVAL_SECONDS)
            
    except KeyboardInterrupt:
        logger.info("\n\n🛑 Trailing Stop Manager stopped by user")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}", exc_info=True)
    finally:
        mt5.shutdown()
        logger.info("📊 MT5 connection closed")
        logger.info("=" * 60)

if __name__ == "__main__":
    main()