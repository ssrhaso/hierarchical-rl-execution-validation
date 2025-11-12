import numpy as np
from orders import Order

class AdaptiveTWAP:
    """
    TIME WEIGHTED AVERAGE PRICE strategy
    1. Split a large order into smaller orders (slices)
    2. Submits each slice at time based intervals which vary according to market volatility 
    
    urgency : float (0 to 1) , higher means more urgent (faster trading)
    """

    def __init__(self, parent_qty, num_slices, vol_series, vol_threshold = 0.005, min_gap = 1, max_gap = 20, urgency = 0.5):
        self.parent_qty = parent_qty
        self.num_slices = num_slices
        self.vol_series = vol_series                # Time series of volatility values (from src/volatility.py)
        self.vol_threshold = vol_threshold          # Volatility threshold to adjust trade frequency
        self.min_gap = min_gap                      # Minimum steps between slices (fast)
        self.max_gap = max_gap                      # Maximum steps between slices (slow)
        self.urgency = urgency                    
        
        
        
    # Run the TWAP strategy
    def run(self, orderbook, price_series):
        qty_remaining = self.parent_qty
        slice_qty = self.parent_qty // self.num_slices
        fill_records = []
        t = 0
        step = 0
        
        while qty_remaining > 0 and t < len(price_series):
            # DECIDE TRADE GAP BASED ON VOLATILITY
            current_vol = self.vol_series[t]
            
            # Handle NaN volatility
            if np.isnan(current_vol): 
                current_vol = 0
            
            # LOW VOLATILITY , TRADE FASTER
            if current_vol < self.vol_threshold:
                gap =  int (round(self.max_gap - (self.max_gap - self.min_gap) * (self.urgency)))           # When urgency is 0, max gap is highest (slowest trading)
                # SMALL GAP(FAST TRADING) WHEN LOW VOLATILITY
                
                
            # HIGH VOLATILITY , SLOWER TRADING
            else:
                gap =  int (round(self.min_gap + (self.max_gap - self.min_gap) * (1 - self.urgency)))       # When urgency is 1, min gap is lowest (fastest trading)
                # BIG GAP(SLOW TRADING) WHEN HIGH VOLATILITY

    
            # FINAL SLICE ADJUSTMENT FOR LEFTOVER QTY
            qty = qty_remaining if step == self.num_slices -1 else min(slice_qty, qty_remaining)
            if qty <= 0:
                break
            
            # ORDERS
            
            # PLACE BUY ORDER
            orderbook.set_time(t)
            order = Order(side = 'buy', quantity = qty, order_type = 'market', source = 'twap')
            orderbook.add_order(order = order)
            qty_remaining -= qty
            fill_records.append({'time': t, 'quantity': qty})
            
            # SIMULATE SELL ORDER (LIQUIDITY)
            sell_qty = np.random.randint(10, 101) # Random sell quantity between 10 and 100
            order = Order(side = 'sell', quantity = sell_qty, order_type = 'market', source = 'twap')
            orderbook.set_time(t)
            orderbook.add_order(order = order)
            
            # TIME STEP
            t += gap
            step += 1
        return fill_records