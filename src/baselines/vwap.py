import numpy as np
from orders import Order

class VWAPstrategy:
    """
    VOLUME WEIGHTED AVERAGE PRICE strategy
    Split orders into smaller slices based on simulated volume profile
    """
    
    def __init__(self, parent_qty, num_slices):
        self.parent_qty = parent_qty
        self.num_slices = num_slices
    
    # Run the VWAP strategy
    def run(self, orderbook, price_series):
        np.random.seed(42)
        vol_profile = np.abs(np.random.normal(1, 0.5, len(price_series)))
        vol_profile /= vol_profile.sum()
        shares_remaining = self.parent_qty
        fill_records = []
        
        for i, v in enumerate(vol_profile):
            qty = int(round(v*self.parent_qty))
            qty = min(qty, shares_remaining)
            
            if qty > 0:
                
                # PLACE BUY ORDER
                order = Order(side='buy', quantity=qty, order_type='market', price=None, source='vwap')
                orderbook.set_time(i)
                orderbook.add_order(order=order)
                shares_remaining -= qty
                fill_records.append({'time': i, 'quantity': qty})
                
                # SIMULATE SELL ORDER (LIQUIDITY)
                sell_qty = np.random.randint(10, 101) # Random sell quantity between 10 and 100
                order = Order(side='sell', quantity=sell_qty, order_type='market', price=None, source='vwap')
                orderbook.set_time(i)
                orderbook.add_order(order=order)
            
            if shares_remaining <= 0:
                break
        return fill_records
    
    
def execute_vwap(price_series, parent_qty=1000, num_slices=100):
    """
    Run VWAP strategy and return average executed price.
    """
    from orders import OrderBook
    orderbook = OrderBook(price_series)
    vwap_strat = VWAPstrategy(parent_qty, num_slices)
    vwap_strat.run(orderbook, price_series)
    exec_prices = []
    for trade in getattr(orderbook, 'trade_history', []):
        if getattr(trade, 'order', None) and getattr(trade.order, 'side', None) == 'buy':
            exec_prices.append(getattr(trade, 'price', None))
    return np.mean(exec_prices) if len(exec_prices) > 0 else price_series[0]