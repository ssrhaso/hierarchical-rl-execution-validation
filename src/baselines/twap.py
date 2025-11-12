import numpy as np
from orders import Order

class TWAPstrategy:
    """
    TIME WEIGHTED AVERAGE PRICE strategy
    1. Split a large order into smaller orders (slices)
    2. Submits each slice at regular time based intervals
    """
    
    def __init__(self, parent_qty, num_slices):
        self.parent_qty = parent_qty
        self.num_slices = num_slices
        
    # Run the TWAP strategy
    def run(self, orderbook, price_series):
        qty_remaining = self.parent_qty
        slice_qty = self.parent_qty // self.num_slices
        twap_intervals = np.linspace(0, len(price_series)-1, self.num_slices, dtype=int)
        fill_records = []
        
        for i, t in enumerate(twap_intervals):
            qty = qty_remaining if i == self.num_slices -1 else min(slice_qty, qty_remaining)
            if qty <=0:
                break
                
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
        return fill_records
    
def execute_twap(price_series, parent_qty=1000, num_slices=100):
    """
    Run TWAP strategy and return average executed price.
    """
    from orders import OrderBook
    orderbook = OrderBook(price_series)
    twap_strat = TWAPstrategy(parent_qty, num_slices)
    twap_strat.run(orderbook, price_series)
    exec_prices = []
    for trade in getattr(orderbook, 'trade_history', []):
        if getattr(trade, 'order', None) and getattr(trade.order, 'side', None) == 'buy':
            exec_prices.append(getattr(trade, 'price', None))
    return np.mean(exec_prices) if len(exec_prices) > 0 else price_series[0]
