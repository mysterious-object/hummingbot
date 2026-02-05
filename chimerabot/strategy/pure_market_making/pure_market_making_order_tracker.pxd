# distutils: language=c++

from chimerabot.strategy.order_tracker import OrderTracker
from chimerabot.strategy.order_tracker cimport OrderTracker


cdef class PureMarketMakingOrderTracker(OrderTracker):
    pass
