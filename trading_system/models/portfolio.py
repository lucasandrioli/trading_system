from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from datetime import datetime

@dataclass
class TradeHistory:
    """Represents a trade history record."""
    price: float
    quantity: int
    date: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "price": self.price,
            "quantity": self.quantity,
            "date": self.date
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradeHistory':
        """Create from dictionary."""
        if not data:
            return None
        return cls(
            price=data.get("price", 0),
            quantity=data.get("quantity", 0),
            date=data.get("date", datetime.now().isoformat())
        )

@dataclass
class Position:
    """Represents a position in a portfolio."""
    symbol: str
    quantity: int
    avg_price: float
    current_price: float = 0.0
    current_position: float = 0.0
    last_buy: Optional[Dict[str, Any]] = None
    last_sell: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "avg_price": self.avg_price,
            "current_price": self.current_price,
            "current_position": self.current_position or (self.quantity * self.current_price),
            "last_buy": self.last_buy,
            "last_sell": self.last_sell
        }
    
    @property
    def profit_loss_pct(self) -> float:
        """Calculate profit/loss percentage."""
        if self.avg_price <= 0:
            return 0.0
        return ((self.current_price / self.avg_price) - 1) * 100
    
    @property
    def profit_loss_value(self) -> float:
        """Calculate profit/loss value."""
        return self.quantity * (self.current_price - self.avg_price)

@dataclass
class RecoveryGoal:
    """Represents recovery goals for a portfolio."""
    target_recovery: float
    days: int
    start_date: str = None
    
    def __post_init__(self):
        if not self.start_date:
            self.start_date = datetime.now().strftime("%Y-%m-%d")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "target_recovery": self.target_recovery,
            "days": self.days,
            "start_date": self.start_date
        }
    
    def copy(self) -> 'RecoveryGoal':
        """Create a copy of this goal."""
        return RecoveryGoal(
            target_recovery=self.target_recovery,
            days=self.days,
            start_date=self.start_date
        )
    
    @property
    def daily_target(self) -> float:
        """Calculate daily recovery target."""
        if self.days <= 0:
            return 0.0
        return self.target_recovery / self.days
    
    @property
    def days_passed(self) -> int:
        """Calculate days passed since start."""
        if not self.start_date:
            return 0
        
        start = datetime.strptime(self.start_date, "%Y-%m-%d")
        return (datetime.now() - start).days
    
    @property
    def days_remaining(self) -> int:
        """Calculate days remaining."""
        return max(0, self.days - self.days_passed)

@dataclass
class Portfolio:
    """Represents a complete trading portfolio."""
    positions: Dict[str, Position]
    watchlist: Dict[str, Any]
    account_balance: float
    goals: Optional[RecoveryGoal] = None
    
    def get_position(self, ticker: str) -> Optional[Position]:
        """Get a position by ticker."""
        return self.positions.get(ticker)
    
    def add_position(self, position: Position) -> None:
        """Add a new position or update existing one."""
        self.positions[position.symbol] = position
    
    def remove_position(self, ticker: str) -> bool:
        """Remove a position by ticker."""
        if ticker in self.positions:
            del self.positions[ticker]
            return True
        return False
    
    @property
    def portfolio_value(self) -> float:
        """Calculate total portfolio value."""
        return sum(pos.current_position for pos in self.positions.values())
    
    @property
    def total_value(self) -> float:
        """Calculate total value including cash."""
        return self.portfolio_value + self.account_balance
    
    @property
    def total_invested(self) -> float:
        """Calculate total amount invested."""
        return sum(pos.quantity * pos.avg_price for pos in self.positions.values())
    
    @property
    def profit_loss(self) -> float:
        """Calculate total profit/loss."""
        return self.portfolio_value - self.total_invested
    
    @property
    def profit_loss_pct(self) -> float:
        """Calculate profit/loss percentage."""
        if self.total_invested <= 0:
            return 0.0
        return (self.profit_loss / self.total_invested) * 100