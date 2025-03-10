import logging
import os
import json
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class NotificationService:
    """Service for sending notifications about trading events."""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.config_file = os.path.join(data_dir, 'notification_config.json')
        self._ensure_data_dir()
        
        # Load config
        self.config = self._load_config()
    
    def _ensure_data_dir(self):
        """Ensure data directory exists."""
        os.makedirs(self.data_dir, exist_ok=True)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load notification configuration from file."""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading notification config: {e}")
        
        # Default configuration
        default_config = {
            "enabled": False,
            "methods": {
                "email": {
                    "enabled": False,
                    "address": "",
                    "smtp_server": "smtp.gmail.com",
                    "smtp_port": 587,
                    "username": "",
                    "password": ""
                },
                "webhook": {
                    "enabled": False,
                    "url": ""
                }
            },
            "notify_trades": True,
            "notify_thresholds": True,
            "notify_rebalance": True,
            "threshold_value_min": 100.00
        }
        
        # Save default config
        try:
            with open(self.config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving default notification config: {e}")
        
        return default_config
    
    def get_config(self) -> Dict[str, Any]:
        """Get current notification configuration."""
        return self.config
    
    def update_config(self, new_config: Dict[str, Any]) -> Dict[str, Any]:
        """Update notification configuration."""
        try:
            # Update our in-memory config
            self.config.update(new_config)
            
            # Save to file
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            return {"success": True, "message": "Notification settings updated successfully"}
        except Exception as e:
            logger.error(f"Error updating notification config: {e}")
            return {"success": False, "message": str(e)}
    
    def notify_trade(self, ticker: str, action: str, quantity: int, price: float, reason: str = None) -> bool:
        """Send notification about a trade."""
        if not self.config.get("enabled") or not self.config.get("notify_trades"):
            logger.debug(f"Trade notification disabled: {action} {quantity} {ticker}")
            return False
        
        trade_value = quantity * price
        
        # Don't notify for small trades if threshold is set
        min_value = self.config.get("threshold_value_min", 0)
        if trade_value < min_value:
            logger.debug(f"Trade too small for notification: ${trade_value:.2f} < ${min_value:.2f}")
            return False
        
        # Build message
        subject = f"Trade Alert: {action} {quantity} {ticker}"
        
        body = f"""
        Trade Details:
        --------------
        Action: {action}
        Ticker: {ticker}
        Quantity: {quantity}
        Price: ${price:.2f}
        Total Value: ${trade_value:.2f}
        Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        if reason:
            body += f"\nReason: {reason}"
        
        # Send notification through configured methods
        result = self._send_notification(subject, body)
        
        logger.info(f"Trade notification sent: {action} {quantity} {ticker} @ ${price:.2f}")
        return result
    
    def notify_rebalance(self, rebalance_results: Dict[str, Any]) -> bool:
        """Send notification about portfolio rebalancing."""
        if not self.config.get("enabled") or not self.config.get("notify_rebalance"):
            return False
        
        sells = rebalance_results.get("sells_executed", [])
        buys = rebalance_results.get("buys_executed", [])
        
        if not sells and not buys:
            return False
        
        # Build message
        subject = f"Portfolio Rebalancing Completed"
        
        body = f"""
        Portfolio Rebalancing Results:
        -----------------------------
        Executed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        Summary:
        - Starting Balance: ${rebalance_results.get('starting_balance', 0):.2f}
        - Ending Balance: ${rebalance_results.get('ending_balance', 0):.2f}
        - Sells Executed: {len(sells)}
        - Buys Executed: {len(buys)}
        """
        
        if sells:
            body += "\nSells:\n"
            for sell in sells:
                body += f"- {sell.get('ticker')}: {sell.get('quantity')} shares @ ${sell.get('price', 0):.2f}\n"
        
        if buys:
            body += "\nBuys:\n"
            for buy in buys:
                body += f"- {buy.get('ticker')}: {buy.get('quantity')} shares @ ${buy.get('price', 0):.2f}\n"
        
        # Send notification
        result = self._send_notification(subject, body)
        
        logger.info(f"Rebalance notification sent: {len(sells)} sells, {len(buys)} buys")
        return result
    
    def notify_threshold(self, ticker: str, threshold_type: str, current_price: float, 
                        threshold_price: float, message: str = None) -> bool:
        """Send notification about price threshold events."""
        if not self.config.get("enabled") or not self.config.get("notify_thresholds"):
            return False
        
        # Build message
        subject = f"Price Alert: {ticker} {threshold_type.title()}"
        
        body = f"""
        Price Alert:
        ------------
        Ticker: {ticker}
        Alert Type: {threshold_type.title()}
        Current Price: ${current_price:.2f}
        Threshold Price: ${threshold_price:.2f}
        Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        if message:
            body += f"\nDetails: {message}"
        
        # Send notification
        result = self._send_notification(subject, body)
        
        logger.info(f"Threshold notification sent: {ticker} {threshold_type} @ ${current_price:.2f}")
        return result
    
    def _send_notification(self, subject: str, body: str) -> bool:
        """Send notification through all enabled methods."""
        success = False
        
        # Try email
        if self.config.get("methods", {}).get("email", {}).get("enabled"):
            email_success = self._send_email(subject, body)
            success = success or email_success
        
        # Try webhook
        if self.config.get("methods", {}).get("webhook", {}).get("enabled"):
            webhook_success = self._send_webhook(subject, body)
            success = success or webhook_success
        
        return success
    
    def _send_email(self, subject: str, body: str) -> bool:
        """Send notification via email."""
        email_config = self.config.get("methods", {}).get("email", {})
        
        try:
            to_address = email_config.get("address")
            smtp_server = email_config.get("smtp_server")
            smtp_port = email_config.get("smtp_port", 587)
            username = email_config.get("username")
            password = email_config.get("password")
            
            if not all([to_address, smtp_server, username, password]):
                logger.warning("Incomplete email configuration")
                return False
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = username
            msg['To'] = to_address
            msg['Subject'] = subject
            
            # Add body
            msg.attach(MIMEText(body, 'plain'))
            
            # Connect and send
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(username, password)
            server.send_message(msg)
            server.quit()
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending email notification: {e}")
            return False
    
    def _send_webhook(self, subject: str, body: str) -> bool:
        """Send notification via webhook (Discord, Slack, etc.)."""
        webhook_config = self.config.get("methods", {}).get("webhook", {})
        
        try:
            webhook_url = webhook_config.get("url")
            
            if not webhook_url:
                logger.warning("Incomplete webhook configuration")
                return False
            
            # Format for Discord/Slack
            payload = {
                "content": f"**{subject}**\n```{body}```"
            }
            
            # Send request
            response = requests.post(webhook_url, json=payload, timeout=5)
            response.raise_for_status()
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending webhook notification: {e}")
            return False