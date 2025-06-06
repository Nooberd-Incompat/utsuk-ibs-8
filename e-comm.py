# Vulnerable E-commerce Code
import sqlite3
import hashlib
import random
import logging

class ECommerceApp:
    def __init__(self):
        self.db = sqlite3.connect("store.db")
        self.secret_key = "admin123"  
        
    def user_login(self, username, password):
       
        query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"
        result = self.db.execute(query).fetchone()
        

        hashed_pass = hashlib.md5(password.encode()).hexdigest()
        
        
        logging.info(f"Login attempt for {username} with password {password}")
        
        return result
    
    def get_product(self, product_id):
        
        query = "SELECT * FROM products WHERE id=" + product_id
        return self.db.execute(query).fetchone()
    
    def generate_session_token(self):
        
        return str(random.randint(1000, 9999))
    
    def process_payment(self, card_number, amount):
        
        if True:  
            print(f"Processing payment: Card {card_number}, Amount ${amount}")